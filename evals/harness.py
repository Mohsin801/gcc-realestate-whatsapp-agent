"""Conversation evaluation for the agent (spec Definition of Done).

Each case in ``evals/conversations.json`` is a scripted WhatsApp conversation for one lead. Every turn runs the
real LangGraph agent and is checked against:

Universal checks (every turn):
  * ``reply``        at least one non-empty bubble;
  * ``language``     every bubble is in the expected language (English or Arabic);
  * ``persona``      no AI/bot disclosure, in English or Arabic (while ``PERSONA_CONCEAL_AI`` is on);
  * ``bubbles``      at most ``REPLY_MAX_MESSAGES`` bubbles of ``REPLY_MAX_CHARS``;
  * ``grounded``     every number in the reply comes from the lead's messages or from the turn's prompt (verified
                     knowledge passages, the lead profile, the task). The code-written appendix (slot list, booking
                     time, Meet link) is exempt. Stricter than the persona guard, which only checks question turns;
  * ``latency``      the agent finished within ``LATENCY_BUDGET_SECONDS - LATENCY_SEND_RESERVE_SECONDS``, leaving the
                     send reserve for the Meta API (measured, not simulated, in live runs).

Per-turn expectations (``expect``):
  * ``language``     override of the case language for this turn;
  * ``state``        spec state fields after the turn (``budget_usd`` within 1%);
  * ``parts``        reply parts that must all be present (e.g. ``ASK``, ``OFFER_SLOTS``, ``BOOKED``);
  * ``parts_any``    at least one of these parts; ``not_parts``: none of these;
  * ``kb``           ``"defer"``: the question must be deferred to the call, with or without a knowledge base;
                     ``["doc-id", ...]``: with a knowledge base the reply must ANSWER and one of these facts must be
                     among the retrieved sources; without one it must DEFER.

Cases may declare ``"requires": ["llm"]`` (rules alone are not expected to handle them); such cases are skipped
when the agent runs without an LLM.
"""
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph
from agent.language import detect_language
from agent.nodes import numbers_in
from agent.persona_guard import reveals_ai
from agent.runner import AgentService
from agent.services import BookingResult, Knowledge
from core.config import Settings
from core.latency import LatencyStats, percentile
from core.timezones import lead_timezone
from services.inbound import IncomingMessage
from tools.scheduling import WorkingHours, candidate_slots, format_slot, parse_hhmm, pick_slots, to_iso

CASES_FILE = Path(__file__).resolve().parent / "conversations.json"
STATE_FIELDS = ("target_country", "intent", "budget_usd", "timeline", "is_qualified", "booking_status", "language")


# ---------------------------------------------------------------------------------------------- demo calendar
class DemoScheduler:
    """An empty calendar inside the configured working hours: the real slot finder and label formatter, with
    no Google account. Offers are kept per lead (phone), like ``CalendarScheduler``."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.offered: dict[str, list[datetime]] = {}
        self.booked: dict[str, datetime] = {}

    def _labels(self, phone: str, slots: list[datetime], language: str) -> list[str]:
        tz = lead_timezone(phone, self.settings.owner_timezone)
        now = datetime.now(UTC)
        return [format_slot(s, tz, language, now) for s in slots]

    async def offer_slots(self, phone, country, language):
        s = self.settings
        hours = WorkingHours(s.owner_timezone, tuple(s.working_days), parse_hhmm(s.work_start), parse_hhmm(s.work_end))
        duration = timedelta(minutes=s.slot_duration_minutes)
        cands = candidate_slots(datetime.now(UTC), hours, duration, timedelta(minutes=s.min_notice_minutes),
                                s.slot_search_days)
        taken = {to_iso(t) for t in self.booked.values()}
        self.offered[phone] = pick_slots(cands, [], taken, duration=duration, count=s.slot_count,
                                         per_day=s.slots_per_day, owner_tz=s.owner_timezone)
        return self._labels(phone, self.offered[phone], language)

    async def offered_labels(self, phone, language):
        return self._labels(phone, self.offered.get(phone, []), language)

    async def book(self, phone, lead_name, choice, language):
        slots = self.offered.get(phone, [])
        if not 1 <= choice <= len(slots):
            return BookingResult(ok=False, slot_taken=True)
        slot = slots[choice - 1]
        self.booked[phone] = slot
        self.offered.pop(phone, None)
        [label] = self._labels(phone, [slot], language)
        return BookingResult(ok=True, slot_iso=to_iso(slot), slot_label=label,
                             meet_link="https://meet.google.com/demo-demo-demo")

    async def add_email(self, phone, email):
        return phone in self.booked

    def prefetch(self) -> None:
        return None


# ---------------------------------------------------------------------------------------------- results
@dataclass
class TurnResult:
    case_id: str
    index: int
    user: str
    language: str
    replies: list[str]
    latency_ms: float
    failures: list[str]
    parts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    extraction_source: str | None = None
    reply_source: str | None = None
    budget_skips: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class CaseResult:
    case_id: str
    description: str
    turns: list[TurnResult]
    skipped: str | None = None

    @property
    def passed(self) -> bool:
        return self.skipped is None and all(t.passed for t in self.turns)


@dataclass
class Report:
    cases: list[CaseResult]
    budget_ms: float  # agent share of the latency budget (budget minus send reserve)
    mode: dict[str, Any]

    @property
    def run(self) -> list[CaseResult]:
        return [c for c in self.cases if c.skipped is None]

    @property
    def turns(self) -> list[TurnResult]:
        return [t for c in self.run for t in c.turns]

    @property
    def passed(self) -> bool:
        return bool(self.run) and all(c.passed for c in self.run)

    def latency(self, language: str | None = None) -> LatencyStats:
        return LatencyStats.of([t.latency_ms for t in self.turns if language in (None, t.language)], self.budget_ms)

    def as_dict(self) -> dict:
        turns = self.turns
        node_ms: dict[str, list[float]] = defaultdict(list)
        for t in turns:
            for node, ms in t.timings.items():
                node_ms[node].append(ms)
        return {
            "mode": self.mode,
            "passed": self.passed,
            "cases": {"total": len(self.cases), "run": len(self.run), "passed": sum(c.passed for c in self.run),
                      "skipped": [c.case_id for c in self.cases if c.skipped]},
            "turns": {"total": len(turns), "passed": sum(t.passed for t in turns)},
            "failed_checks": dict(Counter(f.split(":")[0] for t in turns for f in t.failures)),
            "latency_budget_ms": self.budget_ms,
            "latency": {k: vars(self.latency(lang)) for k, lang in (("all", None), ("en", "en"), ("ar", "ar"))},
            "reply_source": dict(Counter(t.reply_source for t in turns)),
            "extraction_source": dict(Counter(t.extraction_source for t in turns)),
            "budget_skips": dict(Counter(s for t in turns for s in t.budget_skips)),
            "node_p95_ms": {n: percentile(v, 95) for n, v in sorted(node_ms.items())},
            "failures": [
                {"case": t.case_id, "turn": t.index, "user": t.user, "replies": t.replies, "failures": t.failures}
                for t in turns if t.failures
            ],
        }


# ---------------------------------------------------------------------------------------------- checks
def check_turn(expect: dict, *, language: str, replies: list[str], turn: dict, state: dict, user_texts: list[str],
               settings: Settings, latency_ms: float, budget_ms: float, has_knowledge: bool) -> list[str]:
    """Return the failed checks for one turn, as ``"check: detail"`` strings (empty list = pass)."""
    failures: list[str] = []
    want_lang = expect.get("language", language)

    if not replies or not all(r and r.strip() for r in replies):
        failures.append("reply: empty reply")
    for i, bubble in enumerate(replies, 1):
        got = detect_language(bubble)
        if got not in (None, want_lang):
            failures.append(f"language: bubble {i} is {got}, expected {want_lang}")
        if settings.persona_conceal_ai and reveals_ai(bubble):
            failures.append(f"persona: bubble {i} reveals AI/bot")
        if len(bubble) > settings.reply_max_chars:
            failures.append(f"bubbles: bubble {i} has {len(bubble)} chars > {settings.reply_max_chars}")
    if len(replies) > settings.reply_max_messages:
        failures.append(f"bubbles: {len(replies)} bubbles > {settings.reply_max_messages}")

    text_bubbles = replies[:-1] if turn.get("appendix") and replies else replies
    allowed = numbers_in(turn.get("system", ""))
    for t in user_texts:
        allowed |= numbers_in(t)
    stray = sorted(set().union(*(numbers_in(b) for b in text_bubbles)) - allowed) if text_bubbles else []
    if stray:
        failures.append(f"grounded: figures not from the lead or verified knowledge: {', '.join(stray)}")

    if latency_ms > budget_ms:
        failures.append(f"latency: {latency_ms:.0f} ms > {budget_ms:.0f} ms")

    for key, want in (expect.get("state") or {}).items():
        got = state.get(key)
        if key == "budget_usd" and isinstance(want, (int, float)) and isinstance(got, (int, float)):
            ok = abs(got - want) <= 0.01 * want
        else:
            ok = got == want
        if not ok:
            failures.append(f"state: {key} is {got!r}, expected {want!r}")

    parts = [p["kind"] for p in turn.get("parts") or []]
    missing = [p for p in expect.get("parts") or [] if p not in parts]
    if missing:
        failures.append(f"parts: missing {missing} (got {parts})")
    if expect.get("parts_any") and not set(expect["parts_any"]) & set(parts):
        failures.append(f"parts: none of {expect['parts_any']} (got {parts})")
    unwanted = [p for p in expect.get("not_parts") or [] if p in parts]
    if unwanted:
        failures.append(f"parts: unexpected {unwanted}")

    kb = expect.get("kb")
    if kb == "defer":
        if "DEFER" not in parts:
            failures.append(f"knowledge: expected a deferral to the call (got {parts})")
    elif kb:
        if not has_knowledge:
            if "DEFER" not in parts:
                failures.append(f"knowledge: no knowledge base, expected a deferral (got {parts})")
        elif "ANSWER" not in parts:
            failures.append(f"knowledge: expected an answer from {kb} (got {parts})")
        else:
            got_ids = [s.get("doc_id") for s in turn.get("sources") or []]
            if not set(kb) & set(got_ids):
                failures.append(f"knowledge: retrieved {got_ids}, expected one of {kb}")
    return failures


# ---------------------------------------------------------------------------------------------- running
def load_cases(path: str | Path = CASES_FILE) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [c["id"] for c in cases]
    dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dupes:
        raise ValueError(f"duplicate case ids: {dupes}")
    return cases


def make_agent(settings: Settings, llm=None, knowledge: Knowledge | None = None, scheduler=None):
    graph = build_graph(llm=llm, settings=settings, scheduler=scheduler or DemoScheduler(settings),
                        knowledge=knowledge, checkpointer=InMemorySaver())
    return AgentService(graph, budget_seconds=settings.latency_budget_seconds), graph


async def run_case(case: dict, settings: Settings, *, llm=None, knowledge: Knowledge | None = None) -> CaseResult:
    missing = [r for r in case.get("requires") or [] if r == "llm" and llm is None]
    if missing:
        return CaseResult(case["id"], case.get("description", ""), [], skipped="needs an LLM")
    agent, graph = make_agent(settings, llm, knowledge)
    budget_ms = (settings.latency_budget_seconds - settings.latency_send_reserve_seconds) * 1000
    phone, name, language = case["phone"], case.get("name"), case["language"]
    config = {"configurable": {"thread_id": phone}}
    user_texts: list[str] = []
    results: list[TurnResult] = []
    for i, step in enumerate(case["turns"], 1):
        text = step["user"]
        expect = step.get("expect") or {}
        user_texts.append(text)
        turn_lang = expect.get("language") or detect_language(text) or language
        msg = IncomingMessage(f"wamid.eval.{case['id']}.{i}", phone, name, "text", text, turn_lang,
                              time.monotonic())
        try:
            replies = await agent(msg, turn_lang)
        except Exception as exc:  # noqa: BLE001 - report, keep evaluating
            replies, crash = [], f"crash: {exc.__class__.__name__}: {exc}"
        else:
            crash = None
        latency_ms = (time.monotonic() - msg.received_at) * 1000
        values = (await graph.aget_state(config)).values
        turn = values.get("turn") or {}
        failures = check_turn(expect, language=expect.get("language", language), replies=replies, turn=turn,
                              state=values, user_texts=user_texts, settings=settings, latency_ms=latency_ms,
                              budget_ms=budget_ms, has_knowledge=knowledge is not None)
        if crash:
            failures.insert(0, crash)
        results.append(TurnResult(
            case_id=case["id"], index=i, user=text, language=expect.get("language", language), replies=replies,
            latency_ms=latency_ms, failures=failures, parts=[p["kind"] for p in turn.get("parts") or []],
            sources=[s.get("doc_id") for s in turn.get("sources") or []],
            extraction_source=turn.get("extraction_source"), reply_source=turn.get("reply_source"),
            budget_skips=list(turn.get("budget_skips") or []), timings=dict(turn.get("timings") or {}),
        ))
    return CaseResult(case["id"], case.get("description", ""), results)


async def run_eval(settings: Settings, *, llm=None, knowledge: Knowledge | None = None,
                   cases: list[dict] | None = None, repeat: int = 1) -> Report:
    """Run every case ``repeat`` times, one after another (so latencies are not skewed by concurrency)."""
    results = []
    for _ in range(max(repeat, 1)):
        for case in cases if cases is not None else load_cases():
            results.append(await run_case(case, settings, llm=llm, knowledge=knowledge))
    budget_ms = (settings.latency_budget_seconds - settings.latency_send_reserve_seconds) * 1000
    mode = {"llm": llm is not None, "knowledge": knowledge is not None, "repeat": repeat,
            "persona_conceal_ai": settings.persona_conceal_ai}
    return Report(results, budget_ms, mode)
