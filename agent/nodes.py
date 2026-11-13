"""LangGraph nodes for the bilingual GCC real-estate agent.

Per turn:  detect_language -> extract -> normalize -> [answer_question] -> one step node -> respond -> persona_guard
Step nodes: qualify | offer_slots | book_slot | capture_email | follow_up.

The spec state (``GCCRealEstateState``) is persisted per phone number by the checkpointer. Per-turn
working data (the extraction, what to say, the draft, the final bubbles) lives under ``turn``, which the
runner resets at the start of every turn.

Every node is fail-safe: LLM errors fall back to rule-based extraction and template replies, and
calendar/knowledge errors fall back to a handoff or deferral, so the lead always gets a sensible answer.
"""
import logging
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent import templates as T
from agent.language import detect_language
from agent.llm import LLMProvider
from agent.normalize import (
    extract_email,
    looks_like_budget,
    match_country,
    match_intent,
    match_timeline,
    normalize_text,
    parse_amount,
    parse_slot_choice,
    resolve_budget,
    to_western_digits,
)
from agent.persona_guard import check_reply, reveals_ai, split_bubbles, to_whatsapp_format
from agent.prompts import EXTRACTION_SYSTEM, Extraction, extraction_context, reply_system_prompt
from agent.services import BookingResult, Knowledge, Scheduler
from agent.state import COUNTRIES, INTENTS, TIMELINES, compute_is_qualified
from core.config import Settings
from core.timezones import lead_timezone

log = logging.getLogger(__name__)

QUALIFY_ORDER = ("target_country", "intent", "budget_usd", "timeline")
QUESTION_INTENTS = {"FAQ_QUESTION", "PROPERTY_QUESTION"}
_FAQ_HINT = re.compile(
    r"\?|؟|\b(fee|fees|tax|vat|visa|residency|mortgage|law|legal|rules?|allowed|can foreigners|freehold|"
    r"transfer|registration|commission|how much|what is|what are)\b|رسوم|ضريب|فيزا|اقامه|قانون|نظام|تمويل|"
    r"رهن|يحق|مسموح|تملك|(^|\s)(كم|وش|ايش|شنو|هل)(\s|$)"
)
_GREETING_HINT = re.compile(
    r"^\s*(hi|hello|hey|salam|good (morning|evening|afternoon)|السلام|سلام|هلا|مرحبا|صباح|مساء)\b"
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _text(msg: BaseMessage) -> str:
    return msg.content if isinstance(msg.content, str) else str(msg.content)


def last_human_text(state: dict) -> str:
    for msg in reversed(state.get("messages") or []):
        if isinstance(msg, HumanMessage):
            return _text(msg)
    return ""


def first_name(config: RunnableConfig | None) -> str | None:
    name = ((config or {}).get("configurable") or {}).get("lead_name")
    return name.split()[0] if name and name.strip() else None


def _num(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER_RE.findall(to_western_digits(text))}


class AgentNodes:
    def __init__(self, *, llm: LLMProvider | None, settings: Settings, scheduler: Scheduler, knowledge: Knowledge):
        self.llm = llm
        self.settings = settings
        self.scheduler = scheduler
        self.knowledge = knowledge

    # ------------------------------------------------------------------ helpers
    def _history(self, state: dict) -> list[BaseMessage]:
        msgs = list(state.get("messages") or [])[-self.settings.llm_history_messages:]
        while msgs and not isinstance(msgs[0], HumanMessage):  # providers expect the user to speak first
            msgs.pop(0)
        return msgs

    @staticmethod
    def _turn(state: dict) -> dict[str, Any]:
        return dict(state.get("turn") or {})

    @staticmethod
    def _add_part(turn: dict, kind: str, **data) -> None:
        turn.setdefault("parts", []).append({"kind": kind, **data})

    # ------------------------------------------------------------------ 1. language
    async def detect_language(self, state: dict) -> dict:
        lang = detect_language(last_human_text(state)) or state.get("language") or "en"
        return {"language": lang}

    # ------------------------------------------------------------------ 2. extraction (intent classifier)
    async def extract(self, state: dict) -> dict:
        turn = self._turn(state)
        text = last_human_text(state)
        status = state.get("booking_status") or "NONE"
        offered: list[str] = []
        if status == "SLOTS_OFFERED":
            try:
                offered = await self.scheduler.offered_labels(state["phone_number"], state["language"])
            except Exception:  # noqa: BLE001
                log.warning("could not load offered slots", exc_info=True)
        turn["offered"] = offered

        extraction, source = self._fast_path(text, status, offered), "fast_path"
        if extraction is None and self.llm is not None:
            profile = {k: state.get(k) for k in QUALIFY_ORDER}
            tz = lead_timezone(state["phone_number"], self.settings.owner_timezone)
            now_text = datetime.now(ZoneInfo(tz)).strftime("%A %d %B %Y, %H:%M") + f" ({tz})"
            system = EXTRACTION_SYSTEM.replace("{context}", extraction_context(profile, status, offered, now_text))
            try:
                extraction, source = await self.llm.extract(system, self._history(state), Extraction), "llm"
            except Exception:  # noqa: BLE001 - LLM down/slow/invalid: rules take over
                log.warning("LLM extraction failed, using rules", exc_info=True)
        if extraction is None:
            extraction, source = self._rule_extract(text, state, offered), "rules"

        data = extraction.model_dump()
        email = extract_email(text)  # deterministic beats the LLM for emails
        if email:
            data["email"] = email
        if offered and data.get("slot_choice") is None:
            data["slot_choice"] = parse_slot_choice(text, len(offered))
        turn["extraction"] = data
        turn["extraction_source"] = source
        return {"turn": turn}

    @staticmethod
    def _fast_path(text: str, status: str, offered: list[str]) -> Extraction | None:
        """Skip the LLM for unambiguous replies (latency): a slot number or a bare email."""
        if status == "SLOTS_OFFERED" and offered:
            choice = parse_slot_choice(text, len(offered))
            if choice is not None:
                return Extraction(message_intent="SLOT_SELECTION", slot_choice=choice)
        email = extract_email(text)
        if email and len(text.strip()) - len(email) < 25:
            return Extraction(message_intent="EMAIL_PROVIDED", email=email)
        return None

    @staticmethod
    def _rule_extract(text: str, state: dict, offered: list[str]) -> Extraction:
        t = normalize_text(text)
        country, intent, timeline = match_country(text), match_intent(text), match_timeline(text)
        budget_text = None
        asked = next((f for f in QUALIFY_ORDER if state.get(f) is None), None)
        if parse_amount(text) is not None and (looks_like_budget(text) or asked == "budget_usd"):
            budget_text = text
        email = extract_email(text)
        slot = parse_slot_choice(text, len(offered)) if offered else None
        if email:
            kind = "EMAIL_PROVIDED"
        elif slot:
            kind = "SLOT_SELECTION"
        elif _FAQ_HINT.search(t) and not (intent or timeline or budget_text):
            kind = "FAQ_QUESTION"  # a country may come with the question ("DLD fees in Dubai?")
        elif country or intent or timeline or budget_text:
            kind = "PROVIDE_INFO"
        elif _GREETING_HINT.search(t):
            kind = "GREETING"
        else:
            kind = "OTHER"
        return Extraction(
            message_intent=kind, target_country=country, intent=intent, timeline=timeline,
            budget_text=budget_text, email=email, slot_choice=slot,
            question=text if kind == "FAQ_QUESTION" else None,
        )

    # ------------------------------------------------------------------ 3. normalisation
    async def normalize(self, state: dict) -> dict:
        turn = self._turn(state)
        ex = turn.get("extraction") or {}
        updates: dict[str, Any] = {}
        if ex.get("target_country") in COUNTRIES:
            updates["target_country"] = ex["target_country"]
        if ex.get("intent") in INTENTS:
            updates["intent"] = ex["intent"]
        if ex.get("timeline") in TIMELINES:
            updates["timeline"] = ex["timeline"]
        if ex.get("budget_text") or ex.get("budget_amount"):
            budget = resolve_budget(
                self.settings,
                text=ex.get("budget_text"),
                llm_amount=ex.get("budget_amount"),
                llm_currency=ex.get("budget_currency"),
                country=updates.get("target_country") or state.get("target_country"),
            )
            if budget and budget.usd > 0:
                updates["budget_usd"] = budget.usd
                turn["budget"] = {"amount": budget.amount, "currency": budget.currency, "usd": budget.usd}
        turn["provided"] = [k for k, v in updates.items() if state.get(k) != v]
        updates["is_qualified"] = compute_is_qualified({**state, **updates})
        updates["turn"] = turn
        return updates

    # ------------------------------------------------------------------ routing
    @staticmethod
    def route_after_normalize(state: dict) -> str:
        ex = (state.get("turn") or {}).get("extraction") or {}
        if ex.get("message_intent") in QUESTION_INTENTS:
            return "answer_question"
        return AgentNodes.next_step(state)

    @staticmethod
    def next_step(state: dict) -> str:
        ex = (state.get("turn") or {}).get("extraction") or {}
        status = state.get("booking_status") or "NONE"
        if status == "BOOKED":
            return "capture_email" if ex.get("email") else "follow_up"
        if status == "SLOTS_OFFERED":
            return "follow_up" if ex.get("message_intent") == "RESCHEDULE_OR_CANCEL" else "book_slot"
        if not state.get("is_qualified"):
            return "qualify"
        return "offer_slots"

    # ------------------------------------------------------------------ 4. knowledge (RAG)
    async def answer_question(self, state: dict) -> dict:
        turn = self._turn(state)
        ex = turn.get("extraction") or {}
        question = ex.get("question") or last_human_text(state)
        try:
            passages = await self.knowledge.search(question, state.get("target_country"), state["language"])
        except Exception:  # noqa: BLE001
            log.warning("knowledge search failed", exc_info=True)
            passages = []
        if passages:
            turn["knowledge"] = [p.text for p in passages]
            turn["sources"] = [p.source for p in passages if p.source]
            self._add_part(turn, "ANSWER", question=question)
        else:
            self._add_part(turn, "DEFER", question=question)
        return {"turn": turn}

    # ------------------------------------------------------------------ 5. step nodes
    async def qualify(self, state: dict, config: RunnableConfig) -> dict:
        turn = self._turn(state)
        ex = turn.get("extraction") or {}
        humans = sum(isinstance(m, HumanMessage) for m in state.get("messages") or [])
        if humans <= 1 or ex.get("message_intent") == "GREETING":
            self._add_part(turn, "GREETING", name=first_name(config))
        if turn.get("provided"):
            self._add_part(turn, "ACK_INFO")
        missing = next(f for f in QUALIFY_ORDER if state.get(f) is None)
        self._add_part(turn, "ASK", field=missing, booking_requested=ex.get("message_intent") == "BOOKING_REQUEST")
        return {"turn": turn}

    async def _offer(self, state: dict, turn: dict, kind: str = "OFFER_SLOTS") -> dict:
        try:
            labels = await self.scheduler.offer_slots(state["phone_number"], state.get("target_country"),
                                                      state["language"])
        except Exception:  # noqa: BLE001
            log.warning("could not fetch slots", exc_info=True)
            labels = []
        if labels:
            self._add_part(turn, kind, labels=labels)
            return {"booking_status": "SLOTS_OFFERED", "turn": turn}
        self._add_part(turn, "NO_SLOTS")
        return {"booking_status": "NONE", "turn": turn}

    async def offer_slots(self, state: dict) -> dict:
        turn = self._turn(state)
        if turn.get("provided"):
            self._add_part(turn, "ACK_INFO")
        return await self._offer(state, turn)

    async def book_slot(self, state: dict, config: RunnableConfig) -> dict:
        turn = self._turn(state)
        ex = turn.get("extraction") or {}
        offered = turn.get("offered") or []
        if not offered:  # offer expired or lost: fetch fresh times
            return await self._offer(state, turn)
        choice = ex.get("slot_choice")
        if not isinstance(choice, int) or not 1 <= choice <= len(offered):
            self._add_part(turn, "SLOT_UNCLEAR", labels=offered)
            return {"turn": turn}
        name = ((config or {}).get("configurable") or {}).get("lead_name")
        try:
            result = await self.scheduler.book(state["phone_number"], name, choice, state["language"])
        except Exception:  # noqa: BLE001
            log.exception("booking failed")
            result = BookingResult(ok=False)
        if result.ok:
            self._add_part(turn, "BOOKED", slot=result.slot_label or offered[choice - 1], link=result.meet_link)
            return {"booking_status": "BOOKED", "selected_slot": result.slot_iso, "turn": turn}
        if result.slot_taken:
            return await self._offer(state, turn, kind="SLOT_TAKEN")
        self._add_part(turn, "BOOKING_FAILED")
        return {"booking_status": "NONE", "turn": turn}

    async def capture_email(self, state: dict) -> dict:
        turn = self._turn(state)
        email = (turn.get("extraction") or {}).get("email")
        try:
            ok = await self.scheduler.add_email(state["phone_number"], email)
        except Exception:  # noqa: BLE001
            log.exception("adding attendee email failed")
            ok = False
        self._add_part(turn, "EMAIL_SAVED" if ok else "EMAIL_NOTED", email=email)
        return {"turn": turn}

    async def follow_up(self, state: dict) -> dict:
        turn = self._turn(state)
        ex = turn.get("extraction") or {}
        if ex.get("message_intent") == "RESCHEDULE_OR_CANCEL":
            self._add_part(turn, "RESCHEDULE")
        elif state.get("booking_status") == "BOOKED" and not turn.get("parts"):
            self._add_part(turn, "ALREADY_BOOKED")
        return {"turn": turn}

    # ------------------------------------------------------------------ 6. reply
    def _compose(self, parts: list[dict], language: str) -> tuple[list[str], list[str], str | None]:
        """Return (LLM task lines, template fallback lines, deterministic appendix bubble)."""
        tasks, fallback, appendix = [], [], None
        pick = lambda tpl: T.pick(tpl, language)  # noqa: E731

        def slot_list(labels: list[str]) -> str:
            lines = [f"{i}. {label}" for i, label in enumerate(labels, 1)]
            return "\n".join(lines) + "\n\n" + pick(T.SLOT_LIST_FOOTER)

        for p in parts:
            kind = p["kind"]
            if kind == "GREETING":
                name = p.get("name")
                tasks.append(f"Greet them warmly{f' by their first name ({name})' if name else ''}.")
                fallback.append(pick(T.GREETING).format(name=f" {name}" if name else ""))
            elif kind == "ACK_INFO":
                tasks.append("Briefly acknowledge what they just shared (a few words).")
            elif kind == "ANSWER":
                tasks.append(f"Answer their question ({p['question']}) using ONLY the KNOWLEDGE section.")
                fallback.append(pick(T.DEFER_QUESTION))
            elif kind == "DEFER":
                tasks.append(
                    f"They asked: {p['question']}. You have no verified information on this: say the consultant "
                    "will confirm the exact, up-to-date details on the call. Do not guess any figures."
                )
                fallback.append(pick(T.DEFER_QUESTION))
            elif kind == "ASK":
                q = T.ASK[p["field"]]
                if p.get("booking_requested"):
                    tasks.append("They want a call: say you'll set it up right after a couple of quick questions.")
                tasks.append(f'Ask this one question in your own words: "{q["en"]}"')
                fallback.append(pick(q))
            elif kind in ("OFFER_SLOTS", "SLOT_TAKEN", "SLOT_UNCLEAR"):
                lead_in = {"OFFER_SLOTS": T.OFFER_SLOTS, "SLOT_TAKEN": T.SLOT_TAKEN, "SLOT_UNCLEAR": T.SLOT_UNCLEAR}[kind]
                tasks.append(
                    {
                        "OFFER_SLOTS": "Say you'd like to set up a short video call with the senior consultant and "
                                       "ask which time suits them.",
                        "SLOT_TAKEN": "Apologise that the time they picked was just taken and ask them to choose "
                                      "another time.",
                        "SLOT_UNCLEAR": "Ask which of the offered times works for them.",
                    }[kind]
                    + " Do NOT write any times or dates: the numbered list is sent right after your message."
                )
                fallback.append(pick(lead_in))
                appendix = slot_list(p["labels"])
            elif kind == "NO_SLOTS":
                tasks.append("Thank them, say you have what you need and that the senior consultant will contact "
                             "them shortly to arrange a call.")
                fallback.append(pick(T.NO_SLOTS))
            elif kind == "BOOKED":
                tasks.append("Confirm their call is booked and ask for their email so you can send the calendar "
                             "invite. Do NOT write the time or link: they are sent right after your message.")
                fallback.append(pick(T.BOOKED))
                details = [pick(T.BOOKING_DETAILS).format(slot=p["slot"])]
                if p.get("link"):
                    details.append(pick(T.MEET_LINK).format(link=p["link"]))
                appendix = "\n".join(details)
            elif kind == "BOOKING_FAILED":
                tasks.append("Say you couldn't lock that time in just now and the consultant will confirm a time "
                             "with them shortly.")
                fallback.append(pick(T.BOOKING_FAILED))
            elif kind == "EMAIL_SAVED":
                tasks.append(f"Thank them and say the calendar invite is on its way to {p['email']}.")
                fallback.append(pick(T.EMAIL_SAVED).format(email=p["email"]))
            elif kind == "EMAIL_NOTED":
                tasks.append(f"Thank them, say you've noted {p['email']} and the consultant will make sure the "
                             "invite reaches them.")
                fallback.append(pick(T.EMAIL_NOTED).format(email=p["email"]))
            elif kind == "ALREADY_BOOKED":
                tasks.append("Reply to their message naturally; remind them their call is booked and ask if "
                             "there's anything else you can help with.")
                fallback.append(pick(T.ALREADY_BOOKED))
            elif kind == "RESCHEDULE":
                tasks.append("Say no problem: you'll let the consultant know and they'll reach out to arrange a "
                             "new time.")
                fallback.append(pick(T.RESCHEDULE))
        if not tasks:
            tasks.append("Reply naturally and briefly to their message.")
        if not fallback:
            fallback.append(pick(T.ACK))
        return tasks, fallback, appendix

    def _profile(self, state: dict, turn: dict, config: RunnableConfig | None) -> str:
        budget = turn.get("budget")
        bits = {
            "name": first_name(config) or "unknown",
            "country": state.get("target_country") or "unknown",
            "purpose": state.get("intent") or "unknown",
            "budget_usd": state.get("budget_usd") or "unknown",
            "timeline": state.get("timeline") or "unknown",
            "booking_status": state.get("booking_status") or "NONE",
        }
        if budget:
            bits["budget_stated"] = f"{budget['amount']:,.0f} {budget['currency']}"
        return ", ".join(f"{k}={v}" for k, v in bits.items())

    async def respond(self, state: dict, config: RunnableConfig) -> dict:
        turn = self._turn(state)
        language = state["language"]
        tasks, fallback, appendix = self._compose(turn.get("parts") or [], language)
        task_text = "\n".join(f"{i}. {t}" for i, t in enumerate(tasks, 1))
        knowledge = "\n".join(f"- {k}" for k in turn.get("knowledge") or [])
        system = reply_system_prompt(self.settings, language, task_text, self._profile(state, turn, config), knowledge)
        turn.update(task=task_text, fallback=" ".join(fallback), appendix=appendix, system=system)
        draft, source = None, "template"
        if self.llm is not None:
            try:
                draft, source = await self.llm.generate(system, self._history(state)), "llm"
            except Exception:  # noqa: BLE001
                log.warning("LLM reply failed, using template", exc_info=True)
        turn["draft"] = draft if draft else turn["fallback"]
        turn["draft_source"] = source if draft else "template"
        return {"turn": turn}

    # ------------------------------------------------------------------ 7. persona guard
    def _problems(self, draft: str, state: dict, turn: dict) -> list[str]:
        s = self.settings
        budget = s.reply_max_messages - (1 if turn.get("appendix") else 0)
        problems = check_reply(draft, conceal_ai=s.persona_conceal_ai, max_chars=s.reply_max_chars,
                               max_messages=max(budget, 1))
        draft_lang = detect_language(draft)
        if draft_lang and draft_lang != state["language"]:
            problems.append("wrong_language")
        if not turn.get("knowledge"):
            kinds = {p["kind"] for p in turn.get("parts") or []}
            if re.search(r"\d\s*%|٪|percent|بالمي[هة]", to_western_digits(draft).lower()):
                problems.append("ungrounded_figure")
            elif "DEFER" in kinds:
                allowed = _num(last_human_text(state)) | _num(turn.get("task", ""))
                if _num(draft) - allowed:
                    problems.append("ungrounded_figure")
        return problems

    async def persona_guard(self, state: dict) -> dict:
        turn = self._turn(state)
        s = self.settings
        draft = to_whatsapp_format(turn.get("draft") or "")
        problems = self._problems(draft, state, turn) if turn.get("draft_source") == "llm" else []
        if problems and self.llm is not None:
            log.info("reply rejected by persona guard", extra={"problems": problems})
            retry_system = (
                turn["system"] + f"\n\nYour previous draft was rejected ({', '.join(problems)}): \"{draft}\". "
                "Rewrite it so it follows every rule above, in fewer words."
            )
            try:
                retry = to_whatsapp_format(await self.llm.generate(retry_system, self._history(state)))
                problems = self._problems(retry, state, turn)
                draft = retry
            except Exception:  # noqa: BLE001
                log.warning("persona-guard regeneration failed", exc_info=True)
        if problems or not draft:
            if problems:
                log.warning("persona guard fell back to template", extra={"problems": problems})
            draft = turn["fallback"]
        if s.persona_conceal_ai and reveals_ai(draft):  # templates are vetted; this is a last line of defence
            draft = turn["fallback"]
        appendix = turn.get("appendix")
        text_budget = max(s.reply_max_messages - (1 if appendix else 0), 1)
        bubbles = split_bubbles(draft, s.reply_max_chars, text_budget)
        if appendix:
            bubbles.append(appendix)
        turn["replies"] = bubbles
        turn["guard_problems"] = problems
        return {"messages": [AIMessage("\n\n".join(bubbles))], "turn": turn}
