"""Chat with the agent in the terminal, without WhatsApp, Redis or Postgres.

    .\\.venv\\Scripts\\python.exe scripts\\chat_cli.py            # uses the LLM from .env (GEMINI_API_KEY, GEMINI_MODEL)
    .\\.venv\\Scripts\\python.exe scripts\\chat_cli.py --no-llm   # rules + templates only
    .\\.venv\\Scripts\\python.exe scripts\\chat_cli.py --slots    # fake calendar so the booking flow can be tried
    .\\.venv\\Scripts\\python.exe scripts\\chat_cli.py --kb       # answer questions from the Qdrant knowledge base
                                                               # (needs QDRANT_URL, GEMINI_EMBEDDING_MODEL and an ingest)

Type messages in English or Arabic; `/state` prints the session state, `/quit` exits.
"""
import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent.graph import build_graph  # noqa: E402
from agent.language import detect_language  # noqa: E402
from agent.llm import build_llm  # noqa: E402
from agent.runner import AgentService  # noqa: E402
from agent.services import BookingResult  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.timezones import lead_timezone  # noqa: E402
from services.inbound import IncomingMessage  # noqa: E402
from tools.rag import build_knowledge  # noqa: E402
from tools.scheduling import WorkingHours, candidate_slots, format_slot, parse_hhmm, pick_slots, to_iso  # noqa: E402


class DemoScheduler:
    """Empty calendar inside the configured working hours, for trying the booking flow without Google.
    Uses the real slot finder and label formatter (lead timezone from the phone country code)."""

    def __init__(self, settings):
        self.settings = settings
        self.offered: list = []

    def _labels(self, phone, language):
        tz = lead_timezone(phone, self.settings.owner_timezone)
        now = datetime.now(UTC)
        return [format_slot(s, tz, language, now) for s in self.offered]

    async def offer_slots(self, phone, country, language):
        s = self.settings
        hours = WorkingHours(s.owner_timezone, tuple(s.working_days), parse_hhmm(s.work_start), parse_hhmm(s.work_end))
        duration = timedelta(minutes=s.slot_duration_minutes)
        cands = candidate_slots(datetime.now(UTC), hours, duration, timedelta(minutes=s.min_notice_minutes),
                                s.slot_search_days)
        self.offered = pick_slots(cands, [], set(), duration=duration, count=s.slot_count,
                                  per_day=s.slots_per_day, owner_tz=s.owner_timezone)
        return self._labels(phone, language)

    async def offered_labels(self, phone, language):
        return self._labels(phone, language)

    async def book(self, phone, lead_name, choice, language):
        label = self._labels(phone, language)[choice - 1]
        return BookingResult(ok=True, slot_iso=to_iso(self.offered[choice - 1]), slot_label=label,
                             meet_link="https://meet.google.com/demo-demo-demo")

    async def add_email(self, phone, email):
        return True


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--slots", action="store_true", help="use a fake calendar to try booking")
    parser.add_argument("--kb", action="store_true", help="answer questions from the Qdrant knowledge base")
    parser.add_argument("--name", default="Ahmed")
    parser.add_argument("--phone", default="971500000000", help="lead number; its country code sets the timezone")
    args = parser.parse_args()

    settings = get_settings()
    knowledge, qdrant = build_knowledge(settings) if args.kb else (None, None)
    try:
        await chat(args, settings, knowledge)
    finally:
        if qdrant is not None:
            await qdrant.close()


async def chat(args, settings, knowledge) -> None:
    phone = args.phone
    llm = None if args.no_llm else build_llm(settings)
    graph = build_graph(llm=llm, settings=settings, scheduler=DemoScheduler(settings) if args.slots else None,
                        knowledge=knowledge, checkpointer=InMemorySaver())
    agent = AgentService(graph)
    config = {"configurable": {"thread_id": phone}}
    language = "en"
    print("Chat started. /state shows the session, /quit exits.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/state":
            values = (await graph.aget_state(config)).values
            print({k: v for k, v in values.items() if k not in ("messages", "turn")})
            continue
        language = detect_language(text) or language
        start = time.perf_counter()
        msg = IncomingMessage("cli", phone, args.name, "text", text, language, time.monotonic())
        replies = await agent(msg, language)
        elapsed = (time.perf_counter() - start) * 1000
        for r in replies:
            print(f"agent> {r}")
        for src in (await graph.aget_state(config)).values.get("turn", {}).get("sources") or []:
            print(f"       [source {src.get('doc_id')} score={src.get('score')} checked {src.get('checked_on')}: "
                  f"{src.get('source')}]")
        print(f"       ({elapsed:.0f} ms)\n")


if __name__ == "__main__":
    asyncio.run(main())
