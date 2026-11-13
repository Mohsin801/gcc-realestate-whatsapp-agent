"""Chat with the agent in the terminal, without WhatsApp, Redis or Postgres.

    .\\.venv\\Scripts\\python.exe scripts\\chat_cli.py            # uses the LLM from .env (GEMINI_API_KEY, GEMINI_MODEL)
    .\\.venv\\Scripts\\python.exe scripts\\chat_cli.py --no-llm   # rules + templates only
    .\\.venv\\Scripts\\python.exe scripts\\chat_cli.py --slots    # fake calendar so the booking flow can be tried

Type messages in English or Arabic; `/state` prints the session state, `/quit` exits.
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent.graph import build_graph  # noqa: E402
from agent.language import detect_language  # noqa: E402
from agent.llm import build_llm  # noqa: E402
from agent.runner import AgentService  # noqa: E402
from agent.services import BookingResult  # noqa: E402
from core.config import get_settings  # noqa: E402
from services.inbound import IncomingMessage  # noqa: E402

PHONE = "971500000000"


class DemoScheduler:
    """Pretend calendar for trying the booking flow locally (Phase 3 replaces it with Google Calendar)."""

    LABELS = {"en": ["Tomorrow at 4:00 PM GST", "Thursday at 11:00 AM GST", "Friday at 2:30 PM GST"],
              "ar": ["بكرة الساعة ٤:٠٠ العصر بتوقيت الخليج", "الخميس الساعة ١١:٠٠ الصبح بتوقيت الخليج",
                     "الجمعة الساعة ٢:٣٠ الظهر بتوقيت الخليج"]}

    def __init__(self):
        self.offered = False

    async def offer_slots(self, phone, country, language):
        self.offered = True
        return self.LABELS[language]

    async def offered_labels(self, phone, language):
        return self.LABELS[language] if self.offered else []

    async def book(self, phone, lead_name, choice, language):
        return BookingResult(ok=True, slot_iso="2026-11-16T12:00:00+00:00", slot_label=self.LABELS[language][choice - 1],
                             meet_link="https://meet.google.com/demo-demo-demo")

    async def add_email(self, phone, email):
        return True


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--slots", action="store_true", help="use a fake calendar to try booking")
    parser.add_argument("--name", default="Ahmed")
    args = parser.parse_args()

    settings = get_settings()
    llm = None if args.no_llm else build_llm(settings)
    graph = build_graph(llm=llm, settings=settings, scheduler=DemoScheduler() if args.slots else None,
                        checkpointer=InMemorySaver())
    agent = AgentService(graph)
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
            values = (await graph.aget_state({"configurable": {"thread_id": PHONE}})).values
            print({k: v for k, v in values.items() if k not in ("messages", "turn")})
            continue
        language = detect_language(text) or language
        start = time.perf_counter()
        msg = IncomingMessage("cli", PHONE, args.name, "text", text, language, time.monotonic())
        replies = await agent(msg, language)
        elapsed = (time.perf_counter() - start) * 1000
        for r in replies:
            print(f"agent> {r}")
        print(f"       ({elapsed:.0f} ms)\n")


if __name__ == "__main__":
    asyncio.run(main())
