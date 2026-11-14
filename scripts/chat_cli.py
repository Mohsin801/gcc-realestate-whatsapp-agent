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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent.graph import build_graph  # noqa: E402
from agent.language import detect_language  # noqa: E402
from agent.llm import build_llm  # noqa: E402
from agent.runner import AgentService  # noqa: E402
from core.config import get_settings  # noqa: E402
from evals.harness import DemoScheduler  # noqa: E402
from services.inbound import IncomingMessage  # noqa: E402
from tools.rag import build_knowledge  # noqa: E402


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
    agent = AgentService(graph, budget_seconds=settings.latency_budget_seconds)  # same budget as production
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
        turn = (await graph.aget_state(config)).values.get("turn", {})
        for src in turn.get("sources") or []:
            print(f"       [source {src.get('doc_id')} score={src.get('score')} checked {src.get('checked_on')}: "
                  f"{src.get('source')}]")
        slow = sorted((turn.get("timings") or {}).items(), key=lambda kv: -kv[1])[:3]
        detail = ", ".join(f"{k} {v:.0f}" for k, v in slow)
        skipped = f"; skipped for time: {', '.join(turn['budget_skips'])}" if turn.get("budget_skips") else ""
        print(f"       ({elapsed:.0f} ms: {detail}; reply from {turn.get('reply_source')}{skipped})\n")


if __name__ == "__main__":
    asyncio.run(main())
