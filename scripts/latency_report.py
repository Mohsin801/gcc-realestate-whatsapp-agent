"""Production latency against the spec DoD (reply in < 3 s, English and Arabic), from Postgres ``chat_logs``.

    .\\.venv\\Scripts\\python.exe scripts\\latency_report.py               # last 7 days
    .\\.venv\\Scripts\\python.exe scripts\\latency_report.py --days 1 --json
    .\\.venv\\Scripts\\python.exe scripts\\latency_report.py --check       # exit 1 if p95 is over the budget

``chat_logs.latency_ms`` is measured on the first reply bubble: from webhook receipt to Meta accepting it.
"""
import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings  # noqa: E402
from core.latency import LatencyStats  # noqa: E402
from db import repository  # noqa: E402
from db.session import dispose_engine, get_sessionmaker, init_engine  # noqa: E402


def summarise(rows: list[tuple[str | None, int]], budget_ms: float) -> dict[str, LatencyStats]:
    out = {"all": LatencyStats.of([ms for _, ms in rows], budget_ms)}
    for lang in ("en", "ar"):
        out[lang] = LatencyStats.of([ms for lg, ms in rows if lg == lang], budget_ms)
    return out


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=7)
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument("--check", action="store_true", help="exit 1 if p95 (all, en or ar) is over the budget")
    args = parser.parse_args()

    settings = get_settings()
    budget_ms = settings.latency_budget_seconds * 1000
    init_engine(settings.database_url)
    try:
        async with get_sessionmaker()() as session:
            rows = await repository.reply_latencies(session, datetime.now(UTC) - timedelta(days=args.days))
    finally:
        await dispose_engine()

    stats = summarise(rows, budget_ms)
    if args.json:
        print(json.dumps({"days": args.days, "budget_ms": budget_ms, **{k: vars(v) for k, v in stats.items()}}))
    else:
        print(f"Reply latency, last {args.days:g} days (budget {budget_ms:.0f} ms):")
        for name, label in (("all", "All"), ("en", "English"), ("ar", "Arabic")):
            print(f"  {label:<8} {stats[name].line()}")
    over = [k for k, v in stats.items() if v.count and v.p95_ms > budget_ms]
    if args.check and over:
        print(f"p95 over budget: {', '.join(over)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
