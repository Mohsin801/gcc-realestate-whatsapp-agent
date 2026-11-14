"""Run the conversation evaluation (evals/conversations.json) against the agent and report against the spec DoD.

    .\\.venv\\Scripts\\python.exe scripts\\eval_agent.py                  # live: the LLM from .env, no knowledge base
    .\\.venv\\Scripts\\python.exe scripts\\eval_agent.py --kb             # + the Qdrant knowledge base (after an ingest)
    .\\.venv\\Scripts\\python.exe scripts\\eval_agent.py --no-llm         # rules + templates only (offline)
    .\\.venv\\Scripts\\python.exe scripts\\eval_agent.py --repeat 5 --json report.json
    .\\.venv\\Scripts\\python.exe scripts\\eval_agent.py --rpm 5              # Gemini free tier: 5 requests/minute
    .\\.venv\\Scripts\\python.exe scripts\\eval_agent.py --case en-saudi-rett --case ar-dubai-dld-fee

Cases run one after another, so the latencies are those of a single conversation. The calendar is a demo
calendar (no Google), and WhatsApp is not called: a turn passes the latency check when the agent finishes
within LATENCY_BUDGET_SECONDS - LATENCY_SEND_RESERVE_SECONDS. Exit code 0 when every case passes.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import build_llm  # noqa: E402
from core.config import get_settings  # noqa: E402
from evals.harness import Report, load_cases, run_eval  # noqa: E402
from tools.rag import build_knowledge  # noqa: E402


def print_report(report: Report) -> None:
    summary = report.as_dict()
    for case in report.cases:
        if case.skipped:
            print(f"SKIP  {case.case_id}  ({case.skipped})")
            continue
        print(f"{'PASS' if case.passed else 'FAIL'}  {case.case_id}")
        for t in case.turns:
            mark = "  " if t.passed else "x "
            extra = f"  skipped:{','.join(t.budget_skips)}" if t.budget_skips else ""
            print(f"   {mark}{t.index}. {t.latency_ms:5.0f} ms  [{t.extraction_source}/{t.reply_source}]  "
                  f"{t.user[:60]}{extra}")
            for r in t.replies if not t.passed else []:
                print(f"        > {r[:160]!r}")
            for f in t.failures:
                print(f"        ! {f}")
    cases, turns = summary["cases"], summary["turns"]
    print()
    print(f"Mode: LLM={'on' if report.mode['llm'] else 'off'}  knowledge base={'on' if report.mode['knowledge'] else 'off'}"
          f"  repeat={report.mode['repeat']}")
    print(f"Cases: {cases['passed']}/{cases['run']} passed ({len(cases['skipped'])} skipped)   "
          f"Turns: {turns['passed']}/{turns['total']} passed")
    if summary["failed_checks"]:
        print("Failed checks: " + ", ".join(f"{k}={v}" for k, v in summary["failed_checks"].items()))
    print(f"Agent latency (budget {report.budget_ms:.0f} ms = total budget - send reserve):")
    for name, label in ((None, "All"), ("en", "English"), ("ar", "Arabic")):
        print(f"  {label:<8} {report.latency(name).line()}")
    print("Reply source: " + ", ".join(f"{k}={v}" for k, v in summary["reply_source"].items()))
    print("Extraction source: " + ", ".join(f"{k}={v}" for k, v in summary["extraction_source"].items()))
    if summary["llm_errors"]:
        print("LLM errors: " + ", ".join(f"{k}={v}" for k, v in summary["llm_errors"].items()))
    if summary["budget_skips"]:
        print("Skipped for time: " + ", ".join(f"{k}={v}" for k, v in summary["budget_skips"].items()))
    print("Node p95 (ms): " + ", ".join(f"{k}={v:.0f}" for k, v in summary["node_p95_ms"].items()))
    print(f"\nRESULT: {'PASS' if report.passed else 'FAIL'}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="rules + templates only")
    parser.add_argument("--kb", action="store_true", help="use the Qdrant knowledge base")
    parser.add_argument("--repeat", type=int, default=1, help="run every case N times (latency distribution)")
    parser.add_argument("--case", action="append", help="only these case ids (repeatable)")
    parser.add_argument("--cases-file", help="another conversations file")
    parser.add_argument("--json", metavar="PATH", help="also write the full report as JSON")
    parser.add_argument("--rpm", type=int, help="pace LLM requests below this many per minute (waits between "
                                                "turns, not counted as latency), e.g. 5 on the Gemini free tier")
    args = parser.parse_args()

    settings = get_settings()
    cases = load_cases(args.cases_file) if args.cases_file else load_cases()
    if args.case:
        unknown = set(args.case) - {c["id"] for c in cases}
        if unknown:
            parser.error(f"unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c["id"] in args.case]
    llm = None if args.no_llm else build_llm(settings)
    knowledge, qdrant = build_knowledge(settings) if args.kb else (None, None)
    try:
        report = await run_eval(settings, llm=llm, knowledge=knowledge, cases=cases, repeat=args.repeat,
                                rpm=args.rpm)
    finally:
        if qdrant is not None:
            await qdrant.close()
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report written to {args.json}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
