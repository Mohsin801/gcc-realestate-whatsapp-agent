"""Calibrate RAG_SCORE_THRESHOLD against labelled questions (knowledge_base/calibration.json).

    .\\.venv\\Scripts\\python.exe scripts\\calibrate_kb.py                       # embed every question (1 call each)
    .\\.venv\\Scripts\\python.exe scripts\\calibrate_kb.py --scores scores.json  # cache: only new questions are embedded
    .\\.venv\\Scripts\\python.exe scripts\\calibrate_kb.py --scores scores.json --offline  # re-analyse, no API calls

Each question is searched exactly like the agent does (country filter, best score per fact over its English and
Arabic points, top RAG_TOP_K), but with no threshold, so every fact's score is recorded. Then each threshold
on a grid is scored:
  * positive question ("expect": fact ids): ANSWER if an expected fact is in the top k at or above the threshold;
    WRONG if other facts are there but none expected (the reply would be built on the wrong passage); DEFER if
    nothing passes (safe: "I'll confirm on the call").
  * negative question ("expect": []): FALSE if any fact passes that isn't listed in "ok" (facts that make a fair
    partial answer, e.g. Golden Visa thresholds for a duration question); otherwise DEFER, as it should.
The recommendation is the lowest threshold with no FALSE and no WRONG, which keeps the most answers.
Re-run after an ingest with a new embedding model, or after adding facts (add questions for them too).
"""
import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CALIBRATION_FILE = Path(__file__).resolve().parents[1] / "knowledge_base" / "calibration.json"


@dataclass(frozen=True)
class Outcome:
    threshold: float
    answered: int  # positives answered from an expected fact
    deferred: int  # positives with nothing above the threshold (safe miss)
    wrong: int  # positives answered from unexpected facts only
    false: int  # negatives that would get an answer
    positives: int
    negatives: int

    @property
    def safe(self) -> bool:
        return self.wrong == 0 and self.false == 0


def judge(item: dict, scores: dict[str, float], threshold: float, top_k: int) -> str:
    """ANSWER | WRONG | DEFER for a positive question; FALSE | DEFER for a negative one."""
    passing = sorted((d for d, s in scores.items() if s >= threshold), key=lambda d: -scores[d])[:top_k]
    if item["expect"]:
        if set(passing) & set(item["expect"]):
            return "ANSWER"
        return "WRONG" if passing else "DEFER"
    return "FALSE" if set(passing) - set(item.get("ok") or []) else "DEFER"


def sweep(items: list[dict], scores: list[dict[str, float]], thresholds: list[float], top_k: int) -> list[Outcome]:
    out = []
    for t in thresholds:
        verdicts = [judge(i, s, t, top_k) for i, s in zip(items, scores)]
        pos = [v for i, v in zip(items, verdicts) if i["expect"]]
        neg = [v for i, v in zip(items, verdicts) if not i["expect"]]
        out.append(Outcome(round(t, 3), pos.count("ANSWER"), pos.count("DEFER"), pos.count("WRONG"),
                           neg.count("FALSE"), len(pos), len(neg)))
    return out


def recommend(outcomes: list[Outcome]) -> Outcome | None:
    """Lowest safe threshold: no false answers on negatives and no wrong answers on positives."""
    return next((o for o in outcomes if o.safe), None)


def cache_key(item: dict) -> str:
    """Scores depend on the question text, its language and the country filter."""
    return f"{item.get('country') or '-'}|{item['lang']}|{item['q']}"


async def collect_scores(items: list[dict], settings) -> dict[str, dict[str, float]]:
    """Every fact's score for each question (searched like the agent, but with no threshold)."""
    from tools.rag import build_knowledge

    # every fact's score is kept (top_k raised); judging later uses the real RAG_TOP_K
    kb, client = build_knowledge(settings.model_copy(update={"rag_top_k": 50}))
    try:
        results = {}
        for n, item in enumerate(items, 1):
            passages = await kb._search(item["q"], item.get("country"), item["lang"], threshold=0.0)
            results[cache_key(item)] = {p.metadata["doc_id"]: round(p.score, 4) for p in passages}
            print(f"  embedded {n}/{len(items)}", end="\r", file=sys.stderr)
        print(file=sys.stderr)
        return results
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", metavar="PATH",
                        help="score cache: reused for known questions, new ones are embedded and added")
    parser.add_argument("--offline", action="store_true", help="use only the --scores cache (no embedding calls)")
    parser.add_argument("--min", type=float, default=0.30)
    parser.add_argument("--max", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    from core.config import get_settings

    settings = get_settings()
    items = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    cache_path = Path(args.scores) if args.scores else None
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path and cache_path.exists() else {}
    missing = [i for i in items if cache_key(i) not in cache]
    if missing and args.offline:
        parser.error(f"{len(missing)} question(s) not in the score cache; run without --offline")
    if missing:
        print(f"Embedding {len(missing)} question(s) ({len(items) - len(missing)} from the cache)", file=sys.stderr)
        cache.update(asyncio.run(collect_scores(missing, settings)))
        if cache_path:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    scores = [cache[cache_key(i)] for i in items]
    top_k = settings.rag_top_k

    print("Per question (top fact and score; expected fact's score):")
    for item, s in zip(items, scores):
        ranked = sorted(s.items(), key=lambda kv: -kv[1])
        top = f"{ranked[0][0]} {ranked[0][1]:.3f}" if ranked else "-"
        if item["expect"]:
            best = max((s.get(d, 0.0) for d in item["expect"]), default=0.0)
            label = f"POS expected {best:.3f}"
        else:
            label = "NEG" + (" (ok: " + ", ".join(item["ok"]) + ")" if item.get("ok") else "")
        print(f"  [{item.get('country') or '-':6}] {item['q'][:62]:<62}  top {top:<42} {label}")

    grid = [args.min + i * args.step for i in range(int(round((args.max - args.min) / args.step)) + 1)]
    outcomes = sweep(items, scores, grid, top_k)
    print(f"\nThreshold sweep (top_k={top_k}):  answered / deferred / WRONG of positives,  FALSE of negatives")
    for o in outcomes:
        mark = "  <- safe" if o.safe else ""
        print(f"  {o.threshold:.2f}:  {o.answered:2}/{o.positives} answered  {o.deferred:2} deferred  {o.wrong} wrong"
              f"   {o.false}/{o.negatives} false{mark}")
    best = recommend(outcomes)
    print(f"\nCurrent RAG_SCORE_THRESHOLD={settings.rag_score_threshold}")
    if best is None:
        print("No safe threshold on this grid: fix the facts or labels before choosing one.")
        return 1
    print(f"Lowest safe threshold: {best.threshold:.2f} ({best.answered}/{best.positives} positives answered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
