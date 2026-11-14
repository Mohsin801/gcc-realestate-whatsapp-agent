"""Manage the Qdrant knowledge base.

    .\\.venv\\Scripts\\python.exe scripts\\ingest_kb.py validate                 # check the fact files (offline)
    .\\.venv\\Scripts\\python.exe scripts\\ingest_kb.py ingest                   # embed + upsert the facts
    .\\.venv\\Scripts\\python.exe scripts\\ingest_kb.py ingest --listings my_listings.json   # + owner's listings
    .\\.venv\\Scripts\\python.exe scripts\\ingest_kb.py ingest --recreate        # after changing EMBEDDING_DIM/model
    .\\.venv\\Scripts\\python.exe scripts\\ingest_kb.py search "DLD fees?" --country DUBAI --all-scores

`ingest` needs GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL and a running Qdrant (QDRANT_URL). It is idempotent:
documents removed from the files are deleted from Qdrant. `search --all-scores` shows scores below
RAG_SCORE_THRESHOLD too, to calibrate the threshold.
"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import get_settings  # noqa: E402
from tools.rag import (  # noqa: E402
    KnowledgeBaseError,
    KnowledgeConfigError,
    build_knowledge,
    load_facts,
    load_listings,
    stale_facts,
)


def _facts_dir(settings) -> Path:
    path = Path(settings.knowledge_dir)
    return path if path.is_absolute() else ROOT / path


def validate(settings, listings_path: str | None) -> int:
    facts = load_facts(_facts_dir(settings))
    by_country: dict[str, int] = {}
    for f in facts:
        by_country[f.country] = by_country.get(f.country, 0) + 1
    print(f"{len(facts)} facts OK: " + ", ".join(f"{c} {n}" for c, n in sorted(by_country.items())))
    for f in stale_facts(facts, settings.rag_fact_max_age_days):
        print(f"  WARNING: {f.id} was last checked on {f.checked_on}; re-check its sources")
    if listings_path:
        print(f"{len(load_listings(listings_path))} listings OK")
    return 0


async def ingest(settings, listings_path: str | None, recreate: bool) -> int:
    validate(settings, listings_path)
    kb, client = build_knowledge(settings)
    try:
        report = await kb.ingest_facts(load_facts(_facts_dir(settings)), recreate=recreate)
        print(f"ingested {report.documents} facts ({report.points} points) into {report.collection!r}")
        if listings_path:
            report = await kb.ingest_listings(load_listings(listings_path), recreate=recreate)
            print(f"ingested {report.documents} listings ({report.points} points) into {report.collection!r}")
    finally:
        await client.close()
    return 0


async def search(settings, query: str, country: str | None, language: str, all_scores: bool) -> int:
    kb, client = build_knowledge(settings)
    try:
        passages = await kb._search(query, country, language, threshold=0.0 if all_scores else None)
    finally:
        await client.close()
    print(f"threshold RAG_SCORE_THRESHOLD={settings.rag_score_threshold}")
    if not passages:
        print("no passages: the agent would say it will confirm on the call")
    for p in passages:
        mark = "" if p.score >= settings.rag_score_threshold else "  (below threshold: not used)"
        print(f"\n{p.score:.3f} {p.metadata.get('doc_id')} [{p.country}]{mark}\n  {p.text}\n  source: {p.source}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_val = sub.add_parser("validate", help="validate fact files (and listings) without network access")
    p_val.add_argument("--listings")
    p_ing = sub.add_parser("ingest", help="embed and upsert into Qdrant")
    p_ing.add_argument("--listings", help="JSON file with the owner's property listings")
    p_ing.add_argument("--recreate", action="store_true", help="drop and recreate the collections")
    p_s = sub.add_parser("search", help="try a question against the knowledge base")
    p_s.add_argument("query")
    p_s.add_argument("--country", choices=["DUBAI", "SAUDI", "QATAR", "KUWAIT"])
    p_s.add_argument("--lang", default="en", choices=["en", "ar"])
    p_s.add_argument("--all-scores", action="store_true", help="also show passages below the threshold")
    args = parser.parse_args()

    settings = get_settings()
    try:
        if args.command == "validate":
            return validate(settings, args.listings)
        if args.command == "ingest":
            return asyncio.run(ingest(settings, args.listings, args.recreate))
        return asyncio.run(search(settings, args.query, args.country, args.lang, args.all_scores))
    except (KnowledgeBaseError, KnowledgeConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
