"""The conversation evaluation (evals/) run offline: rules + templates, the real facts in in-memory Qdrant."""
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from core.config import Settings
from evals.harness import check_turn, load_cases, run_eval
from tests.agent_fakes import FakeEmbedder, FakeLLM
from tools.rag import KnowledgeBase, load_facts

FACTS_DIR = Path(__file__).resolve().parents[1] / "knowledge_base" / "facts"


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


@pytest.fixture
async def kb():
    client = AsyncQdrantClient(location=":memory:")
    kb = KnowledgeBase(client, FakeEmbedder(), settings(rag_score_threshold=0.25))
    await kb.ingest_facts(load_facts(FACTS_DIR))
    yield kb
    await client.close()


def _explain(report) -> str:
    return "\n".join(f"{f['case']}#{f['turn']} {f['user']!r} -> {f['replies']}: {f['failures']}"
                     for f in report.as_dict()["failures"])


def test_cases_cover_both_languages_and_all_markets():
    cases = load_cases()
    assert {c["language"] for c in cases} == {"en", "ar"}
    countries = {v for c in cases for t in c["turns"] for k, v in (t.get("expect", {}).get("state") or {}).items()
                 if k == "target_country"}
    assert countries == {"DUBAI", "SAUDI", "QATAR", "KUWAIT"}
    assert {c["phone"] for c in cases}.__len__() == len(cases)  # one lead (thread) per case


async def test_rules_only_agent_passes_every_offline_case_with_knowledge_base(kb):
    report = await run_eval(settings(), knowledge=kb)
    summary = report.as_dict()
    assert report.passed, _explain(report)
    assert summary["cases"]["skipped"] == ["llm-lives-elsewhere-buys-dubai", "llm-khaleeji-colloquial"]
    assert summary["latency"]["all"]["within_budget"] == 1.0
    assert set(summary["node_p95_ms"]) >= {"extract", "respond", "persona_guard"}


async def test_without_knowledge_base_every_question_is_deferred():
    report = await run_eval(settings())
    assert report.passed, _explain(report)


async def test_eval_runs_llm_cases_with_an_llm():
    report = await run_eval(settings(), llm=FakeLLM(), cases=[c for c in load_cases() if c.get("requires")])
    assert not report.as_dict()["cases"]["skipped"]
    assert report.turns and all(t.reply_source == "llm" for t in report.turns)


# --- the checks themselves catch bad replies ----------------------------------------------------------
def _check(replies, expect=None, turn=None, user="Hi", language="en", latency_ms=10.0, **kw):
    return check_turn(expect or {}, language=language, replies=replies, turn=turn or {"system": ""},
                      state=kw.pop("state", {}), user_texts=[user], settings=settings(**kw),
                      latency_ms=latency_ms, budget_ms=2500, has_knowledge=kw.pop("has_knowledge", False))


def test_check_flags_ai_disclosure_in_both_languages():
    assert any(f.startswith("persona") for f in _check(["Sure! As an AI assistant I can help."]))
    assert any(f.startswith("persona") for f in _check(["أنا مساعد ذكي"], language="ar"))
    assert not _check(["Sure! As an AI assistant I can help."], persona_conceal_ai=False)


def test_check_flags_wrong_language_and_walls_of_text():
    assert any(f.startswith("language") for f in _check(["Hello, how can I help?"], language="ar"))
    assert any(f.startswith("bubbles") for f in _check(["word " * 100]))
    assert any(f.startswith("bubbles") for f in _check(["a", "b", "c"]))


def test_check_flags_figures_not_from_lead_or_knowledge():
    assert any(f.startswith("grounded") for f in _check(["The DLD fee is 4% of the price."]))
    ok_turn = {"system": "KNOWLEDGE:\n- The DLD fee is 4% of the property value."}
    assert not _check(["The DLD fee is 4% of the price."], turn=ok_turn)
    assert not _check(["Got it, 2,000,000 AED works."], user="2000000 AED")  # the lead's own number
    slots = {"system": "", "appendix": "1. Tomorrow at 4:00 PM GST"}
    assert not _check(["Which time suits you?", "1. Tomorrow at 4:00 PM GST"], turn=slots)


def test_check_flags_slow_turn_state_parts_and_knowledge():
    assert any(f.startswith("latency") for f in _check(["Hi!"], latency_ms=2600))
    turn = {"system": "", "parts": [{"kind": "DEFER"}], "sources": []}
    fails = _check(["I'll confirm on the call."], turn=turn, expect={"state": {"target_country": "DUBAI"},
                   "parts": ["ASK"], "kb": ["dubai-dld-sale-registration-fee"]}, has_knowledge=True)
    assert {f.split(":")[0] for f in fails} == {"state", "parts", "knowledge"}
