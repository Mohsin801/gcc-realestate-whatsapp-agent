"""Knowledge base on an in-memory Qdrant with a deterministic fake embedder (no network)."""
import asyncio
import re
from types import SimpleNamespace

import pytest
from qdrant_client import AsyncQdrantClient

from core.config import Settings
from tests.agent_fakes import FakeEmbedder
from tools.rag import GeminiEmbedder, KnowledgeBase, KnowledgeBaseError, Listing, is_official_url, load_facts

FACTS = load_facts("knowledge_base/facts")


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **{"rag_score_threshold": 0.25, **overrides})


@pytest.fixture
async def client():
    c = AsyncQdrantClient(location=":memory:")
    yield c
    await c.close()


@pytest.fixture
async def kb(client):
    kb = KnowledgeBase(client, FakeEmbedder(), settings())
    await kb.ingest_facts(FACTS)
    return kb


async def test_ingest_stores_one_point_per_language(kb, client):
    count = await client.count(kb.facts_collection, exact=True)
    assert count.count == 2 * len(FACTS)
    assert kb.embedder.documents == 2 * len(FACTS)


async def test_search_finds_the_right_fact_with_source_and_date(kb):
    passages = await kb.search("Dubai Land Department sale registration transfer fee", "DUBAI", "en")
    top = passages[0]
    assert top.metadata["doc_id"] == "dubai-dld-sale-registration-fee"
    assert "4%" in top.text and top.country == "DUBAI"
    assert is_official_url(top.source)
    assert top.metadata["checked_on"] == "2026-11-14"
    assert all(is_official_url(s["url"]) for s in top.metadata["sources"])


async def test_arabic_question_gets_arabic_passage(kb):
    passages = await kb.search("كم رسوم تسجيل بيع العقار في دائرة الأراضي والأملاك؟", "DUBAI", "ar")
    assert passages and passages[0].metadata["doc_id"].startswith("dubai-dld")
    assert re.search(r"[؀-ۿ]", passages[0].text)


async def test_passage_is_returned_in_the_leads_language_whatever_matched(kb):
    passages = await kb.search("Saudi Real Estate Transaction Tax RETT rate", "SAUDI", "ar")
    rett = next(p for p in passages if p.metadata["doc_id"] == "saudi-rett-rate")
    assert rett.text == next(f.ar for f in FACTS if f.id == "saudi-rett-rate")


async def test_results_are_filtered_by_country_and_deduplicated(kb):
    passages = await kb.search("Real Estate Transaction Tax RETT rate ZATCA", "DUBAI", "en")
    assert all(p.country == "DUBAI" for p in passages)
    passages = await kb.search("residency property owner Qatar", None, "en")
    ids = [p.metadata["doc_id"] for p in passages]
    assert len(ids) == len(set(ids)) and len(ids) <= kb.top_k


async def test_nothing_above_threshold_returns_empty(kb):
    assert await kb.search("best football team this season", None, "en") == []
    assert await kb.search("   ", "DUBAI", "en") == []


async def test_missing_collection_returns_empty(client):
    kb = KnowledgeBase(client, FakeEmbedder(), settings())
    assert await kb.search("DLD fees", "DUBAI", "en") == []


async def test_search_is_time_boxed(client):
    kb = KnowledgeBase(client, FakeEmbedder(delay=0.5), settings(rag_timeout_seconds=0.05))
    with pytest.raises(asyncio.TimeoutError):
        await kb.search("DLD fees", "DUBAI", "en")


async def test_reingest_removes_deleted_facts(kb, client):
    await kb.ingest_facts(FACTS[:3])
    assert (await client.count(kb.facts_collection, exact=True)).count == 6
    with pytest.raises(KnowledgeBaseError):
        await kb.ingest_facts([])


async def test_dimension_change_requires_recreate(kb, client):
    other = KnowledgeBase(client, FakeEmbedder(dim=128), settings())
    with pytest.raises(KnowledgeBaseError, match="--recreate"):
        await other.ingest_facts(FACTS)
    report = await other.ingest_facts(FACTS, recreate=True)
    assert report.points == 2 * len(FACTS)


async def test_listings_are_searchable_alongside_facts(kb):
    villa = Listing(id="villa-1", country="DUBAI", title="Palm Jumeirah sea view villa", price=12_500_000,
                    currency="AED", location="Palm Jumeirah", property_type="villa", bedrooms=5,
                    url="https://example.com/villa-1")
    flat = Listing(id="flat-1", country="QATAR", title="The Pearl apartment", title_ar="شقة في اللؤلؤة",
                   price=1_800_000, currency="QAR", location="The Pearl", bedrooms=2)
    report = await kb.ingest_listings([villa, flat])
    assert (report.documents, report.points) == (2, 3)  # the Qatar flat also has an Arabic point
    passages = await kb.search("sea view villa Palm Jumeirah price", "DUBAI", "ar")
    hit = next(p for p in passages if p.metadata["kind"] == "listing")
    assert "12,500,000" in hit.text and hit.text.startswith("Listing:")  # no Arabic text: English is used
    assert hit.source == "https://example.com/villa-1"


async def test_gemini_embedder_prefix_mode_sends_one_request_per_text():
    calls = []

    async def embed_content(*, model, contents, config):
        calls.append((model, contents, config))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0])])

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_content)))
    emb = GeminiEmbedder(client, "gemini-embedding-2", 2)
    vectors = await emb.embed_documents([("Title A", "text a"), ("", "text b")])
    assert vectors == [[0.6, 0.8], [0.6, 0.8]]  # L2-normalised
    assert [c[1] for c in calls] == ["title: Title A | text: text a", "title: none | text: text b"]
    assert all(c[2].task_type is None and c[2].output_dimensionality == 2 for c in calls)
    await emb.embed_query("DLD fees?")
    assert calls[-1][1] == "task: question answering | query: DLD fees?"


async def test_gemini_embedder_task_type_mode_and_dimension_check():
    calls = []

    async def embed_content(*, model, contents, config):
        calls.append((contents, config))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 0.0, 0.0])])

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(embed_content=embed_content)))
    emb = GeminiEmbedder(client, "gemini-embedding-001", 3, task_prefixes=False)
    await emb.embed_documents([("Title", "body")])
    await emb.embed_query("question")
    assert calls[0][0] == "body" and calls[0][1].task_type == "RETRIEVAL_DOCUMENT" and calls[0][1].title == "Title"
    assert calls[1][0] == "question" and calls[1][1].task_type == "RETRIEVAL_QUERY"
    wrong_dim = GeminiEmbedder(client, "gemini-embedding-001", 768, task_prefixes=False)
    with pytest.raises(ValueError, match="dimensions"):
        await wrong_dim.embed_query("question")


def test_gemini_embedder_requires_configuration():
    from tools.rag import KnowledgeConfigError

    with pytest.raises(KnowledgeConfigError):
        GeminiEmbedder.from_settings(Settings(_env_file=None, gemini_api_key="", gemini_embedding_model=""))
