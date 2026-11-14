"""Knowledge base on Qdrant (spec: "Qdrant: GCC Legal & Properties").

Two collections:
  * ``RAG_COLLECTION`` (facts): legal / tax / residency facts from OFFICIAL government sources only, kept in
    ``knowledge_base/facts/*.json``. Every fact is written in English and Arabic and carries its source URLs
    and the date it was checked. A fact whose source is not an allow-listed government domain is rejected.
  * ``PROPERTIES_COLLECTION`` (listings): the owner's own property listings (optional, loaded from a JSON file).

Each document is stored as one point per language (both texts in the payload), so an Arabic or English
question finds it and the passage is returned in the lead's language. ``search`` filters by country when it
is known and drops anything below ``RAG_SCORE_THRESHOLD``. An empty result means "no verified answer": the
agent then says it will confirm on the call instead of guessing (spec DoD: no hallucination).
"""
import asyncio
import json
import logging
import math
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from qdrant_client import AsyncQdrantClient, models

from agent.normalize import to_western_digits
from agent.services import Passage
from core.config import Settings

log = logging.getLogger(__name__)

Country = Literal["DUBAI", "SAUDI", "QATAR", "KUWAIT"]

# Government domains a fact may cite (a host matches the domain itself or any subdomain of it).
OFFICIAL_DOMAINS: tuple[str, ...] = (
    # UAE / Dubai
    "dubailand.gov.ae",  # Dubai Land Department
    "dubai.gov.ae",  # Government of Dubai (incl. dlp.dubai.gov.ae, Dubai Legislation Portal)
    "u.ae",  # official UAE government portal
    "icp.gov.ae",  # Federal Authority for Identity, Citizenship, Customs & Port Security
    # Saudi Arabia
    "zatca.gov.sa",  # Zakat, Tax and Customs Authority
    "rega.gov.sa",  # Real Estate General Authority
    "spa.gov.sa",  # Saudi Press Agency (official)
    "my.gov.sa",  # unified national platform
    "housing.gov.sa",  # Ministry of Municipalities and Housing
    # Qatar
    "moj.gov.qa",  # Ministry of Justice
    "aqarat.gov.qa",  # Real Estate Regulatory Authority
    "sharek.gov.qa",  # government services portal
    "hukoomi.gov.qa",  # official government portal
    # Kuwait
    "kdipa.gov.kw",  # Kuwait Direct Investment Promotion Authority
    "moj.gov.kw",  # Ministry of Justice
    "e.gov.kw",  # Kuwait Government Online
)

_ARABIC = re.compile(r"[؀-ۿ]")
_FIGURE = re.compile(r"\d+(?:[.,]\d+)*")
_ID = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "gcc-realestate-agent/knowledge")


class KnowledgeBaseError(RuntimeError):
    """Invalid knowledge-base content or an incompatible Qdrant collection."""


class KnowledgeConfigError(RuntimeError):
    """The knowledge base can't run with the current settings (e.g. no embedding model configured)."""


def is_official_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        return False
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)


def figures(text: str) -> set[str]:
    """Every number in a text, normalised (Arabic-Indic digits -> ASCII, thousands commas removed)."""
    return {m.replace(",", "") for m in _FIGURE.findall(to_western_digits(text))}


# ---------------------------------------------------------------------------------------------- content models
class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3)
    url: str

    @field_validator("url")
    @classmethod
    def _official(cls, url: str) -> str:
        if not is_official_url(url):
            raise ValueError(f"not an https URL on an official government domain: {url}")
        return url


class Fact(BaseModel):
    """One verified fact, in English and Arabic, with where and when it was checked."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=_ID)
    country: Country
    topic: Literal["fees", "tax", "mortgage", "ownership", "golden_visa", "process"]
    title_en: str = Field(min_length=10, max_length=200)
    title_ar: str = Field(min_length=10, max_length=200)
    # <= one WhatsApp bubble (REPLY_MAX_CHARS default), so the fact can be sent as-is if the LLM is down
    en: str = Field(min_length=20, max_length=400)
    ar: str = Field(min_length=20, max_length=400)
    sources: list[Source] = Field(min_length=1)
    checked_on: date

    @model_validator(mode="after")
    def _consistent(self) -> "Fact":
        if self.checked_on > date.today():
            raise ValueError("checked_on is in the future")
        for name in ("title_ar", "ar"):
            if not _ARABIC.search(getattr(self, name)):
                raise ValueError(f"{name} must be written in Arabic")
        # The two languages must state exactly the same figures (guards against translation slips).
        if figures(self.en) != figures(self.ar):
            raise ValueError(f"figures differ between en {sorted(figures(self.en))} and ar {sorted(figures(self.ar))}")
        if figures(self.title_en) != figures(self.title_ar):
            raise ValueError("figures differ between title_en and title_ar")
        return self


class Listing(BaseModel):
    """One of the owner's property listings (supplied by the owner, not by us)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=_ID)
    country: Country
    title: str = Field(min_length=3)
    price: float = Field(gt=0)
    currency: Literal["AED", "SAR", "QAR", "KWD", "USD"]
    location: str = ""
    property_type: str = ""
    bedrooms: int | None = Field(None, ge=0)
    description: str = ""
    title_ar: str | None = None
    description_ar: str | None = None
    url: str | None = None

    def text(self, language: str) -> str:
        if language == "ar":
            bits = [f"عقار معروض: {self.title_ar or self.title}."]
            if self.location:
                bits.append(f"الموقع: {self.location}.")
            if self.property_type:
                bits.append(f"النوع: {self.property_type}.")
            if self.bedrooms is not None:
                bits.append(f"غرف النوم: {self.bedrooms}.")
            bits.append(f"السعر: {self.price:,.0f} {self.currency}.")
            desc = self.description_ar or self.description
        else:
            bits = [f"Listing: {self.title}."]
            if self.location:
                bits.append(f"Location: {self.location}.")
            if self.property_type:
                bits.append(f"Type: {self.property_type}.")
            if self.bedrooms is not None:
                bits.append(f"Bedrooms: {self.bedrooms}.")
            bits.append(f"Price: {self.currency} {self.price:,.0f}.")
            desc = self.description
        if desc:
            bits.append(desc)
        return " ".join(bits)


def _read_json_list(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise KnowledgeBaseError(f"{path}: {e}") from e
    if not isinstance(data, list):
        raise KnowledgeBaseError(f"{path}: expected a JSON list")
    return data


def load_facts(directory: str | Path) -> list[Fact]:
    """Load and validate every ``*.json`` fact file in ``directory``. Raises ``KnowledgeBaseError``."""
    directory = Path(directory)
    files = sorted(directory.glob("*.json"))
    if not files:
        raise KnowledgeBaseError(f"no fact files found in {directory}")
    facts: list[Fact] = []
    for path in files:
        for i, raw in enumerate(_read_json_list(path)):
            try:
                facts.append(Fact.model_validate(raw))
            except ValidationError as e:
                raise KnowledgeBaseError(f"{path.name} item {i} ({raw.get('id', '?')}): {e}") from e
    _check_unique([f.id for f in facts], "fact")
    return facts


def load_listings(path: str | Path) -> list[Listing]:
    path = Path(path)
    listings: list[Listing] = []
    for i, raw in enumerate(_read_json_list(path)):
        try:
            listings.append(Listing.model_validate(raw))
        except ValidationError as e:
            raise KnowledgeBaseError(f"{path.name} item {i}: {e}") from e
    _check_unique([x.id for x in listings], "listing")
    return listings


def _check_unique(ids: list[str], what: str) -> None:
    seen, dupes = set(), set()
    for i in ids:
        (dupes if i in seen else seen).add(i)
    if dupes:
        raise KnowledgeBaseError(f"duplicate {what} ids: {sorted(dupes)}")


def stale_facts(facts: Sequence[Fact], max_age_days: int, today: date | None = None) -> list[Fact]:
    """Facts not re-checked within ``max_age_days``: rules and fees change, so they need a fresh look."""
    today = today or date.today()
    return [f for f in facts if (today - f.checked_on).days > max_age_days]


# ---------------------------------------------------------------------------------------------- embeddings
class Embedder(Protocol):
    dim: int

    async def embed_documents(self, docs: Sequence[tuple[str, str]]) -> list[list[float]]:
        """Embed (title, text) pairs for storage."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a lead's question for search."""
        ...


class GeminiEmbedder:
    """Gemini embeddings through the google-genai SDK.

    ``gemini-embedding-2`` takes the task as a text prefix (it rejects ``task_type``) and returns ONE
    aggregated vector for a list of inputs, so every text is sent in its own request.
    ``gemini-embedding-001`` uses ``task_type`` instead: set ``GEMINI_EMBEDDING_TASK_PREFIXES=false``.
    Vectors are L2-normalised (required for 001 at reduced dimensions, harmless otherwise).
    """

    QUERY_PREFIX = "task: question answering | query: "

    def __init__(self, client: Any, model: str, dim: int, *, task_prefixes: bool = True, concurrency: int = 4):
        self._client = client
        self.model = model
        self.dim = dim
        self.task_prefixes = task_prefixes
        self._sem = asyncio.Semaphore(concurrency)

    @classmethod
    def from_settings(cls, settings: Settings) -> "GeminiEmbedder":
        if not settings.gemini_api_key or not settings.gemini_embedding_model:
            raise KnowledgeConfigError("GEMINI_API_KEY and GEMINI_EMBEDDING_MODEL must be set for the knowledge base")
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(settings.embedding_timeout_seconds * 1000)),
        )
        return cls(client, settings.gemini_embedding_model, settings.embedding_dim,
                   task_prefixes=settings.gemini_embedding_task_prefixes)

    async def _embed(self, text: str, task_type: str, title: str | None = None) -> list[float]:
        from google.genai import types

        config = types.EmbedContentConfig(output_dimensionality=self.dim)
        if not self.task_prefixes:
            config.task_type = task_type
            if title and task_type == "RETRIEVAL_DOCUMENT":
                config.title = title
        async with self._sem:
            response = await self._client.aio.models.embed_content(model=self.model, contents=text, config=config)
        embeddings = response.embeddings or []
        if len(embeddings) != 1:
            raise ValueError(f"expected 1 embedding, got {len(embeddings)}")
        values = [float(v) for v in embeddings[0].values or []]
        if len(values) != self.dim:
            raise ValueError(f"embedding has {len(values)} dimensions, EMBEDDING_DIM is {self.dim}")
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0:
            raise ValueError("zero embedding")
        return [v / norm for v in values]

    async def embed_documents(self, docs: Sequence[tuple[str, str]]) -> list[list[float]]:
        tasks = []
        for title, text in docs:
            content = f"title: {title or 'none'} | text: {text}" if self.task_prefixes else text
            tasks.append(self._embed(content, "RETRIEVAL_DOCUMENT", title))
        return list(await asyncio.gather(*tasks))

    async def embed_query(self, text: str) -> list[float]:
        content = self.QUERY_PREFIX + text if self.task_prefixes else text
        return await self._embed(content, "RETRIEVAL_QUERY")


# ---------------------------------------------------------------------------------------------- store + search
@dataclass(frozen=True)
class _Doc:
    kind: str  # "fact" | "listing"
    doc_id: str
    lang: str
    title: str
    text: str
    payload: dict = field(default_factory=dict)

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(POINT_NAMESPACE, f"{self.kind}/{self.doc_id}/{self.lang}"))


@dataclass(frozen=True)
class IngestReport:
    collection: str
    documents: int
    points: int


def _fact_docs(facts: Sequence[Fact]) -> list[_Doc]:
    docs = []
    for f in facts:
        payload = {
            "kind": "fact", "doc_id": f.id, "country": f.country, "topic": f.topic,
            "title_en": f.title_en, "title_ar": f.title_ar, "text_en": f.en, "text_ar": f.ar,
            "sources": [s.model_dump() for s in f.sources], "checked_on": f.checked_on.isoformat(),
        }
        docs.append(_Doc("fact", f.id, "en", f.title_en, f.en, {**payload, "lang": "en"}))
        docs.append(_Doc("fact", f.id, "ar", f.title_ar, f.ar, {**payload, "lang": "ar"}))
    return docs


def _listing_docs(listings: Sequence[Listing]) -> list[_Doc]:
    docs = []
    for x in listings:
        has_ar = bool(x.title_ar or x.description_ar)
        payload = {
            "kind": "listing", "doc_id": x.id, "country": x.country, "title_en": x.title,
            "title_ar": x.title_ar, "text_en": x.text("en"), "text_ar": x.text("ar") if has_ar else None,
            "url": x.url, "sources": [{"name": "Owner listing", "url": x.url}] if x.url else [],
        }
        docs.append(_Doc("listing", x.id, "en", x.title, x.text("en"), {**payload, "lang": "en"}))
        if has_ar:
            docs.append(_Doc("listing", x.id, "ar", x.title_ar or x.title, x.text("ar"), {**payload, "lang": "ar"}))
    return docs


class KnowledgeBase:
    """Implements the agent's ``Knowledge`` protocol on Qdrant, plus ingestion for the CLI."""

    EXISTS_TTL_SECONDS = 300.0

    def __init__(self, client: AsyncQdrantClient, embedder: Embedder, settings: Settings, *, clock=time.monotonic):
        self.client = client
        self.embedder = embedder
        self.facts_collection = settings.rag_collection
        self.listings_collection = settings.properties_collection
        self.top_k = settings.rag_top_k
        self.threshold = settings.rag_score_threshold
        self.timeout = settings.rag_timeout_seconds
        self._clock = clock
        self._exists: dict[str, tuple[bool, float]] = {}

    # ----------------------------------------------------------------- search (agent path)
    async def search(self, query: str, country: str | None, language: str) -> list[Passage]:
        return await asyncio.wait_for(self._search(query, country, language), timeout=self.timeout)

    async def _search(self, query: str, country: str | None, language: str,
                      threshold: float | None = None) -> list[Passage]:
        query = (query or "").strip()
        if not query:
            return []
        vector = await self.embedder.embed_query(query)
        flt = None
        if country:
            flt = models.Filter(must=[models.FieldCondition(key="country", match=models.MatchValue(value=country))])
        limit = self.top_k * 2  # each document has one point per language
        hits = []
        for name in (self.facts_collection, self.listings_collection):
            if await self._available(name):
                response = await self.client.query_points(
                    name, query=vector, query_filter=flt, limit=limit, with_payload=True,
                    score_threshold=self.threshold if threshold is None else threshold,
                )
                hits.extend(response.points)
        best: dict[tuple[str, str], Any] = {}
        for hit in hits:
            p = hit.payload or {}
            key = (p.get("kind", ""), p.get("doc_id", ""))
            if key not in best or hit.score > best[key].score:
                best[key] = hit
        ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)[: self.top_k]
        return [self._passage(h, language) for h in ranked]

    @staticmethod
    def _passage(hit, language: str) -> Passage:
        p = hit.payload or {}
        text = (p.get("text_ar") if language == "ar" else None) or p.get("text_en") or ""
        sources = p.get("sources") or []
        return Passage(
            text=text,
            source=sources[0]["url"] if sources else "",
            country=p.get("country"),
            score=float(hit.score),
            metadata={
                "kind": p.get("kind"), "doc_id": p.get("doc_id"), "topic": p.get("topic"),
                "checked_on": p.get("checked_on"), "sources": sources,
            },
        )

    async def _available(self, name: str) -> bool:
        """Whether a collection exists (cached briefly, so a new ingest is picked up without a restart)."""
        cached = self._exists.get(name)
        now = self._clock()
        if cached and now - cached[1] < self.EXISTS_TTL_SECONDS:
            return cached[0]
        exists = await self.client.collection_exists(name)
        self._exists[name] = (exists, now)
        if not exists and name == self.facts_collection:
            log.warning("knowledge collection %r is missing: run scripts/ingest_kb.py ingest", name)
        return exists

    # ----------------------------------------------------------------- ingestion (CLI path)
    async def ingest_facts(self, facts: Sequence[Fact], *, recreate: bool = False) -> IngestReport:
        if not facts:
            raise KnowledgeBaseError("refusing to ingest an empty fact list")
        return await self._ingest(self.facts_collection, _fact_docs(facts), recreate)

    async def ingest_listings(self, listings: Sequence[Listing], *, recreate: bool = False) -> IngestReport:
        if not listings:
            raise KnowledgeBaseError("refusing to ingest an empty listing file")
        return await self._ingest(self.listings_collection, _listing_docs(listings), recreate)

    async def _ingest(self, collection: str, docs: list[_Doc], recreate: bool) -> IngestReport:
        await self._ensure_collection(collection, recreate)
        vectors = await self.embedder.embed_documents([(d.title, d.text) for d in docs])
        points = [models.PointStruct(id=d.point_id, vector=v, payload=d.payload) for d, v in zip(docs, vectors)]
        await self.client.upsert(collection, points=points, wait=True)
        keep = sorted({d.doc_id for d in docs})
        # Remove documents that were deleted from the source files since the last ingest.
        await self.client.delete(
            collection,
            points_selector=models.FilterSelector(filter=models.Filter(
                must_not=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=keep))]
            )),
            wait=True,
        )
        self._exists[collection] = (True, self._clock())
        return IngestReport(collection, len(keep), len(points))

    async def _ensure_collection(self, name: str, recreate: bool) -> None:
        exists = await self.client.collection_exists(name)
        if exists and recreate:
            await self.client.delete_collection(name)
            exists = False
        if exists:
            info = await self.client.get_collection(name)
            vectors = info.config.params.vectors
            size = vectors.size if isinstance(vectors, models.VectorParams) else None
            if size != self.embedder.dim:
                raise KnowledgeBaseError(
                    f"collection {name!r} has vector size {size}, EMBEDDING_DIM is {self.embedder.dim}: "
                    "re-run with --recreate"
                )
            return
        await self.client.create_collection(
            name, vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE)
        )
        for key in ("country", "doc_id", "kind"):
            try:
                await self.client.create_payload_index(name, field_name=key,
                                                       field_schema=models.PayloadSchemaType.KEYWORD)
            except Exception:  # noqa: BLE001 - an index is an optimisation (and a no-op in local mode)
                log.debug("payload index on %s.%s not created", name, key, exc_info=True)


def build_knowledge(settings: Settings) -> tuple[KnowledgeBase, AsyncQdrantClient]:
    """Production wiring. Raises ``KnowledgeConfigError`` when embeddings aren't configured."""
    embedder = GeminiEmbedder.from_settings(settings)
    client = open_qdrant(settings)
    return KnowledgeBase(client, embedder, settings), client


def open_qdrant(settings: Settings) -> AsyncQdrantClient:
    """Qdrant server at ``QDRANT_URL``, or embedded Qdrant in the ``QDRANT_PATH`` folder (no server needed)."""
    if settings.qdrant_path:
        Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
        return AsyncQdrantClient(path=settings.qdrant_path)
    return AsyncQdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None,
        timeout=max(1, math.ceil(settings.qdrant_timeout_seconds)), check_compatibility=False,
    )
