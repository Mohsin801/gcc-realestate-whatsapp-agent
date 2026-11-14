"""Application settings, loaded from environment variables / .env."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    public_base_url: str = "http://localhost:8000"  # used for the Google OAuth redirect

    # --- Infrastructure ---
    database_url: str = "postgresql+asyncpg://gcc:gcc@localhost:5432/gcc_agent"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    # Embedded Qdrant (no server, e.g. without Docker): a folder for the data. When set, QDRANT_URL is ignored.
    # Only one process can open the folder at a time, so use it for development, not with several workers.
    qdrant_path: str = ""

    # --- WhatsApp (Meta Cloud API) ---
    whatsapp_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_app_secret: str = ""
    whatsapp_verify_token: str = ""
    whatsapp_api_version: str = "v23.0"
    whatsapp_dedup_ttl_seconds: int = 86_400  # wamid dedup key lifetime

    # --- LLM ---
    llm_provider: str = "gemini"  # gemini | anthropic | openai
    gemini_api_key: str = ""
    gemini_model: str = ""  # set in .env to the current Gemini Flash model id
    gemini_embedding_model: str = ""  # set in .env to the current Gemini embedding model id
    # minimal | low | medium | high, or empty for the model default. Not every model takes every level:
    # gemini-3.8-flash rejects "minimal" (400 INVALID_ARGUMENT), which would make every LLM call fail.
    gemini_thinking_level: str = "low"
    anthropic_api_key: str = ""
    anthropic_model: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    embedding_dim: int = 768
    llm_temperature: float = 0.4
    llm_timeout_seconds: float = 4.0  # per LLM call; on timeout the agent falls back to rules/templates
    llm_history_messages: int = 12  # how many recent messages are sent to the LLM

    # --- Persona (spec §3) ---
    # Spec default: never reveal being an AI/bot. Set PERSONA_CONCEAL_AI=false to allow the agent to
    # say it is an automated assistant when asked (e.g. where disclosure is legally required).
    persona_conceal_ai: bool = True
    persona_name: str = ""  # optional first name the consultant uses; empty = never invent one
    company_name: str = ""  # optional agency name used in greetings
    reply_max_chars: int = 400  # per WhatsApp bubble ("no walls of text")
    reply_max_messages: int = 2  # bubbles per turn

    # --- Google Calendar (owner OAuth) ---
    google_client_id: str = ""
    google_client_secret: str = ""
    token_encryption_key: str = ""  # Fernet key, encrypts the owner's refresh token at rest
    google_oauth_scopes: list[str] = Field(default_factory=lambda: [
        "https://www.googleapis.com/auth/calendar.events",  # create the meeting + add the lead as attendee
        "https://www.googleapis.com/auth/calendar.freebusy",  # read busy times only (not event details)
    ])
    google_api_timeout_seconds: float = 10.0
    # Protects /auth/google/start (connecting the owner's calendar). Empty = the endpoint is disabled.
    admin_token: str = ""

    # --- Owner / scheduling ---
    default_owner_id: str = "default"
    owner_name: str = "Property Consultant"
    owner_email: str = ""
    owner_calendar_id: str = "primary"
    owner_timezone: str = "Asia/Dubai"
    working_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon=0 … Sun=6
    work_start: str = "09:00"
    work_end: str = "18:00"
    slot_duration_minutes: int = 30
    slot_count: int = 3
    slots_per_day: int = 1
    min_notice_minutes: int = 120
    slot_search_days: int = 14

    # --- Budget normalisation (units of local currency per 1 USD) ---
    rate_aed: float = 3.6725
    rate_sar: float = 3.75
    rate_qar: float = 3.64
    rate_kwd: float = 0.307  # KWD is basket-pegged; update periodically

    # --- RAG (Qdrant knowledge base) ---
    rag_collection: str = "gcc_knowledge"  # official-source legal/tax facts
    properties_collection: str = "properties"  # the owner's listings (optional)
    knowledge_dir: str = "knowledge_base/facts"
    rag_top_k: int = 3
    # Minimum cosine similarity for a passage to count as an answer; below it the agent defers to the call.
    # Calibrate with `scripts/ingest_kb.py search "..." --all-scores` after changing the embedding model.
    rag_score_threshold: float = 0.6
    rag_timeout_seconds: float = 2.5  # embed the question + search; on timeout the agent defers to the call
    rag_fact_max_age_days: int = 180  # ingest warns about facts not re-checked for this long
    # gemini-embedding-2 takes the task as a text prefix; set false for gemini-embedding-001 (uses task_type)
    gemini_embedding_task_prefixes: bool = True
    embedding_timeout_seconds: float = 30.0  # HTTP timeout of the embedding client (searches are capped above)
    qdrant_timeout_seconds: float = 5.0

    # --- Latency (spec DoD: reply in < 3 s, from webhook receipt to the first reply sent) ---
    latency_budget_seconds: float = 3.0
    # Kept free for sending the reply through the Meta Graph API after the agent is done.
    latency_send_reserve_seconds: float = 0.5
    # Kept free for writing the reply while extraction / knowledge search run; if they eat into it they are cut
    # short and rules / a deferral take over.
    latency_reply_reserve_seconds: float = 1.0
    # An LLM call or knowledge search is skipped (fallback used) when less time than this is left.
    latency_min_call_seconds: float = 0.4
    # Hard cap on one agent turn, including calendar writes the budget never cuts short; past it the lead gets a
    # "one moment" template.
    agent_timeout_seconds: float = 20.0
    # Google freeBusy answers are reused for this long, and fetched ahead when the lead is one answer away from
    # being qualified, so offering slots doesn't wait on Google. Booking always re-checks live freeBusy.
    calendar_busy_cache_seconds: int = 120
    # One tiny LLM call (and knowledge search) at startup, so the first lead doesn't pay connection setup.
    warmup_on_startup: bool = True

    # --- Session ---
    session_ttl_minutes: int = 60 * 24 * 30  # LangGraph checkpoint TTL (30 days)
    offered_slots_ttl_seconds: int = 60 * 60 * 48
    phone_lock_timeout_seconds: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
