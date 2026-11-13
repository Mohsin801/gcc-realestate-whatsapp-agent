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
    gemini_thinking_level: str = "minimal"  # minimal | low | medium | high; minimal keeps replies fast
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

    # --- RAG ---
    rag_collection: str = "gcc_knowledge"
    properties_collection: str = "properties"
    rag_top_k: int = 4
    rag_score_threshold: float = 0.6

    # --- Session ---
    session_ttl_minutes: int = 60 * 24 * 30  # LangGraph checkpoint TTL (30 days)
    offered_slots_ttl_seconds: int = 60 * 60 * 48
    phone_lock_timeout_seconds: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
