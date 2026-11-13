"""FastAPI entrypoint for the GCC Real Estate WhatsApp agent."""
import logging
from contextlib import AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph
from agent.llm import LLMConfigError, build_llm
from agent.runner import AgentService
from api import auth_google, health, whatsapp
from core.config import Settings, get_settings
from core.crypto import TokenCipher
from core.logging import setup_logging
from db.session import dispose_engine, get_sessionmaker, init_engine
from services.redis import create_redis
from services.runtime import Runtime, set_runtime
from services.whatsapp_client import WhatsAppClient
from tools import calendar
from tools.calendar import CalendarEngine, CalendarScheduler
from tools.calendar_providers.google import GoogleOAuth

log = logging.getLogger(__name__)


async def open_checkpointer(stack: AsyncExitStack, settings: Settings):
    """Redis-backed LangGraph session store (needs the redis:8 image for its JSON/Search modules)."""
    try:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        return await stack.enter_async_context(  # from_conn_string also creates the indexes (asetup)
            AsyncRedisSaver.from_conn_string(
                settings.redis_url, ttl={"default_ttl": settings.session_ttl_minutes, "refresh_on_read": True}
            )
        )
    except Exception:  # noqa: BLE001 - keep answering leads; sessions just won't survive a restart
        log.exception("Redis checkpointer unavailable, falling back to in-memory sessions")
        return InMemorySaver()


def open_llm(settings: Settings):
    try:
        return build_llm(settings)
    except LLMConfigError:
        log.exception("LLM not configured; the agent will run on rules and templates only")
        return None


def open_cipher(settings: Settings) -> TokenCipher | None:
    if not settings.token_encryption_key:
        return None
    try:
        return TokenCipher(settings.token_encryption_key)
    except ValueError:
        log.exception("TOKEN_ENCRYPTION_KEY is not a valid Fernet key; calendar booking is disabled")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    init_engine(settings.database_url)
    app.state.redis = create_redis(settings.redis_url)
    wa_client = WhatsAppClient(
        token=settings.whatsapp_token,
        phone_number_id=settings.whatsapp_phone_number_id,
        api_version=settings.whatsapp_api_version,
    )
    google_http = httpx.AsyncClient(timeout=settings.google_api_timeout_seconds)
    app.state.sessionmaker = get_sessionmaker()
    app.state.google_oauth = GoogleOAuth(
        google_http, settings.google_client_id, settings.google_client_secret, settings.google_oauth_scopes
    )
    calendar_engine = CalendarEngine(
        settings, get_sessionmaker(), oauth=app.state.google_oauth, cipher=open_cipher(settings)
    )
    calendar.configure(calendar_engine)
    async with AsyncExitStack() as stack:
        checkpointer = await open_checkpointer(stack, settings)
        graph = build_graph(
            llm=open_llm(settings), settings=settings, checkpointer=checkpointer,
            scheduler=CalendarScheduler(calendar_engine, app.state.redis),
        )
        agent = AgentService(graph, get_sessionmaker())
        set_runtime(Runtime(sender=wa_client, sessionmaker=get_sessionmaker(), run_agent=agent))
        try:
            yield
        finally:
            set_runtime(None)
            calendar.configure(None)
            await wa_client.aclose()
            await google_http.aclose()
            await app.state.redis.aclose()
            await dispose_engine()


app = FastAPI(title="GCC Real Estate WhatsApp Agent", version="0.3.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(whatsapp.router)
app.include_router(auth_google.router)
