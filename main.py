"""FastAPI entrypoint for the GCC Real Estate WhatsApp agent."""
import logging
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph
from agent.llm import LLMConfigError, build_llm
from agent.runner import AgentService
from api import health, whatsapp
from core.config import Settings, get_settings
from core.logging import setup_logging
from db.session import dispose_engine, get_sessionmaker, init_engine
from services.redis import create_redis
from services.runtime import Runtime, set_runtime
from services.whatsapp_client import WhatsAppClient

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
    async with AsyncExitStack() as stack:
        checkpointer = await open_checkpointer(stack, settings)
        graph = build_graph(llm=open_llm(settings), settings=settings, checkpointer=checkpointer)
        agent = AgentService(graph, get_sessionmaker())
        set_runtime(Runtime(sender=wa_client, sessionmaker=get_sessionmaker(), run_agent=agent))
        try:
            yield
        finally:
            set_runtime(None)
            await wa_client.aclose()
            await app.state.redis.aclose()
            await dispose_engine()


app = FastAPI(title="GCC Real Estate WhatsApp Agent", version="0.2.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(whatsapp.router)
