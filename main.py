"""FastAPI entrypoint for the GCC Real Estate WhatsApp agent."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.runner import run_agent
from api import health, whatsapp
from core.config import get_settings
from core.logging import setup_logging
from db.session import dispose_engine, get_sessionmaker, init_engine
from services.redis import create_redis
from services.runtime import Runtime, set_runtime
from services.whatsapp_client import WhatsAppClient


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
    set_runtime(Runtime(sender=wa_client, sessionmaker=get_sessionmaker(), run_agent=run_agent))
    try:
        yield
    finally:
        set_runtime(None)
        await wa_client.aclose()
        await app.state.redis.aclose()
        await dispose_engine()


app = FastAPI(title="GCC Real Estate WhatsApp Agent", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(whatsapp.router)
