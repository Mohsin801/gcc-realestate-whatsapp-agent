"""FastAPI entrypoint for the GCC Real Estate WhatsApp agent."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api import health
from core.config import get_settings
from core.logging import setup_logging
from db.session import dispose_engine, init_engine
from services.redis import create_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    init_engine(settings.database_url)
    app.state.redis = create_redis(settings.redis_url)
    try:
        yield
    finally:
        await app.state.redis.aclose()
        await dispose_engine()


app = FastAPI(title="GCC Real Estate WhatsApp Agent", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
