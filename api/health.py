"""Liveness / readiness endpoint."""
import logging

import httpx
from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import text

from core.config import get_settings
from db.session import get_sessionmaker
from services.redis import get_redis

router = APIRouter()
log = logging.getLogger(__name__)


@router.get("/health")
async def health(request: Request, response: Response, redis=Depends(get_redis)) -> dict:
    checks: dict[str, str] = {}

    try:
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc.__class__.__name__}"

    try:
        async with get_sessionmaker()() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = f"error: {exc.__class__.__name__}"

    try:
        settings = get_settings()
        headers = {"api-key": settings.qdrant_api_key} if settings.qdrant_api_key else {}
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings.qdrant_url}/readyz", headers=headers)
        checks["qdrant"] = "ok" if r.status_code == 200 else f"error: HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        checks["qdrant"] = f"error: {exc.__class__.__name__}"

    healthy = all(v == "ok" for v in checks.values())
    if not healthy:
        response.status_code = 503
        log.warning("health check failed", extra={"checks": checks})
    return {"status": "ok" if healthy else "degraded", "checks": checks}
