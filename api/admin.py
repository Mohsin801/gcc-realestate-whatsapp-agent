"""Development-only admin endpoints.

``POST /admin/reset/{phone}`` wipes one lead's conversation (``services/reset.py``) so the next message starts
fresh, e.g. between demos. It needs the ``X-Admin-Token`` header (a header, so the token stays out of URLs and
tunnel/proxy logs) and only exists when ``APP_ENV`` is ``development`` or ``test``; anywhere else it answers 404.
``scripts/reset_lead.py`` calls it with the token from ``.env``.
"""
import hmac
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from core.config import get_settings
from services.reset import normalize_phone, reset_lead

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

DEV_ENVS = {"development", "test"}


def require_dev_admin(x_admin_token: str = Header("")) -> None:
    settings = get_settings()
    if settings.app_env.lower() not in DEV_ENVS:
        raise HTTPException(status_code=404, detail="Not Found")
    if not settings.admin_token or not hmac.compare_digest(x_admin_token.encode(), settings.admin_token.encode()):
        raise HTTPException(status_code=403, detail="forbidden")


@router.post("/reset/{phone}", dependencies=[Depends(require_dev_admin)])
async def reset(phone: str, request: Request) -> dict:
    try:
        phone = normalize_phone(phone)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    state = request.app.state
    result = await reset_lead(
        phone, redis=state.redis, sessionmaker=state.sessionmaker,
        checkpointer=getattr(state, "checkpointer", None), scheduler=getattr(state, "scheduler", None),
    )
    log.warning("dev reset of lead", extra=result.as_dict())
    return result.as_dict()
