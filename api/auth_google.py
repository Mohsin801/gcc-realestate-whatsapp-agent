"""One-time connection of the owner's Google Calendar (OAuth 2.0, owner's own account).

1. The owner opens  GET /auth/google/start?token=<ADMIN_TOKEN>  and is redirected to Google consent.
2. Google redirects to GET /auth/google/callback?code=...&state=...
3. The refresh token is stored Fernet-encrypted in ``owners.google_refresh_token_enc``.

``/start`` is protected by ADMIN_TOKEN (disabled when empty), because whoever connects their calendar
receives every booking. ``state`` is a single-use random value kept in Redis for 10 minutes (CSRF).
"""
import hmac
import logging
import secrets

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from core.config import get_settings
from core.crypto import TokenCipher
from db import repository
from tools.calendar_providers.base import CalendarError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth/google", tags=["auth"])

STATE_TTL_SECONDS = 600


def _redirect_uri() -> str:
    return get_settings().public_base_url.rstrip("/") + "/auth/google/callback"


def _page(title: str, body: str, status: int = 200) -> HTMLResponse:
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head>"
            f"<body style='font-family:sans-serif;max-width:40rem;margin:4rem auto'><h1>{title}</h1>"
            f"<p>{body}</p></body></html>")
    return HTMLResponse(html, status_code=status)


@router.get("/start")
async def start(request: Request, token: str = Query("")):
    settings = get_settings()
    if not settings.admin_token or not hmac.compare_digest(token.encode(), settings.admin_token.encode()):
        raise HTTPException(status_code=403, detail="forbidden")
    oauth = request.app.state.google_oauth
    if not oauth.configured:
        raise HTTPException(status_code=503, detail="GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not set")
    if not settings.token_encryption_key:
        raise HTTPException(status_code=503, detail="TOKEN_ENCRYPTION_KEY not set")
    state = secrets.token_urlsafe(32)
    await request.app.state.redis.set(f"oauth_state:{state}", "1", ex=STATE_TTL_SECONDS)
    return RedirectResponse(oauth.authorization_url(state, _redirect_uri()), status_code=302)


@router.get("/callback")
async def callback(request: Request, state: str = Query(""), code: str = Query(""), error: str = Query("")):
    settings = get_settings()
    if not state or not await request.app.state.redis.getdel(f"oauth_state:{state}"):
        return _page("Link expired", "This sign-in link is invalid or expired. Please start again.", 400)
    if error or not code:
        return _page("Not connected", "Google sign-in was cancelled, so the calendar was not connected.", 400)
    oauth = request.app.state.google_oauth
    try:
        tokens = await oauth.exchange_code(code, _redirect_uri())
    except CalendarError:
        log.exception("Google OAuth code exchange failed")
        return _page("Not connected", "Google sign-in failed. Please try again.", 502)
    refresh_token = tokens.get("refresh_token")
    granted = set((tokens.get("scope") or "").split())
    missing = [s for s in settings.google_oauth_scopes if s not in granted]
    if not refresh_token or missing:
        log.warning("Google OAuth incomplete", extra={"has_refresh_token": bool(refresh_token), "missing": missing})
        return _page("Not connected", "Please allow all the requested calendar permissions and try again.", 400)

    cipher = TokenCipher(settings.token_encryption_key)
    async with request.app.state.sessionmaker() as session:
        await repository.upsert_owner(
            session, settings.default_owner_id,
            name=settings.owner_name, email=settings.owner_email or None,
            calendar_id=settings.owner_calendar_id, timezone=settings.owner_timezone,
            working_hours={"days": settings.working_days, "start": settings.work_start, "end": settings.work_end},
            google_refresh_token_enc=cipher.encrypt(refresh_token),
        )
    log.info("owner Google Calendar connected", extra={"owner_id": settings.default_owner_id})
    return _page("Calendar connected ✅", "Bookings from WhatsApp will now appear in your Google Calendar.")
