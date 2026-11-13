"""WhatsApp Cloud API webhook router & deduplication (spec §5)."""
import hashlib
import hmac
import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from core.config import get_settings
from services.inbound import parse_incoming
from services.processor import process_message
from services.redis import claim_wamid, get_redis
from services.runtime import get_runtime

router = APIRouter()
log = logging.getLogger(__name__)


@router.get("/whatsapp/webhook", response_class=PlainTextResponse)
async def verify_webhook(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
) -> str:
    """Meta's one-time subscription handshake: echo hub.challenge if the verify token matches."""
    expected = get_settings().whatsapp_verify_token
    if (
        hub_mode == "subscribe"
        and expected
        and hub_verify_token
        and hmac.compare_digest(hub_verify_token, expected)
    ):
        return hub_challenge or ""
    raise HTTPException(status_code=403, detail="Verification failed")


async def verify_meta_signature(request: Request) -> None:
    """Validate X-Hub-Signature-256 (HMAC-SHA256 of the raw body with the app secret)."""
    settings = get_settings()
    secret = settings.whatsapp_app_secret
    if not secret:
        if settings.app_env == "development":
            log.warning("WHATSAPP_APP_SECRET not set: skipping webhook signature check (development only)")
            return
        log.error("WHATSAPP_APP_SECRET not set: rejecting webhook")
        raise HTTPException(status_code=403, detail="Signature check not configured")

    header = request.headers.get("X-Hub-Signature-256", "")
    body = await request.body()
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(header, expected):
        raise HTTPException(status_code=403, detail="Invalid signature")


@router.post("/whatsapp/webhook", dependencies=[Depends(verify_meta_signature)])
async def handle_incoming_msg(payload: dict, background_tasks: BackgroundTasks, redis=Depends(get_redis)):
    # 1. Immediately return HTTP 200 OK (< 200ms Meta SLA): only parsing + one Redis SET NX per message here.
    received_at = time.monotonic()
    messages = parse_incoming(payload, received_at, get_settings().whatsapp_phone_number_id or None)
    for msg in messages:
        # 2. Redis lock check using WhatsApp message ID ('wamid') to prevent duplicate responses
        if not await claim_wamid(redis, msg.wamid):
            log.info("duplicate wamid ignored", extra={"wamid": msg.wamid})
            continue
        # 3. Language (EN/AR) was detected in parse_incoming -> pass to background agent execution
        background_tasks.add_task(process_message, msg, redis=redis, runtime=get_runtime())
    return {"status": "ok"}
