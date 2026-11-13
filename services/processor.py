"""Background processing of one inbound WhatsApp message (spec §5 step 3 onward)."""
import asyncio
import logging
import time

from redis.asyncio import Redis
from redis.exceptions import LockError

from agent.templates import TEMPORARY_ERROR, UNSUPPORTED_MEDIA, pick
from db import repository
from services.inbound import IncomingMessage
from services.redis import phone_lock
from services.runtime import Runtime

log = logging.getLogger(__name__)


def _elapsed_ms(received_at: float) -> int:
    return int((time.monotonic() - received_at) * 1000)


async def process_message(msg: IncomingMessage, *, redis: Redis, runtime: Runtime) -> None:
    """Serialise per lead, then run the agent and reply. Never raises: the lead always gets an answer."""
    lock = phone_lock(redis, msg.phone)
    acquired = False
    try:
        acquired = await lock.acquire()
    except Exception:  # noqa: BLE001 - Redis hiccup must not stop the reply
        log.warning("phone lock unavailable, processing without it", extra={"phone": msg.phone}, exc_info=True)
    if not acquired:
        log.warning("phone lock not acquired in time, processing anyway", extra={"phone": msg.phone})
    try:
        await _process(msg, runtime)
    finally:
        if acquired:
            try:
                await lock.release()
            except LockError:
                log.warning("phone lock expired before release", extra={"phone": msg.phone})


async def _process(msg: IncomingMessage, runtime: Runtime) -> None:
    await runtime.sender.mark_read(msg.wamid, typing=True)

    language = msg.language or "en"
    lead_id: int | None = None
    try:
        async with runtime.sessionmaker() as session:
            lead = await repository.get_or_create_lead(session, msg.phone, msg.profile_name, msg.language)
            lead_id = lead.id
            language = msg.language or lead.language or "en"  # digits/emoji only -> keep previous language
            inbound_text = msg.text if msg.text is not None else f"[{msg.msg_type}]"
            logged = await repository.log_message(
                session, lead_id, "in", inbound_text, language=msg.language, wamid=msg.wamid
            )
            if logged is None:
                log.info("wamid already logged, skipping", extra={"wamid": msg.wamid})
                return
    except Exception:  # noqa: BLE001 - DB outage must not leave the lead unanswered
        log.exception("failed to persist inbound message", extra={"wamid": msg.wamid})

    if msg.text is None:
        replies = [pick(UNSUPPORTED_MEDIA, language)]
    else:
        try:
            replies = await asyncio.wait_for(
                runtime.run_agent(msg, language), timeout=runtime.agent_timeout_seconds
            )
            replies = [r for r in replies if r and r.strip()] or [pick(TEMPORARY_ERROR, language)]
        except Exception:  # noqa: BLE001
            log.exception("agent failed", extra={"wamid": msg.wamid, "phone": msg.phone})
            replies = [pick(TEMPORARY_ERROR, language)]

    for i, reply in enumerate(replies):
        latency_ms = _elapsed_ms(msg.received_at)
        try:
            await runtime.sender.send_text(msg.phone, reply)
        except Exception:  # noqa: BLE001
            log.exception("failed to send reply", extra={"phone": msg.phone})
            return
        if i == 0:
            log.info("reply sent", extra={"phone": msg.phone, "latency_ms": latency_ms, "language": language})
        if lead_id is not None:
            try:
                async with runtime.sessionmaker() as session:
                    await repository.log_message(
                        session, lead_id, "out", reply, language=language, latency_ms=latency_ms if i == 0 else None
                    )
            except Exception:  # noqa: BLE001
                log.exception("failed to persist outbound message", extra={"phone": msg.phone})
