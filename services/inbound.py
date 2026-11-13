"""Parse Meta WhatsApp Cloud API webhook payloads into IncomingMessage objects."""
import logging
from dataclasses import dataclass

from agent.language import detect_language

log = logging.getLogger(__name__)

# Message types we never reply to (no user intent to answer).
IGNORED_TYPES = {"reaction", "system", "ephemeral", "request_welcome"}


@dataclass(frozen=True)
class IncomingMessage:
    wamid: str
    phone: str  # sender wa_id, digits only, e.g. "971501234567"
    profile_name: str | None
    msg_type: str
    text: str | None  # None => media we can't read (voice note, image without caption, ...)
    language: str | None  # 'en' | 'ar' | None (no letters: keep lead's previous language)
    received_at: float  # time.monotonic() at webhook receipt, for latency_ms


def _extract_text(message: dict) -> str | None:
    mtype = message.get("type")
    if mtype == "text":
        return (message.get("text") or {}).get("body")
    if mtype == "interactive":
        inter = message.get("interactive") or {}
        reply = inter.get("button_reply") or inter.get("list_reply") or {}
        return reply.get("title")
    if mtype == "button":
        return (message.get("button") or {}).get("text")
    if mtype in {"image", "video", "document"}:
        return (message.get(mtype) or {}).get("caption")  # caption if any, else None
    return None


def parse_incoming(payload: dict, received_at: float, phone_number_id: str | None = None) -> list[IncomingMessage]:
    """Extract user messages from a webhook payload. Status callbacks and other events are skipped.

    If `phone_number_id` is given, events addressed to other business numbers are ignored.
    """
    if payload.get("object") != "whatsapp_business_account":
        return []

    out: list[IncomingMessage] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            if change.get("field") != "messages":
                continue
            value = change.get("value") or {}
            meta_id = (value.get("metadata") or {}).get("phone_number_id")
            if phone_number_id and meta_id and meta_id != phone_number_id:
                log.info("ignoring event for another phone_number_id", extra={"phone_number_id": meta_id})
                continue

            names = {c.get("wa_id"): (c.get("profile") or {}).get("name") for c in value.get("contacts") or []}

            for message in value.get("messages") or []:  # 'statuses' (sent/delivered/read) are ignored
                mtype = message.get("type", "unknown")
                wamid, sender = message.get("id"), message.get("from")
                if not wamid or not sender or mtype in IGNORED_TYPES:
                    continue
                text = _extract_text(message)
                if text is not None:
                    text = text.strip() or None
                out.append(
                    IncomingMessage(
                        wamid=wamid,
                        phone=sender,
                        profile_name=names.get(sender),
                        msg_type=mtype,
                        text=text,
                        language=detect_language(text),
                        received_at=received_at,
                    )
                )
    return out
