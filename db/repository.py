"""Data access for leads and chat logs."""
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Booking, ChatLog, Lead


async def get_lead(session: AsyncSession, phone: str) -> Lead | None:
    return await session.scalar(select(Lead).where(Lead.phone == phone))


async def get_or_create_lead(
    session: AsyncSession, phone: str, profile_name: str | None = None, language: str | None = None
) -> Lead:
    """Fetch the lead by phone (creating it if new) and refresh name/language when provided.

    Processing per phone is serialised by a Redis lock; the IntegrityError branch only covers
    the rare case where that lock could not be taken.
    """
    lead = await get_lead(session, phone)
    if lead is None:
        lead = Lead(phone=phone, profile_name=profile_name, language=language or "en")
        session.add(lead)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            lead = await get_lead(session, phone)
            assert lead is not None
        return lead

    changed = False
    if profile_name and lead.profile_name != profile_name:
        lead.profile_name, changed = profile_name, True
    if language and lead.language != language:
        lead.language, changed = language, True
    if changed:
        await session.commit()
    return lead


_PROFILE_FIELDS = {
    "target_country", "intent", "budget_usd", "budget_original_amount", "budget_original_currency",
    "timeline", "is_qualified", "booking_status", "language",
}


async def update_lead_profile(session: AsyncSession, phone: str, **fields) -> Lead | None:
    """Write the agent's qualification results to the durable lead record (created if missing)."""
    unknown = set(fields) - _PROFILE_FIELDS
    if unknown:
        raise ValueError(f"not lead profile fields: {sorted(unknown)}")
    lead = await get_or_create_lead(session, phone)
    changed = False
    for key, value in fields.items():
        if value is not None and getattr(lead, key) != value:
            setattr(lead, key, value)
            changed = True
    if changed:
        await session.commit()
    return lead


async def latest_booking(session: AsyncSession, lead_id: int) -> Booking | None:
    return await session.scalar(
        select(Booking).where(Booking.lead_id == lead_id).order_by(Booking.id.desc()).limit(1)
    )


async def log_message(
    session: AsyncSession,
    lead_id: int,
    direction: str,
    text: str,
    language: str | None = None,
    wamid: str | None = None,
    latency_ms: int | None = None,
) -> ChatLog | None:
    """Insert a chat log row. Returns None if this inbound wamid was already logged."""
    row = ChatLog(
        lead_id=lead_id, direction=direction, text=text, language=language, wamid=wamid, latency_ms=latency_ms
    )
    session.add(row)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return None
    return row
