"""Data access for leads and chat logs."""
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Booking, ChatLog, Lead, Owner


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


async def delete_lead(session: AsyncSession, phone: str) -> dict[str, int]:
    """Delete a lead with its chat logs and bookings (dev reset). Children are deleted explicitly, so this does
    not rely on the database enforcing ON DELETE CASCADE (SQLite in tests doesn't by default).

    ``google_events`` counts confirmed bookings that have a Google Calendar event: those events are not
    cancelled here and stay in the owner's calendar."""
    lead = await get_lead(session, phone)
    if lead is None:
        return {"leads": 0, "chat_logs": 0, "bookings": 0, "google_events": 0}
    google_events = await session.scalar(
        select(func.count()).select_from(Booking).where(
            Booking.lead_id == lead.id, Booking.status == "CONFIRMED", Booking.google_event_id.is_not(None)
        )
    )
    bookings = await session.execute(delete(Booking).where(Booking.lead_id == lead.id))
    logs = await session.execute(delete(ChatLog).where(ChatLog.lead_id == lead.id))
    await session.execute(delete(Lead).where(Lead.id == lead.id))
    await session.commit()
    return {"leads": 1, "chat_logs": logs.rowcount, "bookings": bookings.rowcount, "google_events": google_events or 0}


async def latest_booking(session: AsyncSession, lead_id: int) -> Booking | None:
    return await session.scalar(
        select(Booking).where(Booking.lead_id == lead_id).order_by(Booking.id.desc()).limit(1)
    )


# --- owners -----------------------------------------------------------------------------------------
async def get_owner(session: AsyncSession, owner_id: str) -> Owner | None:
    return await session.get(Owner, owner_id)


async def upsert_owner(session: AsyncSession, owner_id: str, **fields) -> Owner:
    owner = await session.get(Owner, owner_id)
    if owner is None:
        owner = Owner(id=owner_id, **fields)
        session.add(owner)
    else:
        for key, value in fields.items():
            setattr(owner, key, value)
    await session.commit()
    return owner


# --- bookings ---------------------------------------------------------------------------------------
async def reserve_slot(session: AsyncSession, lead_id: int, owner_id: str, slot_iso: str) -> Booking | None:
    """Insert a CONFIRMED booking row for the slot. Returns None if the owner's slot is already taken
    (partial unique index ``uq_bookings_owner_slot_confirmed``), which makes this the booking lock."""
    booking = Booking(lead_id=lead_id, owner_id=owner_id, slot_iso=slot_iso, status="CONFIRMED")
    session.add(booking)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return None
    return booking


async def confirmed_slots(session: AsyncSession, owner_id: str, from_iso: str) -> set[str]:
    rows = await session.scalars(
        select(Booking.slot_iso).where(
            Booking.owner_id == owner_id, Booking.status == "CONFIRMED", Booking.slot_iso >= from_iso
        )
    )
    return set(rows.all())


async def get_booking(session: AsyncSession, booking_id: int) -> Booking | None:
    return await session.get(Booking, booking_id)


async def get_booking_by_event(session: AsyncSession, event_id: str) -> Booking | None:
    return await session.scalar(select(Booking).where(Booking.google_event_id == event_id))


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


async def reply_latencies(session: AsyncSession, since: datetime) -> list[tuple[str | None, int]]:
    """(language, latency_ms) of every reply since ``since``: webhook receipt -> first bubble accepted by Meta."""
    rows = await session.execute(
        select(ChatLog.language, ChatLog.latency_ms).where(
            ChatLog.direction == "out", ChatLog.latency_ms.is_not(None), ChatLog.created_at >= since
        )
    )
    return [(lang, ms) for lang, ms in rows.all()]
