"""Development only: wipe one lead's conversation, so their next message starts a fresh session (testing, demos).

Clears everything the agent keeps per phone number:
  * the LangGraph checkpoint (Redis, ``thread_id = phone``);
  * the offered slots (Redis ``offered:{phone}``);
  * the Postgres ``leads`` row with its ``chat_logs`` and ``bookings`` (the runner would otherwise re-seed the
    qualification answers from ``leads`` when the next session starts);
  * the demo calendar's in-memory offers and booking (``CALENDAR_DEMO``), when the running app does the reset.

It does not cancel Google Calendar events: ``google_events`` in the result counts bookings whose event stays in
the owner's calendar.

Reached through ``POST /admin/reset/{phone}`` (``api/admin.py``) or ``scripts/reset_lead.py``.
"""
import re
from dataclasses import asdict, dataclass

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from db import repository
from services.redis import clear_offered_slots, phone_lock


def normalize_phone(raw: str) -> str:
    """WhatsApp ids are the full number in digits (e.g. ``923001234567``); accept ``+92 300-1234567`` too."""
    digits = re.sub(r"[\s()+\-.]", "", raw or "")
    if not re.fullmatch(r"\d{7,15}", digits):
        raise ValueError("phone must be 7-15 digits including the country code")
    return digits


@dataclass
class ResetResult:
    phone: str
    checkpoint_cleared: bool
    demo_calendar_cleared: bool
    leads: int
    chat_logs: int
    bookings: int
    google_events: int  # bookings whose Google Calendar event was left in place

    def as_dict(self) -> dict:
        return asdict(self)


async def reset_lead(phone: str, *, redis: Redis, sessionmaker: async_sessionmaker[AsyncSession],
                     checkpointer=None, scheduler=None) -> ResetResult:
    # Hold the phone lock so a turn that is still running can't write its state back after the wipe.
    async with phone_lock(redis, phone):
        if checkpointer is not None:
            await checkpointer.adelete_thread(phone)
        await clear_offered_slots(redis, phone)
        forget = getattr(scheduler, "forget", None)
        if forget is not None:
            forget(phone)
        async with sessionmaker() as session:
            counts = await repository.delete_lead(session, phone)
    return ResetResult(phone=phone, checkpoint_cleared=checkpointer is not None,
                       demo_calendar_cleared=forget is not None, **counts)
