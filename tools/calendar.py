"""Live calendar engine (spec §4): real-time slot fetch and booking in the owner's Google Calendar.

Spec functions (signatures fixed by the spec):
    get_available_slots(owner_id, timezone="Asia/Dubai") -> list[str]
    create_calendar_event(lead_phone, lead_name, slot_iso) -> bool
They delegate to the process-wide ``CalendarEngine`` set with ``configure()`` at startup. The booking
reference (Google event id) and Meet link are stored in Postgres ``bookings``.

Booking is race-safe: the slot is first *reserved* by inserting a CONFIRMED ``bookings`` row (a partial
unique index allows one per owner slot), then re-checked against Google freeBusy, then the event is
created. Any failure releases the reservation.

``CalendarScheduler`` adapts the engine to the agent's ``Scheduler`` protocol: it remembers the offered
slots per lead in Redis (as UTC ISO strings) and renders labels in the lead's timezone, chosen from
the phone country code (+971/+966/+974/+965) with the owner's timezone as fallback.

Latency: freeBusy answers are cached per owner for ``CALENDAR_BUSY_CACHE_SECONDS`` and fetched ahead
(``CalendarScheduler.prefetch``) when a lead is one answer away from being offered slots. Bookings made since
the fetch still count, because ``bookings`` is always read live, and ``book()`` always re-checks live freeBusy
before it creates the event. The cache lives in the process (one per worker).
"""
import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from redis.asyncio import Redis
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agent.services import BookingResult
from core.config import Settings
from core.crypto import TokenCipher
from core.timezones import lead_timezone
from db import repository
from db.models import Booking
from services.redis import clear_offered_slots, load_offered_slots, save_offered_slots
from tools.calendar_providers.base import CalendarError, CalendarNotConnected, CalendarProvider
from tools.calendar_providers.google import GoogleCalendarProvider, GoogleOAuth
from tools.scheduling import (
    WorkingHours,
    candidate_slots,
    format_slot,
    from_iso,
    overlaps,
    parse_hhmm,
    pick_slots,
    to_iso,
)

log = logging.getLogger(__name__)

_MARKETS = {"DUBAI": "UAE (Dubai)", "SAUDI": "Saudi Arabia", "QATAR": "Qatar", "KUWAIT": "Kuwait"}
_PURPOSES = {"INVESTMENT": "Investment", "PERSONAL_USE": "Personal use", "GOLDEN_VISA": "Golden Visa"}
_TIMELINES = {"IMMEDIATE": "Immediate", "3_MONTHS": "Within 3 months", "OFF_PLAN": "Off-plan"}


@dataclass(frozen=True)
class OwnerConfig:
    id: str
    name: str
    calendar_id: str
    hours: WorkingHours
    refresh_token_enc: str | None


@dataclass(frozen=True)
class _BusyWindow:
    fetched_at: datetime
    start: datetime
    end: datetime
    busy: list[tuple[datetime, datetime]]


@dataclass(frozen=True)
class BookingOutcome:
    ok: bool
    taken: bool = False
    event_id: str | None = None
    meet_link: str | None = None
    booking_id: int | None = None


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CalendarEngine:
    def __init__(
        self,
        settings: Settings,
        sessionmaker: async_sessionmaker[AsyncSession],
        *,
        oauth: GoogleOAuth | None = None,
        cipher: TokenCipher | None = None,
        provider_factory: Callable[[OwnerConfig], CalendarProvider] | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ):
        self.settings = settings
        self.sessionmaker = sessionmaker
        self.oauth = oauth
        self.cipher = cipher
        self._provider_factory = provider_factory
        self._clock = clock
        self._busy_cache: dict[str, _BusyWindow] = {}

    def now(self) -> datetime:
        return self._clock()

    @property
    def duration(self) -> timedelta:
        return timedelta(minutes=self.settings.slot_duration_minutes)

    async def owner_config(self, owner_id: str | None = None) -> OwnerConfig:
        s = self.settings
        owner_id = owner_id or s.default_owner_id
        async with self.sessionmaker() as session:
            row = await repository.get_owner(session, owner_id)
        wh = (row.working_hours if row else None) or {}
        hours = WorkingHours(
            timezone=(row.timezone if row and row.timezone else s.owner_timezone),
            days=tuple(wh.get("days", s.working_days)),
            start=parse_hhmm(wh.get("start", s.work_start)),
            end=parse_hhmm(wh.get("end", s.work_end)),
        )
        return OwnerConfig(
            id=owner_id,
            name=(row.name if row else s.owner_name),
            calendar_id=(row.calendar_id if row and row.calendar_id else s.owner_calendar_id),
            hours=hours,
            refresh_token_enc=row.google_refresh_token_enc if row else None,
        )

    def provider(self, owner: OwnerConfig) -> CalendarProvider:
        if self._provider_factory is not None:
            return self._provider_factory(owner)
        if not owner.refresh_token_enc:
            raise CalendarNotConnected(f"owner {owner.id!r} has not connected Google Calendar (/auth/google/start)")
        if self.oauth is None or not self.oauth.configured or self.cipher is None:
            raise CalendarNotConnected("GOOGLE_CLIENT_ID/SECRET and TOKEN_ENCRYPTION_KEY must be set")
        return GoogleCalendarProvider(self.oauth, self.cipher.decrypt(owner.refresh_token_enc), owner.calendar_id)

    def _candidates(self, owner: OwnerConfig, now: datetime) -> list[datetime]:
        s = self.settings
        return list(candidate_slots(now, owner.hours, self.duration, timedelta(minutes=s.min_notice_minutes),
                                    s.slot_search_days))

    async def find_open_slots(self, owner_id: str | None = None) -> list[datetime]:
        """Next ``slot_count`` free slots (UTC): working hours, min notice, freeBusy, existing bookings,
        at most ``slots_per_day`` per day."""
        owner = await self.owner_config(owner_id)
        provider = self.provider(owner)
        candidates = self._candidates(owner, self.now())
        if not candidates:
            return []
        busy = await self._busy(owner.id, provider, candidates[0], candidates[-1] + self.duration)
        async with self.sessionmaker() as session:
            taken = await repository.confirmed_slots(session, owner.id, to_iso(candidates[0]))
        s = self.settings
        return pick_slots(candidates, busy, taken, duration=self.duration, count=s.slot_count,
                          per_day=s.slots_per_day, owner_tz=owner.hours.timezone)

    async def _busy(self, owner_id: str, provider: CalendarProvider, start: datetime,
                    end: datetime) -> list[tuple[datetime, datetime]]:
        """freeBusy for [start, end], from the cache when a recent fetch covers the window."""
        ttl = timedelta(seconds=self.settings.calendar_busy_cache_seconds)
        now = self.now()
        hit = self._busy_cache.get(owner_id)
        if ttl and hit and now - hit.fetched_at < ttl and hit.start <= start and end <= hit.end:
            return hit.busy
        fetch_end = end + timedelta(days=1)  # margin: the search window slides forward while cached
        busy = await provider.free_busy(start, fetch_end)
        if ttl:
            self._busy_cache[owner_id] = _BusyWindow(now, start, fetch_end, busy)
        return busy

    def forget_busy(self, owner_id: str | None = None) -> None:
        """Drop cached freeBusy (all owners when ``owner_id`` is None)."""
        if owner_id is None:
            self._busy_cache.clear()
        else:
            self._busy_cache.pop(owner_id, None)

    async def book(self, lead_phone: str, lead_name: str | None, slot_iso: str,
                   owner_id: str | None = None) -> BookingOutcome:
        try:
            owner = await self.owner_config(owner_id)
            provider = self.provider(owner)
            slot = from_iso(slot_iso)
        except (CalendarError, ValueError):
            log.exception("cannot book: calendar not available or bad slot", extra={"slot": slot_iso})
            return BookingOutcome(ok=False)
        if slot not in set(self._candidates(owner, self.now())):  # in the past, too soon, or off-hours
            return BookingOutcome(ok=False, taken=True)
        iso, end = to_iso(slot), slot + self.duration
        owner_zone = ZoneInfo(owner.hours.timezone)

        async with self.sessionmaker() as session:
            lead = await repository.get_or_create_lead(session, lead_phone, lead_name)
            # read the lead before reserving: a failed reservation rolls back and expires loaded objects
            description = self._description(lead, lead_phone)
            title_name = lead.profile_name or f"+{lead_phone}"
            booking = await repository.reserve_slot(session, lead.id, owner.id, iso)
        if booking is None:
            return BookingOutcome(ok=False, taken=True)

        try:
            if overlaps(slot, end, await provider.free_busy(slot, end)):
                await self._release(booking.id)
                return BookingOutcome(ok=False, taken=True)
            event = await provider.insert_event(
                summary=f"Property consultation: {title_name}", description=description,
                start=slot.astimezone(owner_zone), end=end.astimezone(owner_zone), timezone=owner.hours.timezone,
                request_id=uuid.uuid4().hex,
            )
        except Exception:  # noqa: BLE001 - release the slot, the agent hands off to the consultant
            log.exception("calendar booking failed", extra={"slot": iso})
            await self._release(booking.id)
            return BookingOutcome(ok=False)

        try:
            async with self.sessionmaker() as session:
                await session.execute(
                    update(Booking).where(Booking.id == booking.id).values(
                        google_event_id=event.event_id, meet_link=event.meet_link, email_requested=True
                    )
                )
                await session.commit()
        except Exception:  # noqa: BLE001 - the event exists; log loudly so it can be reconciled
            log.exception("event created but booking row not updated",
                          extra={"event_id": event.event_id, "booking_id": booking.id})
        return BookingOutcome(ok=True, event_id=event.event_id, meet_link=event.meet_link, booking_id=booking.id)

    async def add_attendee(self, booking_ref: str, email: str, owner_id: str | None = None) -> bool:
        """Add the lead's email to the event; Google sends them the calendar invite (sendUpdates=all)."""
        try:
            owner = await self.owner_config(owner_id)
            await self.provider(owner).set_attendee(booking_ref, email)
        except Exception:  # noqa: BLE001
            log.exception("could not add attendee", extra={"event_id": booking_ref})
            return False
        try:
            async with self.sessionmaker() as session:
                await session.execute(
                    update(Booking).where(Booking.google_event_id == booking_ref).values(lead_email=email)
                )
                await session.commit()
        except Exception:  # noqa: BLE001
            log.exception("attendee added but email not saved", extra={"event_id": booking_ref})
        return True

    async def _release(self, booking_id: int) -> None:
        try:
            async with self.sessionmaker() as session:
                await session.execute(delete(Booking).where(Booking.id == booking_id))
                await session.commit()
        except Exception:  # noqa: BLE001
            log.exception("could not release slot reservation", extra={"booking_id": booking_id})

    @staticmethod
    def _description(lead, phone: str) -> str:
        lines = [f"Lead: {lead.profile_name or 'unknown'}", f"WhatsApp: +{phone}"]
        if lead.target_country:
            lines.append(f"Market: {_MARKETS.get(lead.target_country, lead.target_country)}")
        if lead.intent:
            lines.append(f"Purpose: {_PURPOSES.get(lead.intent, lead.intent)}")
        if lead.budget_usd:
            budget = f"Budget: ~USD {lead.budget_usd:,.0f}"
            if lead.budget_original_amount and lead.budget_original_currency:
                budget += f" ({lead.budget_original_amount:,.0f} {lead.budget_original_currency})"
            lines.append(budget)
        if lead.timeline:
            lines.append(f"Timeline: {_TIMELINES.get(lead.timeline, lead.timeline)}")
        lines.append(f"Language: {'Arabic' if lead.language == 'ar' else 'English'}")
        lines.append("Booked via WhatsApp.")
        return "\n".join(lines)


# --- spec §4 functions ------------------------------------------------------------------------------
_engine: CalendarEngine | None = None


def configure(engine: CalendarEngine | None) -> None:
    global _engine
    _engine = engine


def get_engine() -> CalendarEngine:
    if _engine is None:
        raise RuntimeError("calendar engine not configured (app lifespan has not run)")
    return _engine


async def get_available_slots(owner_id: str, timezone: str = "Asia/Dubai") -> list[str]:
    """Fetches next 3 available 30-min slots from Owner's Calendar.

    Returns formatted strings, e.g. ["Tomorrow, Tue 18 Nov at 4:00 PM GST", "Friday 21 Nov at 11:00 AM GST"].
    Raises ``CalendarNotConnected`` if the owner has not connected their calendar.
    """
    engine = get_engine()
    slots = await engine.find_open_slots(owner_id)
    now = engine.now()
    return [format_slot(slot, timezone, "en", now) for slot in slots]


async def create_calendar_event(lead_phone: str, lead_name: str, slot_iso: str) -> bool:
    """Inserts confirmed event directly into Owner's Google Calendar (with a Google Meet link) and records
    the booking reference (Google event id) in ``bookings``. The calendar invite email is sent when the
    lead's email is added with ``add_attendee_to_event``."""
    return (await get_engine().book(lead_phone, lead_name, slot_iso)).ok


async def add_attendee_to_event(booking_ref: str, email: str) -> bool:
    """Adds the lead as attendee; Google emails them the calendar invite."""
    return await get_engine().add_attendee(booking_ref, email)


# --- agent adapter ----------------------------------------------------------------------------------
class CalendarScheduler:
    """Implements ``agent.services.Scheduler`` on top of ``CalendarEngine``."""

    def __init__(self, engine: CalendarEngine, redis: Redis, owner_id: str | None = None):
        self.engine = engine
        self.redis = redis
        self.owner_id = owner_id
        self._tasks: set[asyncio.Task] = set()

    def prefetch(self) -> None:
        """Fill the freeBusy cache in the background so the coming slot offer doesn't wait on Google."""
        task = asyncio.create_task(self.warm())
        self._tasks.add(task)  # keep a reference until it finishes
        task.add_done_callback(self._tasks.discard)

    async def warm(self) -> bool:
        """Fetch open slots once (fills the cache and the OAuth token). Never raises."""
        try:
            await self.engine.find_open_slots(self.owner_id)
            return True
        except CalendarNotConnected as exc:
            log.info("calendar prefetch skipped: %s", exc)
        except Exception:  # noqa: BLE001 - an optimisation; the real offer reports errors
            log.info("calendar prefetch failed", exc_info=True)
        return False

    async def _lead_tz(self, phone: str) -> str:
        owner = await self.engine.owner_config(self.owner_id)
        return lead_timezone(phone, owner.hours.timezone)

    async def _labels(self, phone: str, isos: list[str], language: str) -> list[str]:
        tz, now = await self._lead_tz(phone), self.engine.now()
        return [format_slot(from_iso(iso), tz, language, now) for iso in isos]

    async def offer_slots(self, phone: str, country: str | None, language: str) -> list[str]:
        try:
            slots = await self.engine.find_open_slots(self.owner_id)
        except CalendarError:
            log.exception("could not fetch open slots")
            return []
        if not slots:
            return []
        isos = [to_iso(s) for s in slots]
        await save_offered_slots(self.redis, phone, isos)
        return await self._labels(phone, isos, language)

    async def offered_labels(self, phone: str, language: str) -> list[str]:
        """Re-rendered from the stored instants on every call, so 'Today'/'Tomorrow' stay correct."""
        return await self._labels(phone, await load_offered_slots(self.redis, phone), language)

    async def book(self, phone: str, lead_name: str | None, choice: int, language: str) -> BookingResult:
        isos = await load_offered_slots(self.redis, phone)
        if not 1 <= choice <= len(isos):
            return BookingResult(ok=False, slot_taken=True)  # offer expired: the agent re-offers
        iso = isos[choice - 1]
        outcome = await self.engine.book(phone, lead_name, iso, self.owner_id)
        if not outcome.ok:
            return BookingResult(ok=False, slot_taken=outcome.taken)
        await clear_offered_slots(self.redis, phone)
        [label] = await self._labels(phone, [iso], language)
        return BookingResult(ok=True, slot_iso=iso, slot_label=label, meet_link=outcome.meet_link)

    async def add_email(self, phone: str, email: str) -> bool:
        async with self.engine.sessionmaker() as session:
            lead = await repository.get_lead(session, phone)
            booking = await repository.latest_booking(session, lead.id) if lead else None
        if booking is None or booking.status != "CONFIRMED" or not booking.google_event_id:
            return False
        return await self.engine.add_attendee(booking.google_event_id, email, self.owner_id)
