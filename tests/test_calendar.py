"""tools/calendar.py: spec §4 functions, CalendarEngine booking, CalendarScheduler (agent adapter)."""
import inspect
from datetime import UTC, datetime, timedelta

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from agent.graph import build_graph
from agent.prompts import Extraction
from agent.runner import AgentService
from core.config import Settings
from db.models import Booking, Lead, Owner
from services.inbound import IncomingMessage
from tools import calendar
from tools.calendar import CalendarEngine, CalendarScheduler
from tools.calendar_providers.base import CalendarError, CreatedEvent
from tools.scheduling import to_iso
from tests.agent_fakes import FakeLLM

NOW = datetime(2026, 11, 16, 6, 0, tzinfo=UTC)  # Mon 16 Nov 2026, 10:00 Dubai
MON_12 = "2026-11-16T08:00:00+00:00"  # Mon 12:00 Dubai (first slot after 2h notice)
TUE_09 = "2026-11-17T05:00:00+00:00"
WED_09 = "2026-11-18T05:00:00+00:00"
UAE, KSA, UK = "971501234567", "966512345678", "447911123456"


class FakeProvider:
    def __init__(self, busy=None, fail_insert=False):
        self.busy = busy or []
        self.fail_insert = fail_insert
        self.events: list[dict] = []
        self.attendees: list[tuple[str, str]] = []

    async def free_busy(self, start, end):
        return [(s, e) for s, e in self.busy if s < end and start < e]

    async def insert_event(self, **kwargs):
        if self.fail_insert:
            raise CalendarError("boom")
        self.events.append(kwargs)
        return CreatedEvent(event_id=f"evt{len(self.events)}", meet_link=f"https://meet.google.com/aaa-bbbb-cc{len(self.events)}")

    async def set_attendee(self, event_id, email):
        self.attendees.append((event_id, email))


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
async def owner(sessionmaker):
    async with sessionmaker() as s:
        s.add(Owner(id="default", name="Owner", timezone="Asia/Dubai",
                    working_hours={"days": [0, 1, 2, 3, 4], "start": "09:00", "end": "18:00"}))
        await s.commit()


@pytest.fixture
def provider():
    return FakeProvider()


@pytest.fixture
def clock():
    return Clock(NOW)


@pytest.fixture
def engine(sessionmaker, owner, provider, clock):
    return CalendarEngine(Settings(_env_file=None), sessionmaker, provider_factory=lambda o: provider, clock=clock)


@pytest.fixture
def scheduler(engine, fake_redis):
    return CalendarScheduler(engine, fake_redis)


async def _bookings(sessionmaker):
    async with sessionmaker() as s:
        return (await s.scalars(select(Booking).order_by(Booking.id))).all()


# --- spec contract ----------------------------------------------------------------------------------
def test_spec_signatures():
    sig = inspect.signature(calendar.get_available_slots)
    assert list(sig.parameters) == ["owner_id", "timezone"]
    assert sig.parameters["timezone"].default == "Asia/Dubai" and sig.return_annotation == list[str]
    sig = inspect.signature(calendar.create_calendar_event)
    assert list(sig.parameters) == ["lead_phone", "lead_name", "slot_iso"] and sig.return_annotation is bool
    assert inspect.iscoroutinefunction(calendar.get_available_slots)
    assert inspect.iscoroutinefunction(calendar.create_calendar_event)


async def test_spec_functions_delegate_to_engine(engine, provider, sessionmaker):
    calendar.configure(engine)
    try:
        assert await calendar.get_available_slots("default") == [
            "Today, Mon 16 Nov at 12:00 PM GST", "Tomorrow, Tue 17 Nov at 9:00 AM GST",
            "Wednesday 18 Nov at 9:00 AM GST",
        ]
        assert await calendar.get_available_slots("default", timezone="Asia/Riyadh") == [
            "Today, Mon 16 Nov at 11:00 AM AST", "Tomorrow, Tue 17 Nov at 8:00 AM AST",
            "Wednesday 18 Nov at 8:00 AM AST",
        ]
        assert await calendar.create_calendar_event(UAE, "Ahmed", TUE_09) is True
        [booking] = await _bookings(sessionmaker)
        assert booking.google_event_id == "evt1"  # booking reference stored
        assert await calendar.add_attendee_to_event("evt1", "a@b.com") is True
    finally:
        calendar.configure(None)


# --- engine -----------------------------------------------------------------------------------------
async def test_find_open_slots_skips_busy_and_booked(engine, provider, sessionmaker):
    provider.busy = [(datetime(2026, 11, 16, 8, 0, tzinfo=UTC), datetime(2026, 11, 16, 14, 0, tzinfo=UTC))]
    assert await engine.book(KSA, "Sara", TUE_09)  # someone already took Tue 09:00
    slots = [to_iso(s) for s in await engine.find_open_slots()]
    assert slots == ["2026-11-17T05:30:00+00:00", WED_09, "2026-11-19T05:00:00+00:00"]


async def test_book_creates_meet_event_in_owner_timezone_and_stores_reference(engine, provider, sessionmaker):
    async with sessionmaker() as s:
        s.add(Lead(phone=UAE, profile_name="Ahmed", language="ar", target_country="DUBAI", intent="INVESTMENT",
                   budget_usd=544588.16, budget_original_amount=2_000_000, budget_original_currency="AED",
                   timeline="IMMEDIATE"))
        await s.commit()
    outcome = await engine.book(UAE, "Ahmed", TUE_09)
    assert outcome.ok and outcome.event_id == "evt1" and outcome.meet_link.startswith("https://meet.google.com/")
    [event] = provider.events
    assert event["start"].isoformat() == "2026-11-17T09:00:00+04:00"
    assert event["end"].isoformat() == "2026-11-17T09:30:00+04:00"
    assert event["timezone"] == "Asia/Dubai" and event["summary"] == "Property consultation: Ahmed"
    for line in ("WhatsApp: +971501234567", "Market: UAE (Dubai)", "Purpose: Investment",
                 "Budget: ~USD 544,588 (2,000,000 AED)", "Timeline: Immediate", "Language: Arabic"):
        assert line in event["description"]
    [b] = await _bookings(sessionmaker)
    assert (b.slot_iso, b.google_event_id, b.meet_link, b.status, b.email_requested) == \
        (TUE_09, "evt1", outcome.meet_link, "CONFIRMED", True)


async def test_same_slot_cannot_be_booked_twice(engine, sessionmaker):
    assert (await engine.book(UAE, "A", TUE_09)).ok
    second = await engine.book(KSA, "B", TUE_09)
    assert not second.ok and second.taken
    assert len(await _bookings(sessionmaker)) == 1


async def test_slot_busy_in_google_is_taken_and_reservation_released(engine, provider, sessionmaker):
    provider.busy = [(datetime(2026, 11, 17, 5, 15, tzinfo=UTC), datetime(2026, 11, 17, 6, 0, tzinfo=UTC))]
    outcome = await engine.book(UAE, "A", TUE_09)
    assert not outcome.ok and outcome.taken
    assert await _bookings(sessionmaker) == [] and provider.events == []


@pytest.mark.parametrize("slot", [
    "2026-11-16T05:00:00+00:00",  # 09:00 today: already in the past
    "2026-11-16T07:00:00+00:00",  # 11:00 today: inside the 2-hour notice
    "2026-11-17T01:00:00+00:00",  # 05:00 Dubai: outside working hours
    "2026-11-21T06:00:00+00:00",  # Saturday
    "2026-11-17T05:10:00+00:00",  # not on the 30-minute grid
])
async def test_invalid_slots_are_refused(engine, provider, slot):
    outcome = await engine.book(UAE, "A", slot)
    assert not outcome.ok and outcome.taken and provider.events == []


async def test_event_insert_failure_releases_slot(engine, provider, sessionmaker):
    provider.fail_insert = True
    outcome = await engine.book(UAE, "A", TUE_09)
    assert not outcome.ok and not outcome.taken
    assert await _bookings(sessionmaker) == []


async def test_not_connected_owner_cannot_book_or_list(sessionmaker, owner, clock):
    engine = CalendarEngine(Settings(_env_file=None), sessionmaker, clock=clock)  # no refresh token
    assert not (await engine.book(UAE, "A", TUE_09)).ok
    with pytest.raises(CalendarError):
        await engine.find_open_slots()


async def test_add_attendee_saves_email(engine, provider, sessionmaker):
    await engine.book(UAE, "A", TUE_09)
    assert await engine.add_attendee("evt1", "ahmed@example.com")
    assert provider.attendees == [("evt1", "ahmed@example.com")]
    [b] = await _bookings(sessionmaker)
    assert b.lead_email == "ahmed@example.com"


# --- scheduler (agent adapter) -------------------------------------------------------------------
async def test_offer_uses_lead_timezone_from_phone(scheduler, fake_redis):
    assert await scheduler.offer_slots(KSA, "DUBAI", "en") == [
        "Today, Mon 16 Nov at 11:00 AM AST", "Tomorrow, Tue 17 Nov at 8:00 AM AST",
        "Wednesday 18 Nov at 8:00 AM AST",
    ]
    assert await scheduler.offer_slots(UAE, None, "ar") == [
        "اليوم الاثنين 16 نوفمبر، الساعة 12:00 الظهر بتوقيت الإمارات",
        "بكرة الثلاثاء 17 نوفمبر، الساعة 9:00 الصبح بتوقيت الإمارات",
        "الأربعاء 18 نوفمبر، الساعة 9:00 الصبح بتوقيت الإمارات",
    ]
    # non-GCC number: owner's timezone
    assert (await scheduler.offer_slots(UK, None, "en"))[0] == "Today, Mon 16 Nov at 12:00 PM GST"
    assert await fake_redis.get(f"offered:{KSA}") == f'["{MON_12}", "{TUE_09}", "{WED_09}"]'


async def test_offered_labels_are_rerendered_so_tomorrow_stays_true(scheduler, clock):
    await scheduler.offer_slots(UAE, None, "en")
    clock.now = NOW + timedelta(days=1)  # the lead answers the next morning
    assert await scheduler.offered_labels(UAE, "en") == [
        "Monday 16 Nov at 12:00 PM GST",  # (now in the past: booking it will be refused)
        "Today, Tue 17 Nov at 9:00 AM GST", "Tomorrow, Wed 18 Nov at 9:00 AM GST",
    ]


async def test_book_and_email_via_scheduler(scheduler, provider, fake_redis):
    await scheduler.offer_slots(KSA, None, "en")
    result = await scheduler.book(KSA, "Sara", 2, "en")
    assert result.ok and result.slot_iso == TUE_09
    assert result.slot_label == "Tomorrow, Tue 17 Nov at 8:00 AM AST"
    assert result.meet_link == "https://meet.google.com/aaa-bbbb-cc1"
    assert await fake_redis.get(f"offered:{KSA}") is None
    assert await scheduler.add_email(KSA, "sara@example.com")
    assert provider.attendees == [("evt1", "sara@example.com")]


async def test_book_past_slot_reports_taken(scheduler, clock):
    await scheduler.offer_slots(UAE, None, "en")
    clock.now = NOW + timedelta(days=1)
    result = await scheduler.book(UAE, "A", 1, "en")  # Monday 12:00 is gone
    assert not result.ok and result.slot_taken


async def test_add_email_without_booking_returns_false(scheduler):
    assert await scheduler.add_email(UAE, "x@y.com") is False


async def test_offer_returns_empty_when_calendar_down(sessionmaker, owner, fake_redis, clock):
    def broken(_):
        raise CalendarError("down")
    engine = CalendarEngine(Settings(_env_file=None), sessionmaker, provider_factory=broken, clock=clock)
    assert await CalendarScheduler(engine, fake_redis).offer_slots(UAE, None, "en") == []


# --- agent + real calendar engine -----------------------------------------------------------------
async def test_agent_books_through_calendar_engine(scheduler, provider, sessionmaker):
    script = {
        "Dubai, investment, 2M AED, ready now": Extraction(
            message_intent="PROVIDE_INFO", target_country="DUBAI", intent="INVESTMENT",
            budget_text="2M AED", timeline="IMMEDIATE"),
    }
    graph = build_graph(llm=FakeLLM(script), settings=Settings(_env_file=None), scheduler=scheduler,
                        checkpointer=InMemorySaver())
    agent = AgentService(graph, sessionmaker)

    def msg(text):
        return IncomingMessage("w", KSA, "Sara Ali", "text", text, "en", 0.0)

    offer = await agent(msg("Dubai, investment, 2M AED, ready now"), "en")
    assert offer[-1].startswith("1. Today, Mon 16 Nov at 11:00 AM AST\n2. Tomorrow, Tue 17 Nov at 8:00 AM AST")
    booked = await agent(msg("2"), "en")
    assert "📅 Tomorrow, Tue 17 Nov at 8:00 AM AST" in booked[-1]
    assert "https://meet.google.com/aaa-bbbb-cc1" in booked[-1]
    st = (await graph.aget_state({"configurable": {"thread_id": KSA}})).values
    assert st["booking_status"] == "BOOKED" and st["selected_slot"] == TUE_09
    assert provider.events[0]["summary"] == "Property consultation: Sara Ali"
    await agent(msg("sara@example.com"), "en")
    assert provider.attendees == [("evt1", "sara@example.com")]
    async with sessionmaker() as s:
        lead = await s.scalar(select(Lead).where(Lead.phone == KSA))
    assert lead.booking_status == "BOOKED"


# --- freeBusy cache and prefetch (Phase 5 latency) ----------------------------------------------------
class CountingProvider(FakeProvider):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.busy_calls = 0

    async def free_busy(self, start, end):
        self.busy_calls += 1
        return await super().free_busy(start, end)


@pytest.fixture
def counting(sessionmaker, owner, clock):
    provider = CountingProvider()
    engine = CalendarEngine(Settings(_env_file=None, calendar_busy_cache_seconds=120), sessionmaker,
                            provider_factory=lambda o: provider, clock=clock)
    return engine, provider


async def test_free_busy_is_cached_briefly(counting, clock):
    engine, provider = counting
    first = await engine.find_open_slots()
    assert await engine.find_open_slots() == first
    assert provider.busy_calls == 1
    clock.now = NOW + timedelta(seconds=121)  # cache expired
    await engine.find_open_slots()
    assert provider.busy_calls == 2
    engine.forget_busy()
    await engine.find_open_slots()
    assert provider.busy_calls == 3


async def test_cached_offer_still_excludes_new_bookings_and_booking_rechecks_live(counting):
    engine, provider = counting
    first = [to_iso(s) for s in await engine.find_open_slots()]
    assert (await engine.book(KSA, "Sara", first[0])).ok  # re-checks freeBusy live, not from the cache
    assert provider.busy_calls == 2
    again = [to_iso(s) for s in await engine.find_open_slots()]  # cache hit, but bookings are read live
    assert provider.busy_calls == 2
    assert first[0] not in again and len(again) == 3


async def test_busy_cache_can_be_turned_off(sessionmaker, owner, clock):
    provider = CountingProvider()
    engine = CalendarEngine(Settings(_env_file=None, calendar_busy_cache_seconds=0), sessionmaker,
                            provider_factory=lambda o: provider, clock=clock)
    await engine.find_open_slots()
    await engine.find_open_slots()
    assert provider.busy_calls == 2


async def test_scheduler_prefetch_fills_the_cache_in_the_background(counting, fake_redis):
    import asyncio

    engine, provider = counting
    sched = CalendarScheduler(engine, fake_redis)
    sched.prefetch()  # returns immediately
    await asyncio.gather(*sched._tasks)
    assert provider.busy_calls == 1
    labels = await sched.offer_slots(UAE, "DUBAI", "en")
    assert len(labels) == 3 and provider.busy_calls == 1  # the offer was served from the prefetch


async def test_prefetch_and_warm_never_raise(sessionmaker, owner, fake_redis, clock):
    import asyncio

    def broken(_):
        raise CalendarError("down")

    engine = CalendarEngine(Settings(_env_file=None), sessionmaker, provider_factory=broken, clock=clock)
    sched = CalendarScheduler(engine, fake_redis)
    assert await sched.warm() is False
    sched.prefetch()
    await asyncio.gather(*sched._tasks)
