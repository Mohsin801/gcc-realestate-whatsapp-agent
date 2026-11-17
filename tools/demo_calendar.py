"""A fake calendar for trying the booking flow without a Google account (``CALENDAR_DEMO=true``).

It uses the real slot finder and label formatter over an empty calendar inside the configured working hours,
so leads see the same slot labels as with Google. Nothing is written to Google or to the ``bookings`` table:
offers and bookings live in process memory and are lost on restart. The Meet link is a placeholder.
"""
from datetime import UTC, datetime, timedelta

from agent.services import BookingResult
from core.config import Settings
from core.timezones import lead_timezone
from tools.scheduling import WorkingHours, candidate_slots, format_slot, parse_hhmm, pick_slots, to_iso

DEMO_MEET_LINK = "https://meet.google.com/demo-demo-demo"


class DemoScheduler:
    """An empty calendar inside the configured working hours: the real slot finder and label formatter, with
    no Google account. Offers are kept per lead (phone), like ``CalendarScheduler``."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.offered: dict[str, list[datetime]] = {}
        self.booked: dict[str, datetime] = {}

    def _labels(self, phone: str, slots: list[datetime], language: str) -> list[str]:
        tz = lead_timezone(phone, self.settings.owner_timezone)
        now = datetime.now(UTC)
        return [format_slot(s, tz, language, now) for s in slots]

    async def offer_slots(self, phone, country, language):
        s = self.settings
        hours = WorkingHours(s.owner_timezone, tuple(s.working_days), parse_hhmm(s.work_start), parse_hhmm(s.work_end))
        duration = timedelta(minutes=s.slot_duration_minutes)
        cands = candidate_slots(datetime.now(UTC), hours, duration, timedelta(minutes=s.min_notice_minutes),
                                s.slot_search_days)
        taken = {to_iso(t) for t in self.booked.values()}
        self.offered[phone] = pick_slots(cands, [], taken, duration=duration, count=s.slot_count,
                                         per_day=s.slots_per_day, owner_tz=s.owner_timezone)
        return self._labels(phone, self.offered[phone], language)

    async def offered_labels(self, phone, language):
        return self._labels(phone, self.offered.get(phone, []), language)

    async def book(self, phone, lead_name, choice, language):
        slots = self.offered.get(phone, [])
        if not 1 <= choice <= len(slots):
            return BookingResult(ok=False, slot_taken=True)
        slot = slots[choice - 1]
        self.booked[phone] = slot
        self.offered.pop(phone, None)
        [label] = self._labels(phone, [slot], language)
        return BookingResult(ok=True, slot_iso=to_iso(slot), slot_label=label, meet_link=DEMO_MEET_LINK)

    async def add_email(self, phone, email):
        return phone in self.booked

    def prefetch(self) -> None:
        return None

    def forget(self, phone: str) -> None:
        """Drop the lead's offers and booking (dev reset, ``services/reset.py``)."""
        self.offered.pop(phone, None)
        self.booked.pop(phone, None)
