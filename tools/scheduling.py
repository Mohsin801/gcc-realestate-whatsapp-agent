"""Pure slot logic (no I/O): working-hour candidates, busy filtering, and slot labels.

All instants are timezone-aware. Slots are stored and exchanged as UTC ISO strings (``to_iso``);
labels are always rendered from that stored instant at display time, in the lead's timezone,
relative to "now" in the lead's timezone, and they always include the calendar date, so a relative
word like "Tomorrow" can never disagree with the stored date.
"""
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from core.timezones import tz_display

EN_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
EN_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
AR_WEEKDAYS = ("الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد")
AR_MONTHS = ("يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر",
             "نوفمبر", "ديسمبر")


def to_iso(dt: datetime) -> str:
    """Canonical storage form: UTC, second precision, e.g. '2026-11-18T05:00:00+00:00'."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime")
    return dt.astimezone(UTC).isoformat(timespec="seconds")


def from_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError(f"slot ISO without timezone: {value!r}")
    return dt.astimezone(UTC)


def parse_hhmm(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


@dataclass(frozen=True)
class WorkingHours:
    timezone: str
    days: tuple[int, ...]  # Mon=0 ... Sun=6
    start: time
    end: time


def candidate_slots(
    now: datetime, hours: WorkingHours, duration: timedelta, min_notice: timedelta, search_days: int
) -> Iterator[datetime]:
    """Every slot start (UTC) inside working hours, from now + notice, for ``search_days`` days."""
    tz = ZoneInfo(hours.timezone)
    earliest = now + min_notice
    first_day = now.astimezone(tz).date()
    for offset in range(search_days + 1):
        day = first_day + timedelta(days=offset)
        if day.weekday() not in hours.days:
            continue
        start = datetime.combine(day, hours.start, tz)
        end = datetime.combine(day, hours.end, tz)
        t = start
        while t + duration <= end:
            if t >= earliest:
                yield t.astimezone(UTC)
            t += duration


def overlaps(start: datetime, end: datetime, busy: Iterable[tuple[datetime, datetime]]) -> bool:
    return any(b_start < end and start < b_end for b_start, b_end in busy)


def pick_slots(
    candidates: Iterable[datetime],
    busy: list[tuple[datetime, datetime]],
    taken: set[str],
    *,
    duration: timedelta,
    count: int,
    per_day: int,
    owner_tz: str,
) -> list[datetime]:
    """First ``count`` free candidates, at most ``per_day`` per owner-local day."""
    tz = ZoneInfo(owner_tz)
    chosen: list[datetime] = []
    per_day_count: dict[date, int] = {}
    for slot in candidates:
        if to_iso(slot) in taken or overlaps(slot, slot + duration, busy):
            continue
        day = slot.astimezone(tz).date()
        if per_day_count.get(day, 0) >= per_day:
            continue
        chosen.append(slot)
        per_day_count[day] = per_day_count.get(day, 0) + 1
        if len(chosen) >= count:
            break
    return chosen


def _en_time(local: datetime) -> str:
    hour = local.hour % 12 or 12
    return f"{hour}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}"


def _ar_period(hour: int) -> str:
    if hour < 5:
        return "بالليل"
    if hour < 12:
        return "الصبح"
    if hour < 15:
        return "الظهر"
    if hour < 18:
        return "العصر"
    return "المسا"


def format_slot(slot: datetime, tz: str, language: str, now: datetime) -> str:
    """Label a slot for the lead, e.g. 'Tomorrow, Tue 18 Nov at 4:00 PM GST' /
    'بكرة الثلاثاء 18 نوفمبر، الساعة 4:00 العصر بتوقيت الإمارات'.

    The relative word ('Today'/'Tomorrow') is derived from the slot's own date in ``tz`` compared
    with today's date in ``tz``, and the date itself is always printed next to it.
    """
    zone = ZoneInfo(tz)
    local = slot.astimezone(zone)
    days_ahead = (local.date() - now.astimezone(zone).date()).days
    zone_name = tz_display(tz, language, slot)
    if language == "ar":
        rel = {0: "اليوم", 1: "بكرة"}.get(days_ahead)
        day_text = f"{AR_WEEKDAYS[local.weekday()]} {local.day} {AR_MONTHS[local.month - 1]}"
        when = f"{rel} {day_text}" if rel else day_text
        hour = local.hour % 12 or 12
        return f"{when}، الساعة {hour}:{local.minute:02d} {_ar_period(local.hour)} {zone_name}"
    weekday = EN_WEEKDAYS[local.weekday()]
    month = EN_MONTHS[local.month - 1]
    rel = {0: "Today", 1: "Tomorrow"}.get(days_ahead)
    when = f"{rel}, {weekday[:3]} {local.day} {month}" if rel else f"{weekday} {local.day} {month}"
    return f"{when} at {_en_time(local)} {zone_name}"
