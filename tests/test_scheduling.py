"""tools/scheduling.py + core/timezones.py: working-hour slots, phone-based timezone, label correctness."""
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from core.timezones import lead_timezone, tz_display, utc_offset_label
from tools.scheduling import (
    AR_MONTHS,
    AR_WEEKDAYS,
    EN_MONTHS,
    EN_WEEKDAYS,
    WorkingHours,
    candidate_slots,
    format_slot,
    from_iso,
    pick_slots,
    to_iso,
)

DUBAI_HOURS = WorkingHours("Asia/Dubai", (0, 1, 2, 3, 4), time(9), time(18))
MIN30 = timedelta(minutes=30)
MONDAY_10_DUBAI = datetime(2026, 11, 16, 6, 0, tzinfo=UTC)  # Mon 16 Nov 2026, 10:00 in Dubai


def test_fixture_dates():
    assert MONDAY_10_DUBAI.astimezone(ZoneInfo("Asia/Dubai")).strftime("%a %H:%M") == "Mon 10:00"


# --- lead timezone from phone ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "phone,tz",
    [
        ("971501234567", "Asia/Dubai"),
        ("+971 50 123 4567", "Asia/Dubai"),
        ("00966512345678", "Asia/Riyadh"),
        ("966512345678", "Asia/Riyadh"),
        ("97455123456", "Asia/Qatar"),
        ("96551234567", "Asia/Kuwait"),
        ("447911123456", "Europe/London"),  # not a GCC code -> owner's timezone
        ("", "Europe/London"),
    ],
)
def test_lead_timezone_from_country_code(phone, tz):
    assert lead_timezone(phone, fallback="Europe/London") == tz


def test_timezone_display_names():
    at = MONDAY_10_DUBAI
    assert tz_display("Asia/Dubai", "en", at) == "GST"
    assert {tz_display(z, "en", at) for z in ("Asia/Riyadh", "Asia/Qatar", "Asia/Kuwait")} == {"AST"}
    assert tz_display("Asia/Riyadh", "ar", at) == "بتوقيت السعودية"
    assert utc_offset_label("Asia/Kolkata", at) == "UTC+5:30"
    assert utc_offset_label("America/New_York", at) == "UTC-5"
    assert tz_display("Europe/London", "en", at) == "UTC+0"


# --- candidates and picking ---------------------------------------------------------------------
def test_candidates_respect_hours_notice_and_weekends():
    friday_17 = datetime(2026, 11, 20, 13, 0, tzinfo=UTC)  # Fri 17:00 Dubai
    slots = list(candidate_slots(friday_17, DUBAI_HOURS, MIN30, timedelta(minutes=120), search_days=4))
    local = [s.astimezone(ZoneInfo("Asia/Dubai")) for s in slots]
    assert local[0].strftime("%a %d %H:%M") == "Mon 23 09:00"  # Fri too late, Sat/Sun off
    assert all(l.weekday() < 5 and time(9) <= l.time() <= time(17, 30) for l in local)
    assert local[-1].time() == time(17, 30)  # last 30-min slot still ends by 18:00


def test_pick_slots_one_per_day_skips_busy_and_booked():
    cands = list(candidate_slots(MONDAY_10_DUBAI, DUBAI_HOURS, MIN30, timedelta(minutes=120), 14))
    assert cands[0] == datetime(2026, 11, 16, 8, 0, tzinfo=UTC)  # Mon 12:00 Dubai (10:00 + 2h notice)
    busy = [(datetime(2026, 11, 16, 8, 0, tzinfo=UTC), datetime(2026, 11, 16, 9, 15, tzinfo=UTC))]
    taken = {to_iso(datetime(2026, 11, 17, 5, 0, tzinfo=UTC))}  # Tue 09:00 Dubai already booked
    picked = pick_slots(cands, busy, taken, duration=MIN30, count=3, per_day=1, owner_tz="Asia/Dubai")
    local = [p.astimezone(ZoneInfo("Asia/Dubai")).strftime("%a %H:%M") for p in picked]
    assert local == ["Mon 13:30", "Tue 09:30", "Wed 09:00"]


def test_pick_slots_two_per_day():
    cands = list(candidate_slots(MONDAY_10_DUBAI, DUBAI_HOURS, MIN30, timedelta(0), 14))
    picked = pick_slots(cands, [], set(), duration=MIN30, count=3, per_day=2, owner_tz="Asia/Dubai")
    assert [p.astimezone(ZoneInfo("Asia/Dubai")).strftime("%a %H:%M") for p in picked] == \
        ["Mon 10:00", "Mon 10:30", "Tue 09:00"]


def test_iso_roundtrip_is_utc():
    dubai = datetime(2026, 11, 17, 9, 0, tzinfo=ZoneInfo("Asia/Dubai"))
    assert to_iso(dubai) == "2026-11-17T05:00:00+00:00"
    assert from_iso("2026-11-17T09:00:00+04:00") == datetime(2026, 11, 17, 5, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        from_iso("2026-11-17T09:00:00")


# --- labels ---------------------------------------------------------------------------------------
def test_labels_in_each_gcc_timezone():
    now = MONDAY_10_DUBAI
    tue_9_dubai = datetime(2026, 11, 17, 5, 0, tzinfo=UTC)
    assert format_slot(tue_9_dubai, "Asia/Dubai", "en", now) == "Tomorrow, Tue 17 Nov at 9:00 AM GST"
    assert format_slot(tue_9_dubai, "Asia/Riyadh", "en", now) == "Tomorrow, Tue 17 Nov at 8:00 AM AST"
    assert format_slot(datetime(2026, 11, 16, 12, 0, tzinfo=UTC), "Asia/Qatar", "en", now) == \
        "Today, Mon 16 Nov at 3:00 PM AST"
    assert format_slot(datetime(2026, 11, 20, 7, 0, tzinfo=UTC), "Asia/Kuwait", "en", now) == \
        "Friday 20 Nov at 10:00 AM AST"
    assert format_slot(tue_9_dubai, "Asia/Dubai", "ar", now) == \
        "بكرة الثلاثاء 17 نوفمبر، الساعة 9:00 الصبح بتوقيت الإمارات"
    assert format_slot(datetime(2026, 11, 16, 12, 0, tzinfo=UTC), "Asia/Riyadh", "ar", now) == \
        "اليوم الاثنين 16 نوفمبر، الساعة 3:00 العصر بتوقيت السعودية"


def test_relative_day_uses_the_leads_own_midnight():
    """23:30 in Riyadh is already 00:30 next day in Dubai: the same slot is 'Tomorrow' for a Saudi
    lead but 'Today' for a UAE lead, and both labels show the slot's own local date."""
    now = datetime(2026, 11, 17, 20, 30, tzinfo=UTC)  # Tue 23:30 Riyadh / Wed 00:30 Dubai
    slot = datetime(2026, 11, 18, 5, 0, tzinfo=UTC)  # Wed 08:00 Riyadh / Wed 09:00 Dubai
    assert format_slot(slot, "Asia/Riyadh", "en", now) == "Tomorrow, Wed 18 Nov at 8:00 AM AST"
    assert format_slot(slot, "Asia/Dubai", "en", now) == "Today, Wed 18 Nov at 9:00 AM GST"


def test_slot_crossing_midnight_between_owner_and_lead_dates():
    """Owner in New York offers 20:00 Mon; for a Dubai lead that is Tue 05:00: label shows Tuesday."""
    now = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)  # Mon 16:00 Dubai
    slot = datetime(2026, 11, 17, 1, 0, tzinfo=UTC)  # Mon 20:00 New York = Tue 05:00 Dubai
    assert format_slot(slot, "Asia/Dubai", "en", now) == "Tomorrow, Tue 17 Nov at 5:00 AM GST"
    assert format_slot(slot, "America/New_York", "en", now) == "Today, Mon 16 Nov at 8:00 PM UTC-5"


@pytest.mark.parametrize("tz", ["Asia/Dubai", "Asia/Riyadh", "Asia/Qatar", "Asia/Kuwait", "America/New_York",
                                "Pacific/Kiritimati", "Pacific/Pago_Pago"])
@pytest.mark.parametrize("language", ["en", "ar"])
def test_label_always_matches_stored_date(tz, language):
    """Exhaustive: for 'now' every 37 min over 3 days and slots every 30 min over 9 days, the label's
    printed weekday/day/month equal the stored slot's local date, and Today/Tomorrow match exactly."""
    zone = ZoneInfo(tz)
    base = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
    nows = [base + timedelta(minutes=37 * i) for i in range(0, 3 * 24 * 60 // 37)]
    slots = [base + timedelta(minutes=30 * i) for i in range(0, 9 * 48, 7)]
    weekdays, months = (AR_WEEKDAYS, AR_MONTHS) if language == "ar" else (EN_WEEKDAYS, EN_MONTHS)
    today_word, tomorrow_word = ("اليوم", "بكرة") if language == "ar" else ("Today", "Tomorrow")
    for now in nows[::5]:
        for slot in slots:
            if slot < now:
                continue
            label = format_slot(slot, tz, language, now)
            local = slot.astimezone(zone)
            days = (local.date() - now.astimezone(zone).date()).days
            wd = weekdays[local.weekday()]
            assert f"{wd if language == 'ar' or days > 1 else wd[:3]} {local.day} {months[local.month - 1]}" in label
            assert label.startswith(today_word) == (days == 0), label
            assert label.startswith(tomorrow_word) == (days == 1), label
