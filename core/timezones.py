"""Lead timezone from the WhatsApp phone number, and timezone display names."""
import re
from datetime import datetime
from zoneinfo import ZoneInfo

# WhatsApp delivers numbers as digits with the country code and no '+', e.g. "971501234567".
PHONE_PREFIX_TZ = {
    "971": "Asia/Dubai",  # UAE
    "966": "Asia/Riyadh",  # Saudi Arabia
    "974": "Asia/Qatar",  # Qatar
    "965": "Asia/Kuwait",  # Kuwait
}

TZ_ABBR_EN = {"Asia/Dubai": "GST", "Asia/Riyadh": "AST", "Asia/Qatar": "AST", "Asia/Kuwait": "AST"}
TZ_NAME_AR = {
    "Asia/Dubai": "بتوقيت الإمارات",
    "Asia/Riyadh": "بتوقيت السعودية",
    "Asia/Qatar": "بتوقيت قطر",
    "Asia/Kuwait": "بتوقيت الكويت",
}


def lead_timezone(phone: str, fallback: str) -> str:
    """IANA timezone for a lead: from the phone country code (+971/+966/+974/+965), else ``fallback``
    (the owner's timezone)."""
    digits = re.sub(r"\D", "", phone or "")
    if digits.startswith("00"):
        digits = digits[2:]
    for prefix, tz in PHONE_PREFIX_TZ.items():
        if digits.startswith(prefix):
            return tz
    return fallback


def utc_offset_label(tz: str, at: datetime) -> str:
    """'UTC+4', 'UTC-5', 'UTC+5:30' for timezones without a GCC abbreviation."""
    offset = at.astimezone(ZoneInfo(tz)).utcoffset()
    minutes = int(offset.total_seconds() // 60) if offset else 0
    sign = "+" if minutes >= 0 else "-"
    h, m = divmod(abs(minutes), 60)
    return f"UTC{sign}{h}" + (f":{m:02d}" if m else "")


def tz_display(tz: str, language: str, at: datetime) -> str:
    if language == "ar":
        return TZ_NAME_AR.get(tz) or f"بتوقيت {utc_offset_label(tz, at)}"
    return TZ_ABBR_EN.get(tz) or utc_offset_label(tz, at)
