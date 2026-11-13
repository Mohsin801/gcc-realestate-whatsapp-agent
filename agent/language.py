"""Fast, dependency-free English/Arabic detection (runs inside the webhook, so it must be ~µs)."""
import re
import unicodedata

# Arabic, Arabic Supplement, Arabic Extended-A, Presentation Forms A/B
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
ARABIC_RATIO_THRESHOLD = 0.3


def _is_arabic(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_RANGES)


_NON_LANGUAGE = re.compile(r"\S+@\S+\.\S+|https?://\S+|www\.\S+")  # emails/links say nothing about language


def detect_language(text: str | None) -> str | None:
    """Return 'ar' or 'en', or None when the text has no letters (emoji, digits, punctuation).

    None lets the caller keep the lead's previous language — e.g. a lone "2" picking a slot, or a
    bare email address.
    Arabic wins if Arabic letters are more than 30% of all letters, so mixed messages like
    "ابي apartment في Dubai" are answered in Arabic.
    """
    if not text:
        return None
    text = _NON_LANGUAGE.sub(" ", text)
    arabic = latin = 0
    for ch in text:
        if _is_arabic(ch):
            if unicodedata.category(ch).startswith("L"):  # letters only, not Arabic digits/punctuation
                arabic += 1
        elif ch.isalpha():
            latin += 1
    total = arabic + latin
    if total == 0:
        return None
    return "ar" if arabic / total > ARABIC_RATIO_THRESHOLD else "en"
