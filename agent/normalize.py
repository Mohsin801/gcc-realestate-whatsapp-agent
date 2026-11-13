"""Deterministic (no-LLM) normalisation of lead details, in English and Gulf Arabic.

Used in two ways:
  * to clean and convert what the LLM extracted (budget -> USD, currency resolution), and
  * as a rule-based fallback extractor when the LLM is slow or unavailable.
"""
import re
from dataclasses import dataclass

from core.config import Settings

# --- Text helpers -------------------------------------------------------------------------------
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬", "01234567890123456789.,")
_TASHKEEL = re.compile(r"[ً-ْٰـ]")  # harakat, dagger alif, tatweel


def to_western_digits(text: str) -> str:
    """Arabic-Indic / Persian digits and Arabic decimal/thousands separators -> ASCII."""
    return text.translate(_DIGITS)


def normalize_text(text: str) -> str:
    """Lower-case, ASCII digits, and Arabic letter-variant folding so synonyms match reliably."""
    t = to_western_digits(text).lower()
    t = _TASHKEEL.sub("", t)
    t = re.sub("[أإآٱ]", "ا", t)
    return t.replace("ى", "ي").replace("ة", "ه")


def _first_match(text: str, table: dict[str, tuple[str, ...]]) -> str | None:
    """Return the single key whose patterns match; None if none or more than one key matches."""
    t = normalize_text(text)
    hits = {key for key, patterns in table.items() if any(re.search(p, t) for p in patterns)}
    return hits.pop() if len(hits) == 1 else None


# --- Country / intent / timeline synonyms (EN + Khaleeji/MSA) ---
# Patterns must be written in normalize_text() form: ا for أإآ, ي for ى, ه for ة, no diacritics.
COUNTRY_PATTERNS: dict[str, tuple[str, ...]] = {
    "DUBAI": (r"\bdubai\b", r"\buae\b", r"\bemirates\b", r"\babu ?dhabi\b", r"\bsharjah\b", r"\bajman\b",
              r"\bras al ?khaimah\b", r"دبي", r"الامارات", r"امارات", r"ابو ?ظبي", r"الشارقه", r"عجمان",
              r"راس الخيمه"),
    "SAUDI": (r"\bsaudi", r"\bksa\b", r"\briyadh\b", r"\bjedd?ah?\b", r"\bdammam\b", r"\bkhobar\b",
              r"\bmakkah\b", r"\bmecca\b", r"السعودي", r"سعودي", r"الرياض", r"جده", r"الدمام", r"الخبر",
              r"مكه"),
    "QATAR": (r"\bqatar", r"\bdoha\b", r"\blusail\b", r"\bthe pearl\b", r"قطر", r"الدوحه", r"لوسيل"),
    "KUWAIT": (r"\bkuwait", r"الكويت", r"كويت"),
}

INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "GOLDEN_VISA": (r"golden ?visa", r"residen(cy|ce) visa", r"\bresidency\b", r"اقامه ذهبيه",
                    r"الاقامه الذهبيه", r"الفيزا الذهبيه", r"فيزا ذهبيه", r"التاشيره الذهبيه", r"الجولدن فيزا",
                    r"جولدن فيزا"),
    "INVESTMENT": (r"invest", r"\broi\b", r"rental (yield|income)", r"\byield\b", r"\bflip", r"استثمار",
                   r"استثماري", r"عائد", r"ايجار"),
    "PERSONAL_USE": (r"live in", r"to live", r"own use", r"personal use", r"for my family", r"end ?user",
                     r"\bhome\b", r"سكن", r"اسكن", r"للسكن", r"لعايلتي", r"لعائلتي", r"استخدام شخصي"),
}

TIMELINE_PATTERNS: dict[str, tuple[str, ...]] = {
    "OFF_PLAN": (r"off[- ]?plan", r"under construction", r"new launch", r"علي الخارطه", r"قيد الانشاء",
                 r"اوف بلان"),
    "IMMEDIATE": (r"\basap\b", r"\bimmediate", r"right (now|away)", r"\bnow\b", r"this month", r"ready to move",
                  r"ready property", r"فوري", r"الحين", r"علي طول", r"جاهز", r"هالشهر", r"هذا الشهر"),
    "3_MONTHS": (r"\b[1-6] ?months?\b", r"few months", r"couple of months", r"next quarter", r"this year",
                 r"later this year", r"خلال [1-6] ?(شهور|اشهر|شهر)", r"[1-6] ?(شهور|اشهر)", r"كم شهر",
                 r"بعد شهرين", r"شهرين", r"ثلاث شهور", r"ثلاثه اشهر"),
}

COUNTRY_CURRENCY = {"DUBAI": "AED", "SAUDI": "SAR", "QATAR": "QAR", "KUWAIT": "KWD"}
COUNTRY_TIMEZONE = {"DUBAI": "Asia/Dubai", "SAUDI": "Asia/Riyadh", "QATAR": "Asia/Qatar", "KUWAIT": "Asia/Kuwait"}


def match_country(text: str) -> str | None:
    return _first_match(text, COUNTRY_PATTERNS)


def match_intent(text: str) -> str | None:
    return _first_match(text, INTENT_PATTERNS)


def match_timeline(text: str) -> str | None:
    return _first_match(text, TIMELINE_PATTERNS)


# --- Budget ------------------------------------------------------------------------------------
_CURRENCY_PATTERNS: tuple[tuple[str, str], ...] = (  # order matters: specific before generic
    ("QAR", r"\bqar\b|\bqr\b|qatari riyal|ريال قطري"),
    ("KWD", r"\bkwd\b|\bkd\b|dinar|دينار|دنانير"),
    ("AED", r"\baed\b|dirham|\bdhs?\b|درهم|دراهم"),
    ("USD", r"\busd\b|\$|dollar|دولار"),
    ("SAR", r"\bsar\b|\bsr\b|riyal|ريال|ريالات"),
)
_GENERIC_RIYAL = re.compile(r"riyal|ريال")

_MULTIPLIERS = {
    "k": 1e3, "thousand": 1e3, "الف": 1e3, "الاف": 1e3,
    "m": 1e6, "mn": 1e6, "mil": 1e6, "million": 1e6, "millions": 1e6, "مليون": 1e6, "ملايين": 1e6,
    "bn": 1e9, "billion": 1e9, "مليار": 1e9,
}
_NUMBER = re.compile(
    r"(\d+(?:[.,]\d+)*)\s*(k|mn|mil|m|millions|million|thousand|bn|billion|الاف|الف|ملايين|مليون|مليار)?(?![a-z])"
)
_WORD_AMOUNTS: tuple[tuple[str, float], ...] = (  # amounts written only in words
    (r"مليونين و ?(نص|نصف)|two and a half million", 2.5e6),
    (r"مليون و ?(نص|نصف)|(one and a half|a million and a half) million|million and a half", 1.5e6),
    (r"(نص|نصف) مليون|half a million", 0.5e6),
    (r"ربع مليون|quarter (of a )?million", 0.25e6),
    (r"مليونين|two million|2 million", 2e6),
    (r"(ثلاث|ثلاثه) ملايين|three million", 3e6),
    (r"(اربع|اربعه) ملايين|four million", 4e6),
    (r"(خمس|خمسه) ملايين|five million", 5e6),
    (r"\bمليون\b|^مليون|a million|one million", 1e6),
)


def _to_float(token: str) -> float:
    # "1,500,000" -> thousands separators; "2,5" -> decimal comma; "1.5" -> decimal point
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", token):
        return float(token.replace(",", ""))
    return float(token.replace(",", "."))


def detect_currency(text: str, country: str | None = None) -> str | None:
    t = normalize_text(text)
    for code, pattern in _CURRENCY_PATTERNS:
        if re.search(pattern, t):
            if code == "SAR" and _GENERIC_RIYAL.search(t) and country == "QATAR":
                return "QAR"  # a bare "riyal" from a Qatar lead means Qatari riyal
            return code
    return None


def parse_amount(text: str) -> float | None:
    """Parse the largest money amount in a phrase: '2M', '1.5 million', '٢٫٥ مليون', 'مليونين', '1-2M'."""
    t = normalize_text(text)
    values: list[tuple[float, float | None]] = []
    for num, unit in _NUMBER.findall(t):
        try:
            values.append((_to_float(num), _MULTIPLIERS.get(unit) if unit else None))
        except ValueError:
            continue
    amounts: list[float] = []
    last_mult = next((m for _, m in reversed(values) if m), None)
    for value, mult in values:
        if mult is None and last_mult and value < 1000:
            mult = last_mult  # "1 to 1.5 million" -> both in millions
        amounts.append(value * (mult or 1))
    plausible = [a for a in amounts if a >= 1000]  # ignore "3 bedrooms", "2 kids"
    if plausible:
        return max(plausible)
    for pattern, value in _WORD_AMOUNTS:
        if re.search(pattern, t):
            return value
    return None


def to_usd(amount: float, currency: str, settings: Settings) -> float | None:
    rates = {"USD": 1.0, "AED": settings.rate_aed, "SAR": settings.rate_sar,
             "QAR": settings.rate_qar, "KWD": settings.rate_kwd}
    rate = rates.get(currency.upper())
    return round(amount / rate, 2) if rate else None


@dataclass(frozen=True)
class Budget:
    amount: float
    currency: str
    usd: float


def resolve_budget(
    settings: Settings,
    *,
    text: str | None,
    llm_amount: float | None = None,
    llm_currency: str | None = None,
    country: str | None = None,
) -> Budget | None:
    """Combine the rule-based parse of the budget phrase with the LLM's reading.

    Rules win when they can parse the phrase (deterministic for 'مليونين', '٢٫٥ مليون', '2M').
    Currency: explicit mention > LLM > the target country's currency > USD.
    """
    amount = parse_amount(text) if text else None
    if amount is None and llm_amount and llm_amount > 0:
        amount = float(llm_amount)
    if amount is None:
        return None
    currency = detect_currency(text, country) if text else None
    if currency is None and llm_currency and llm_currency.upper() in {"USD", "AED", "SAR", "QAR", "KWD"}:
        currency = llm_currency.upper()
    if currency is None:
        currency = COUNTRY_CURRENCY.get(country or "", "USD")
    usd = to_usd(amount, currency, settings)
    return Budget(amount=amount, currency=currency, usd=usd) if usd is not None else None


_BUDGET_HINT = re.compile(
    r"budget|price|spend|afford|range|\d\s*(k|m|mn|million)\b|aed|sar|qar|kwd|usd|\$|dirham|riyal|dinar|"
    r"ميزانيه|ميزانيتي|سعر|حدود|مليون|ملايين|الف|درهم|ريال|دينار|دولار"
)


def looks_like_budget(text: str) -> bool:
    return bool(_BUDGET_HINT.search(normalize_text(text)))


# --- Slot choice / email ------------------------------------------------------------------------
_ORDINALS = (
    (1, r"\b(first|1st)\b|^(option |slot |number )?one$|الاول|اول واحد|^اول$|^واحد$"),
    (2, r"\b(second|2nd)\b|^(option |slot |number )?two$|الثاني|ثاني واحد|^ثاني$|^اثنين$|^ثنين$"),
    (3, r"\b(third|3rd)\b|^(option |slot |number )?three$|الثالث|ثالث واحد|^ثالث$|^ثلاثه$"),
)
_LONE_DIGIT = re.compile(r"^\s*(?:option|slot|no\.?|number|#|رقم|الخيار|خيار)?\s*([1-9])\s*[.)!]*\s*$")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def parse_slot_choice(text: str, n_slots: int) -> int | None:
    """1-based slot index from replies like '2', '٢', 'option 2', 'the second one', 'الثاني'."""
    t = normalize_text(text).strip()
    m = _LONE_DIGIT.match(t)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= n_slots else None
    if len(t) > 40:  # long free text: leave it to the LLM
        return None
    hits = [n for n, pattern in _ORDINALS if re.search(pattern, t)]
    if len(hits) == 1 and hits[0] <= n_slots:
        return hits[0]
    if re.search(r"\b(last|final) (one|slot|option)\b|الاخير", t) and n_slots:
        return n_slots
    return None


def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    return m.group(0).strip(".").lower() if m else None
