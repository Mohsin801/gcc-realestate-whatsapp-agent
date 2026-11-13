"""agent/normalize.py: budget -> USD, Arabic numerals, EN/AR synonyms, slot choice, email."""
import pytest

from agent.normalize import (
    detect_currency,
    extract_email,
    match_country,
    match_intent,
    match_timeline,
    parse_amount,
    parse_slot_choice,
    resolve_budget,
    to_usd,
    to_western_digits,
)
from core.config import Settings


@pytest.fixture
def settings():
    return Settings(_env_file=None)


def test_arabic_indic_digits_and_separators():
    assert to_western_digits("٢٫٥ مليون و ١٬٠٠٠") == "2.5 مليون و 1,000"
    assert to_western_digits("۱۲۳") == "123"


@pytest.mark.parametrize(
    "text,amount",
    [
        ("2M", 2_000_000),
        ("2.5m AED", 2_500_000),
        ("AED 1,500,000", 1_500_000),
        ("500k", 500_000),
        ("around 3 million riyal", 3_000_000),
        ("1-2M", 2_000_000),
        ("between 800k and 1.2m", 1_200_000),
        ("مليونين درهم", 2_000_000),
        ("مليون ونص", 1_500_000),
        ("نص مليون", 500_000),
        ("٢٫٥ مليون", 2_500_000),
        ("٣ ملايين ريال", 3_000_000),
        ("800 الف", 800_000),
        ("حدود مليون", 1_000_000),
        ("3 bedrooms", None),
        ("no idea yet", None),
    ],
)
def test_parse_amount(text, amount):
    assert parse_amount(text) == amount


@pytest.mark.parametrize(
    "text,country,currency",
    [
        ("2M AED", None, "AED"),
        ("مليونين درهم", None, "AED"),
        ("3m riyal", "SAUDI", "SAR"),
        ("3m riyal", "QATAR", "QAR"),  # bare riyal from a Qatar lead = Qatari riyal
        ("5 مليون ريال قطري", None, "QAR"),
        ("100k KWD", None, "KWD"),
        ("200 الف دينار", None, "KWD"),
        ("$1.2m", None, "USD"),
        ("2 million", None, None),
    ],
)
def test_detect_currency(text, country, currency):
    assert detect_currency(text, country) == currency


def test_usd_conversion_uses_config_rates(settings):
    assert to_usd(3_672_500, "AED", settings) == 1_000_000
    assert to_usd(3_750_000, "SAR", settings) == 1_000_000
    assert to_usd(3_640_000, "QAR", settings) == 1_000_000
    assert to_usd(307_000, "KWD", settings) == 1_000_000
    assert to_usd(1_000, "XYZ", settings) is None


def test_resolve_budget_falls_back_to_country_currency(settings):
    b = resolve_budget(settings, text="around 2 million", country="DUBAI")
    assert (b.amount, b.currency) == (2_000_000, "AED")
    assert b.usd == pytest.approx(2_000_000 / 3.6725, rel=1e-6)


def test_resolve_budget_prefers_rules_over_llm_amount(settings):
    b = resolve_budget(settings, text="مليونين درهم", llm_amount=200.0, llm_currency="AED")
    assert b.amount == 2_000_000


def test_resolve_budget_uses_llm_when_phrase_unparseable(settings):
    b = resolve_budget(settings, text="two and a bit", llm_amount=2_100_000, llm_currency="SAR")
    assert (b.amount, b.currency, b.usd) == (2_100_000, "SAR", 560_000.0)


def test_resolve_budget_defaults_to_usd_without_country(settings):
    assert resolve_budget(settings, text="1.5M").currency == "USD"


@pytest.mark.parametrize(
    "text,country",
    [
        ("Looking at Dubai Marina", "DUBAI"),
        ("Abu Dhabi or Sharjah", "DUBAI"),
        ("ابي شقة في دبي", "DUBAI"),
        ("في الإمارات", "DUBAI"),
        ("villa in Riyadh", "SAUDI"),
        ("أبي بيت في جدة", "SAUDI"),
        ("Doha, the Pearl", "QATAR"),
        ("في الدوحة", "QATAR"),
        ("الكويت", "KUWAIT"),
        ("Dubai or Riyadh?", None),  # ambiguous: leave it to the LLM / ask
        ("hello", None),
    ],
)
def test_match_country(text, country):
    assert match_country(text) == country


@pytest.mark.parametrize(
    "text,intent",
    [
        ("for investment, good rental yield", "INVESTMENT"),
        ("للاستثمار", "INVESTMENT"),
        ("to live in with my family", "PERSONAL_USE"),
        ("ودي أسكن فيها", "PERSONAL_USE"),
        ("I want the golden visa", "GOLDEN_VISA"),
        ("عشان الإقامة الذهبية", "GOLDEN_VISA"),
    ],
)
def test_match_intent(text, intent):
    assert match_intent(text) == intent


@pytest.mark.parametrize(
    "text,timeline",
    [
        ("asap, ready to move", "IMMEDIATE"),
        ("الحين على طول", "IMMEDIATE"),
        ("within 3 months", "3_MONTHS"),
        ("خلال ٣ شهور", "3_MONTHS"),
        ("interested in off-plan", "OFF_PLAN"),
        ("مشروع على الخارطة", "OFF_PLAN"),
    ],
)
def test_match_timeline(text, timeline):
    assert match_timeline(text) == timeline


def test_all_patterns_are_written_in_normalized_form():
    """Patterns are matched against normalize_text(); an unfolded letter (e.g. 'على') would never match."""
    from agent import normalize as n
    from agent.persona_guard import _AI_DISCLOSURE_AR

    tables = [n.COUNTRY_PATTERNS, n.INTENT_PATTERNS, n.TIMELINE_PATTERNS]
    patterns = [p for table in tables for ps in table.values() for p in ps]
    patterns += [p for _, p in n._CURRENCY_PATTERNS] + [p for p, _ in n._WORD_AMOUNTS] + [p for _, p in n._ORDINALS]
    patterns += list(_AI_DISCLOSURE_AR)
    for p in patterns:
        assert n.normalize_text(p) == p, p


@pytest.mark.parametrize(
    "text,choice",
    [
        ("2", 2), ("٣", 3), ("option 1", 1), ("2.", 2), ("the second one", 2), ("الثاني", 2),
        ("الأول", 1), ("first", 1), ("the last one", 3), ("4", None), ("2 bedrooms please", None),
        ("tomorrow works", None),
    ],
)
def test_parse_slot_choice(text, choice):
    assert parse_slot_choice(text, 3) == choice


def test_extract_email():
    assert extract_email("sure, it's Ahmed.K@Gmail.com.") == "ahmed.k@gmail.com"
    assert extract_email("no email here") is None
