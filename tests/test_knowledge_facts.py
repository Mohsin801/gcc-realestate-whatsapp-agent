"""The shipped knowledge base: official sources only, dated, bilingual and consistent."""
import json
from datetime import date, timedelta

import pytest

from core.config import Settings
from tools.rag import Fact, KnowledgeBaseError, figures, is_official_url, load_facts, stale_facts

FACTS_DIR = "knowledge_base/facts"


@pytest.fixture(scope="module")
def facts() -> list[Fact]:
    return load_facts(FACTS_DIR)


def by_id(facts, fact_id):
    return next(f for f in facts if f.id == fact_id)


def test_every_country_is_covered(facts):
    countries = {f.country for f in facts}
    assert countries == {"DUBAI", "SAUDI", "QATAR", "KUWAIT"}
    for country in countries:
        assert sum(f.country == country for f in facts) >= 4


def test_spec_topics_are_covered(facts):
    # Spec §6: "Dubai DLD fees, Saudi RETT tax, Qatar Golden Visa, Kuwait rules"
    assert any(f.country == "DUBAI" and f.topic == "fees" and "Dubai Land Department" in f.en for f in facts)
    assert any(f.country == "SAUDI" and f.topic == "tax" and "RETT" in f.en for f in facts)
    assert any(f.country == "QATAR" and f.topic == "golden_visa" for f in facts)
    assert any(f.country == "KUWAIT" and f.topic == "ownership" for f in facts)


def test_every_fact_cites_official_sources_and_a_check_date(facts):
    for f in facts:
        assert f.sources, f.id
        for s in f.sources:
            assert is_official_url(s.url), (f.id, s.url)
        assert date(2026, 1, 1) <= f.checked_on <= date.today(), f.id


def test_both_languages_state_the_same_figures(facts):
    for f in facts:
        assert figures(f.en) == figures(f.ar), f.id


def test_key_figures_match_the_official_sources(facts):
    """Pin the numbers that were checked against the official pages, so an edit can't silently change them."""
    assert {"4", "2", "30", "2013"} <= figures(by_id(facts, "dubai-dld-sale-registration-fee").en)
    assert {"250", "4000", "500000", "2000"} <= figures(by_id(facts, "dubai-dld-sale-other-fees").en)
    assert "0.25" in figures(by_id(facts, "dubai-mortgage-registration-fee").en)
    assert "2000000" in figures(by_id(facts, "uae-golden-residency-property").en)
    assert "5" in figures(by_id(facts, "saudi-rett-rate").en)
    assert "1000000" in figures(by_id(facts, "saudi-rett-first-home").en)
    assert {"730000", "90"} <= figures(by_id(facts, "qatar-residency-730k").en)
    assert {"3650000", "90"} <= figures(by_id(facts, "qatar-residency-3650k").en)
    assert {"74", "1979"} <= figures(by_id(facts, "kuwait-ownership-restricted").en)


def test_fact_fits_one_whatsapp_bubble(facts):
    limit = Settings(_env_file=None).reply_max_chars
    for f in facts:
        assert len(f.en) <= limit and len(f.ar) <= limit, f.id


def test_stale_facts_are_reported(facts):
    assert stale_facts(facts, 180, today=date(2026, 11, 14)) == []
    assert len(stale_facts(facts, 180, today=date(2026, 11, 14) + timedelta(days=181))) == len(facts)


@pytest.mark.parametrize("url,ok", [
    ("https://dubailand.gov.ae/en/eservices/property-sale-registration/", True),
    ("https://dlp.dubai.gov.ae/Legislation%20Reference/2013/ECR%2030%20of%202013.html", True),
    ("https://zatca.gov.sa/en/RulesRegulations/Taxes/Pages/RETTRegulation.aspx", True),
    ("https://www.moj.gov.qa/en/faq", True),
    ("https://u.ae/en/information-and-services", True),
    ("http://dubailand.gov.ae/insecure", False),  # https only
    ("https://dubailand.gov.ae.example.com/fake", False),  # look-alike host
    ("https://notdubailand.gov.ae/", False),
    ("https://www.propertyfinder.ae/blog/dld-fees", False),  # not a government source
    ("https://timeskuwait.com/news", False),
    ("not a url", False),
])
def test_official_url_allowlist(url, ok):
    assert is_official_url(url) is ok


VALID = {
    "id": "test-fact",
    "country": "DUBAI",
    "topic": "fees",
    "title_en": "A test fact title",
    "title_ar": "عنوان معلومة للاختبار",
    "en": "The fee is 4% of the price in this test.",
    "ar": "الرسوم 4% من السعر في هذا الاختبار.",
    "sources": [{"name": "Dubai Land Department", "url": "https://dubailand.gov.ae/en/"}],
    "checked_on": "2026-11-14",
}


def test_valid_fact_parses():
    assert Fact.model_validate(VALID).id == "test-fact"


@pytest.mark.parametrize("change,message", [
    ({"ar": "الرسوم 5% من السعر في هذا الاختبار."}, "figures differ"),
    ({"ar": "The fee is 4% of the price in this test."}, "Arabic"),
    ({"checked_on": (date.today() + timedelta(days=1)).isoformat()}, "future"),
    ({"sources": [{"name": "A blog", "url": "https://example.com/dld"}]}, "official"),
    ({"sources": []}, "at least 1"),
    ({"country": "BAHRAIN"}, "country"),
    ({"id": "Bad ID"}, "id"),
    ({"source_url": "https://dubailand.gov.ae/"}, "Extra inputs"),
])
def test_invalid_fact_is_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        Fact.model_validate({**VALID, **change})


def test_load_facts_rejects_duplicates_and_bad_files(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([VALID, VALID]), encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="duplicate"):
        load_facts(tmp_path)
    (tmp_path / "a.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="a.json"):
        load_facts(tmp_path)
    with pytest.raises(KnowledgeBaseError, match="no fact files"):
        load_facts(tmp_path / "empty")
