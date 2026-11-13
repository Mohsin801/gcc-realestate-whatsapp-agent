import pytest

from agent.language import detect_language


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hi, I want an apartment in Dubai", "en"),
        ("السلام عليكم", "ar"),
        ("مطلوب شقة في دبي", "ar"),  # spec §3 example
        ("ابي شقة في Dubai Marina", "ar"),  # mixed, Arabic-dominant
        ("Budget is 2M AED, شكرا", "en"),  # mixed, English-dominant
        ("ميزانيتي ٢ مليون", "ar"),  # Arabic-Indic digits
        ("2", None),  # slot pick: no letters -> keep previous language
        ("👍🙏", None),
        ("", None),
        (None, None),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected
