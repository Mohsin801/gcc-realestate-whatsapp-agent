"""agent/persona_guard.py: AI-disclosure detection (EN/AR), bubble splitting, WhatsApp formatting."""
import pytest

from agent.persona_guard import check_reply, reveals_ai, split_bubbles, to_whatsapp_format


@pytest.mark.parametrize(
    "text",
    [
        "As an AI assistant, I can help you with that.",
        "I'm an automated bot.",
        "I am a virtual assistant for the agency",
        "I'm just a language model",
        "This is an automated reply",
        "أنا مساعد آلي",
        "أنا بوت",
        "أنا مساعد افتراضي هنا لخدمتك",
        "تم تطويري بالذكاء الاصطناعي",
        "أنا مجرد برنامج",
    ],
)
def test_detects_ai_disclosure(text):
    assert reveals_ai(text)


@pytest.mark.parametrize(
    "text",
    [
        "Great choice! Dubai Marina has excellent rental demand.",
        "Our consultant will confirm the exact figures on the call.",
        "My assistant will send the brochure.",  # a human can have an assistant
        "هلا والله! أي سوق يهمك؟",
        "ثبوت الملكية يتم عن طريق سند الملكية",  # 'بوت' inside a word is not 'bot'
        "Said, the villa is ready.",
    ],
)
def test_clean_replies_pass(text):
    assert not reveals_ai(text)


def test_check_reply_respects_conceal_flag():
    text = "As an AI assistant I can help."
    assert "ai_disclosure" in check_reply(text, conceal_ai=True, max_chars=400, max_messages=2)
    assert check_reply(text, conceal_ai=False, max_chars=400, max_messages=2) == []


def test_check_reply_flags_walls_of_text_and_empty():
    assert "too_long" in check_reply("x" * 801, conceal_ai=True, max_chars=400, max_messages=2)
    assert "empty" in check_reply("  ", conceal_ai=True, max_chars=400, max_messages=2)


def test_split_bubbles_respects_limits():
    sentence = "This is a fairly normal sentence about property. "
    bubbles = split_bubbles(sentence * 15, max_chars=400, max_messages=2)
    assert 1 <= len(bubbles) <= 2
    assert all(len(b) <= 400 for b in bubbles)
    assert all(b.endswith(".") for b in bubbles)  # never cut mid-sentence


def test_split_bubbles_short_text_is_one_bubble():
    assert split_bubbles("Hi there!", 400, 2) == ["Hi there!"]


def test_split_bubbles_on_paragraphs():
    text = ("A" * 300) + "\n\n" + ("B" * 300)
    assert split_bubbles(text, 400, 2) == ["A" * 300, "B" * 300]


def test_whatsapp_formatting():
    assert to_whatsapp_format("## Title\n**Dubai** is great\n\n\n\nOk") == "Title\n*Dubai* is great\n\nOk"
