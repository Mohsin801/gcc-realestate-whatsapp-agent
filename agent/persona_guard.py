"""Checks every outgoing reply against the persona rules (spec §3).

* Never reveal being an AI/bot (EN + AR) - enforced only while ``persona_conceal_ai`` is on (spec default).
* No walls of text: at most ``reply_max_messages`` WhatsApp bubbles of ``reply_max_chars`` each.
* WhatsApp formatting: Markdown ``**bold**`` -> ``*bold*``, no ``#`` headings.
"""
import re

from agent.normalize import normalize_text

# Patterns are matched against normalize_text(reply): lower-case, Arabic letter variants folded.
_AI_DISCLOSURE_EN = (
    r"\bai\b", r"\ba\.i\.", r"artificial intelligence", r"\bchat ?bot", r"\bbot\b", r"\bbots\b", r"\brobot",
    r"language model", r"\bllm\b", r"\bgpt", r"chatgpt", r"openai",
    r"virtual assistant", r"digital assistant", r"\bai assistant", r"automated (assistant|system|agent|bot|reply|response|message)",
    r"\b(i am|i'm|im|as) (an? )?(\w+ )?assistant\b", r"machine learning", r"\bi was (trained|programmed)\b",
    r"\bi (am|'m) (a |an )?(computer )?program\b", r"\bnot a (real )?(human|person)\b",
)
_AI_DISCLOSURE_AR = (
    r"ذكاء اصطناعي", r"الذكاء الاصطناعي", r"\b(ال)?(شات ?)?بوت\b", r"\b(ال)?روبوت", r"مساعد (الي|افتراضي|ذكي|رقمي)", r"نموذج لغوي",
    r"(برنامج|نظام|رد) الي", r"انا برنامج", r"مجرد برنامج", r"لست (انسان|بشر)", r"مو (انسان|بشر)", r"تشات ?جي بي تي",
)
_AI_RE = re.compile("|".join(_AI_DISCLOSURE_EN + _AI_DISCLOSURE_AR))
_DIGIT_RE = re.compile(r"\d")


def reveals_ai(text: str) -> bool:
    return bool(_AI_RE.search(normalize_text(text)))


def to_whatsapp_format(text: str) -> str:
    t = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)  # Markdown bold -> WhatsApp bold
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.MULTILINE)  # drop headings
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _cut(text: str, limit: int) -> str:
    """Trim to ``limit`` chars at the last sentence end (or word) so nothing is cut mid-word."""
    if len(text) <= limit:
        return text
    window = text[:limit]
    for sep in ("\n", ". ", "! ", "? ", "؟ ", "، ", " "):
        i = window.rfind(sep)
        if i >= limit // 2:
            return window[: i + 1].strip()
    return window.strip()


def split_bubbles(text: str, max_chars: int, max_messages: int) -> list[str]:
    """Split into at most ``max_messages`` bubbles of ``max_chars``, breaking on paragraphs/sentences."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) == 1:
        parts = [p.strip() for p in re.split(r"(?<=[.!?؟])\s+", text) if p.strip()]
    bubbles: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}\n\n{part}" if current and "\n" in text else f"{current} {part}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            bubbles.append(current)
        current = part
    if current:
        bubbles.append(current)
    return [_cut(b, max_chars) for b in bubbles[:max_messages]]


def check_reply(text: str, *, conceal_ai: bool, max_chars: int, max_messages: int) -> list[str]:
    """Return the problems with a draft reply (empty list = OK). Length problems only count when the
    reply would not fit in the allowed bubbles even after splitting."""
    problems = []
    if not text or not text.strip():
        problems.append("empty")
    if conceal_ai and reveals_ai(text):
        problems.append("ai_disclosure")
    if len(text) > max_chars * max_messages:
        problems.append("too_long")
    return problems


def has_digits(text: str) -> bool:
    return bool(_DIGIT_RE.search(text))
