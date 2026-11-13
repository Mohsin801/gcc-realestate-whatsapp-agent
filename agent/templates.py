"""Fixed, persona-safe bilingual messages used outside the LLM path.

Persona rules (spec §3): never say AI/bot/assistant; brief, polite, consultative.
"""

UNSUPPORTED_MEDIA = {
    "en": "Thanks for that! Could you type your message here for me? That way I can help you faster 🙏",
    "ar": "مشكور! ممكن تكتب لي رسالتك هنا؟ عشان أقدر أخدمك أسرع 🙏",
}

TEMPORARY_ERROR = {
    "en": "Sorry, give me just a moment, I'll get right back to you.",
    "ar": "المعذرة، عطني لحظة وأرجع لك على طول.",
}


def pick(template: dict[str, str], language: str | None) -> str:
    return template["ar"] if language == "ar" else template["en"]
