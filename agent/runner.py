"""Agent entrypoint used by the message processor.

PHASE 1 PLACEHOLDER: replies with a short bilingual opener so the webhook pipeline can be
exercised end-to-end. Phase 2 replaces `run_agent` with the LangGraph pipeline.
"""
from services.inbound import IncomingMessage

_OPENER = {
    "en": "Hello! Thanks for reaching out 😊 Which market are you looking at: Dubai, Saudi, Qatar or Kuwait?",
    "ar": "هلا والله! مشكور على تواصلك 😊 أي سوق يهمك: دبي، السعودية، قطر ولا الكويت؟",
}


async def run_agent(msg: IncomingMessage, language: str) -> list[str]:
    return [_OPENER["ar"] if language == "ar" else _OPENER["en"]]
