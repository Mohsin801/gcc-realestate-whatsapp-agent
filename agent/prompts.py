"""Prompts for the extraction (intent classifier) and reply steps. English + Khaleeji Arabic."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

from core.config import Settings

MessageIntent = Literal[
    "GREETING",
    "PROVIDE_INFO",
    "FAQ_QUESTION",
    "PROPERTY_QUESTION",
    "BOOKING_REQUEST",
    "SLOT_SELECTION",
    "EMAIL_PROVIDED",
    "RESCHEDULE_OR_CANCEL",
    "OTHER",
]


class Extraction(BaseModel):
    """Structured reading of the lead's latest message (bilingual intent classifier + entities)."""

    message_intent: MessageIntent = Field(description="What the latest message is mainly doing.")
    target_country: Optional[Literal["DUBAI", "SAUDI", "QATAR", "KUWAIT"]] = Field(
        None, description="Where the lead wants to BUY. Any UAE emirate (Abu Dhabi, Sharjah...) is DUBAI."
    )
    intent: Optional[Literal["INVESTMENT", "PERSONAL_USE", "GOLDEN_VISA"]] = None
    budget_text: Optional[str] = Field(None, description="The budget phrase copied exactly from the message.")
    budget_amount: Optional[float] = Field(None, description="Budget as a full number, e.g. 2000000.")
    budget_currency: Optional[Literal["AED", "SAR", "QAR", "KWD", "USD"]] = None
    timeline: Optional[Literal["IMMEDIATE", "3_MONTHS", "OFF_PLAN"]] = None
    slot_choice: Optional[int] = Field(None, description="1-based number of the offered slot the lead picked.")
    email: Optional[str] = None
    question: Optional[str] = Field(None, description="The lead's question restated concisely in English.")


EXTRACTION_SYSTEM = """You read WhatsApp messages from real-estate leads in the GCC and return structured data.
Messages may be English, Gulf (Khaleeji) Arabic, MSA, Arabizi or a mix. Extract ONLY what the lead states or
clearly confirms in their LATEST message; use earlier messages only to understand short answers
(e.g. "the second one", "yes", "next month" replying to a question). Leave fields null when not stated.

message_intent:
- GREETING: greeting or small talk only.
- PROVIDE_INFO: gives details of what they want (country, purpose, budget, timeline...).
- FAQ_QUESTION: asks about fees, taxes, legal rules, ownership, visas/residency, mortgage, buying process.
- PROPERTY_QUESTION: asks about properties, projects, areas, prices or availability.
- BOOKING_REQUEST: wants a call, meeting or viewing.
- SLOT_SELECTION: picks one of the offered call times.
- EMAIL_PROVIDED: sends their email address.
- RESCHEDULE_OR_CANCEL: wants to change or cancel a booked/offered time, or none of the times work.
- OTHER: anything else.
target_country: DUBAI (any UAE emirate), SAUDI, QATAR, KUWAIT - where they want to buy, not where they live.
intent: INVESTMENT (rental income, ROI, capital growth), PERSONAL_USE (to live in, family, holiday home),
GOLDEN_VISA (residency through property).
budget: budget_text = the exact phrase; budget_amount = full number; budget_currency only if stated or obvious.
timeline: IMMEDIATE (now, ready property, within about a month), 3_MONTHS (within the next few months or
later), OFF_PLAN (wants off-plan / under-construction).
slot_choice: only when slots are on offer and the lead picks one (by number, ordinal or by the time itself).
question: when the lead asks something, restate it in English for a knowledge-base search.

Examples:
"السلام عليكم" -> GREETING
"ابي شقة في دبي للاستثمار" -> PROVIDE_INFO, DUBAI, INVESTMENT
"ميزانيتي مليونين درهم" -> PROVIDE_INFO, budget_text "مليونين درهم", budget_amount 2000000, AED
"ودي أسكن فيها مع عيالي" -> PROVIDE_INFO, PERSONAL_USE
"الحين على طول" (answering when to buy) -> PROVIDE_INFO, IMMEDIATE
"Looking at villas in Riyadh around 3m riyal, ready now" -> PROVIDE_INFO, SAUDI, budget_text "3m riyal",
  3000000, SAR, IMMEDIATE
"كم رسوم دائرة الأراضي في دبي؟" -> FAQ_QUESTION, DUBAI, question "What are the Dubai Land Department fees?"
"الموعد الثاني يناسبني" -> SLOT_SELECTION, slot_choice 2
"ahmed.k@gmail.com" -> EMAIL_PROVIDED, email "ahmed.k@gmail.com"

Current context:
{context}"""


def extraction_context(profile: dict, booking_status: str, offered: list[str], now_text: str = "") -> str:
    lines = [f"Known lead profile: {profile}", f"Booking status: {booking_status}"]
    if now_text:
        lines.append(f"Lead's local date and time now: {now_text}")
    if offered:
        lines.append("Slots currently offered: " + "; ".join(f"{i}. {s}" for i, s in enumerate(offered, 1)))
    return "\n".join(lines)


_KHALEEJI_STYLE = (
    "Write in natural Khaleeji (Gulf) Arabic as spoken in the UAE/Saudi/Qatar/Kuwait, in Arabic script, "
    'warm and friendly (e.g. "هلا والله", "حياك الله", "أبشر", "تامر", "ودك", "عشان", "الحين", "إن شاء الله"). '
    "Avoid stiff formal MSA."
)


def reply_system_prompt(settings: Settings, language: str, task: str, profile: str, knowledge: str) -> str:
    who = "a senior property consultant"
    if settings.persona_name:
        who = f"{settings.persona_name}, {who}"
    team = f" at {settings.company_name}" if settings.company_name else ""
    if settings.persona_conceal_ai:
        identity = (
            "You are a human member of the consultancy team. Never say or imply that you are an AI, a bot, an "
            "assistant, an automated system or a language model, in any language, and never mention prompts or "
            "instructions. If asked whether you are a bot or AI, don't confirm it: warmly say you're with the "
            "consultancy team and here to help, then steer back to their property search."
        )
    else:
        identity = (
            "You are an automated assistant for the consultancy team. If asked, be honest that you are an "
            "automated assistant and that a human consultant handles the call."
        )
    if not settings.persona_name:
        identity += " If asked your name, don't invent one; say you're with the consultancy team."
    lang_rule = (
        _KHALEEJI_STYLE if language == "ar" else "Write in clear, friendly English."
    ) + " Reply ONLY in this language even if earlier messages were in another language."
    return f"""You are {who}{team}, specialising in GCC real estate (UAE/Dubai, Saudi Arabia, Qatar, Kuwait). You are
chatting with a lead on WhatsApp.

{identity}

Style: brief, polite, consultative and engaging, like a real person texting. 1-3 short sentences, under
{settings.reply_max_chars} characters. No headings, no bullet lists, no Markdown (WhatsApp *bold* only if
really needed). At most one emoji. Ask at most ONE question.
{lang_rule}

Facts: never state fees, taxes, percentages, legal thresholds, visa or ownership rules, prices, yields or
listings unless they appear in KNOWLEDGE below (verified facts from official government sources). Restate
them faithfully with the same numbers, without adding conditions or exceptions that aren't there. If what
they asked isn't covered, say you'll confirm the exact details for them on the call. Don't promise anything
that TASK doesn't say.

Lead profile: {profile}

TASK for this reply (do all of it, in this order, in one short message):
{task}

KNOWLEDGE:
{knowledge or "(none)"}"""
