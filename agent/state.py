"""LangGraph session state, exactly as defined by the spec (§3). Do not add or rename fields."""
import operator
from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage

COUNTRIES = ("DUBAI", "SAUDI", "QATAR", "KUWAIT")
INTENTS = ("INVESTMENT", "PERSONAL_USE", "GOLDEN_VISA")
TIMELINES = ("IMMEDIATE", "3_MONTHS", "OFF_PLAN")
BOOKING_STATUSES = ("NONE", "SLOTS_OFFERED", "BOOKED")


class GCCRealEstateState(TypedDict):
    phone_number: str
    language: str  # 'en' (English) or 'ar' (Arabic)
    target_country: Optional[str]  # 'DUBAI', 'SAUDI', 'QATAR', 'KUWAIT'
    intent: Optional[str]  # 'INVESTMENT', 'PERSONAL_USE', 'GOLDEN_VISA'
    budget_usd: Optional[float]  # Standardized numerical budget
    timeline: Optional[str]  # 'IMMEDIATE', '3_MONTHS', 'OFF_PLAN'
    messages: Annotated[Sequence[BaseMessage], operator.add]
    is_qualified: bool  # True when all 4 parameters are collected
    booking_status: str  # 'NONE', 'SLOTS_OFFERED', 'BOOKED'
    selected_slot: Optional[str]  # ISO Timestamp for meeting


def compute_is_qualified(state: dict) -> bool:
    return all(state.get(k) is not None for k in ("target_country", "intent", "budget_usd", "timeline"))
