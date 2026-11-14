"""Interfaces the agent graph uses for scheduling (Phase 3) and knowledge retrieval (Phase 4).

The graph depends only on these protocols. Until the real implementations are wired in, the
``Null*`` versions make the agent behave safely:
  * no live calendar -> after qualification the lead is told the consultant will reach out to set a time;
  * no knowledge base -> legal/tax questions are never answered from the LLM's memory; the agent
    defers the exact figures to the consultant call (spec DoD: "without hallucinating").
"""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class BookingResult:
    ok: bool
    slot_iso: str | None = None
    slot_label: str | None = None  # in the lead's language and timezone
    meet_link: str | None = None
    slot_taken: bool = False  # True if the slot was booked by someone else in the meantime


@dataclass(frozen=True)
class Passage:
    text: str
    source: str = ""
    country: str | None = None
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class Scheduler(Protocol):
    async def offer_slots(self, phone: str, country: str | None, language: str) -> list[str]:
        """Fetch open slots, remember them for this lead, and return their labels (in order)."""
        ...

    async def offered_labels(self, phone: str, language: str) -> list[str]:
        """Labels of the slots currently on offer to this lead (empty if none / expired)."""
        ...

    async def book(self, phone: str, lead_name: str | None, choice: int, language: str) -> BookingResult:
        """Book the 1-based ``choice`` among the offered slots."""
        ...

    async def add_email(self, phone: str, email: str) -> bool:
        """Attach the lead's email to their booking so the calendar invite is sent."""
        ...

    def prefetch(self) -> None:
        """Start loading calendar availability in the background (the lead is about to be offered slots).
        Must return immediately and never raise."""
        ...


class Knowledge(Protocol):
    async def search(self, query: str, country: str | None, language: str) -> list[Passage]:
        """Return passages above the relevance threshold; empty when nothing grounded is found."""
        ...


class NullScheduler:
    async def offer_slots(self, phone: str, country: str | None, language: str) -> list[str]:
        return []

    async def offered_labels(self, phone: str, language: str) -> list[str]:
        return []

    async def book(self, phone: str, lead_name: str | None, choice: int, language: str) -> BookingResult:
        return BookingResult(ok=False)

    async def add_email(self, phone: str, email: str) -> bool:
        return False

    def prefetch(self) -> None:
        return None


class NullKnowledge:
    async def search(self, query: str, country: str | None, language: str) -> list[Passage]:
        return []
