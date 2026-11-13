"""Test doubles for the agent: a scripted LLM, an in-memory scheduler and knowledge base."""
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from agent.prompts import Extraction
from agent.services import BookingResult, Passage

EN_REPLY = "Sounds great, thanks for sharing!"
AR_REPLY = "تمام، مشكور على المعلومات!"


def _last_human(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


class FakeLLM:
    """``extract_map``: exact lead text -> Extraction (missing keys raise, i.e. 'LLM failed').
    ``replies``: queue of reply drafts; when empty, replies in the prompt's language."""

    def __init__(self, extract_map: dict[str, Extraction] | None = None, replies: list[str] | None = None,
                 fail_extract: bool = False, fail_generate: bool = False,
                 reply_fn: Callable[[str], str] | None = None):
        self.extract_map = extract_map or {}
        self.replies = list(replies or [])
        self.fail_extract = fail_extract
        self.fail_generate = fail_generate
        self.reply_fn = reply_fn
        self.extract_calls: list[str] = []
        self.generate_systems: list[str] = []

    async def extract(self, system, messages, schema):
        text = _last_human(messages)
        self.extract_calls.append(text)
        if self.fail_extract or text not in self.extract_map:
            raise RuntimeError("fake LLM: extraction unavailable")
        return self.extract_map[text]

    async def generate(self, system, messages):
        self.generate_systems.append(system)
        if self.fail_generate:
            raise RuntimeError("fake LLM: generation unavailable")
        if self.replies:
            return self.replies.pop(0)
        if self.reply_fn:
            return self.reply_fn(system)
        return AR_REPLY if "Khaleeji" in system else EN_REPLY


class FakeScheduler:
    SLOTS = ["Tomorrow at 4:00 PM GST", "Thursday at 11:00 AM GST", "Friday at 2:30 PM GST"]
    NEW_SLOTS = ["Sunday at 10:00 AM GST", "Monday at 1:00 PM GST", "Tuesday at 3:00 PM GST"]

    def __init__(self, taken: bool = False, fail_book: bool = False):
        self.offered: dict[str, list[str]] = {}
        self.booked: list[tuple[str, str | None, int]] = []
        self.emails: list[tuple[str, str]] = []
        self.taken = taken
        self.fail_book = fail_book
        self.offer_calls = 0

    async def offer_slots(self, phone, country, language):
        self.offer_calls += 1
        labels = self.NEW_SLOTS if self.offer_calls > 1 else self.SLOTS
        self.offered[phone] = labels
        return labels

    async def offered_labels(self, phone, language):
        return self.offered.get(phone, [])

    async def book(self, phone, lead_name, choice, language):
        if self.taken:
            self.taken = False
            return BookingResult(ok=False, slot_taken=True)
        if self.fail_book:
            return BookingResult(ok=False)
        self.booked.append((phone, lead_name, choice))
        return BookingResult(ok=True, slot_iso="2026-11-16T12:00:00+00:00",
                             slot_label=self.offered[phone][choice - 1],
                             meet_link="https://meet.google.com/abc-defg-hij")

    async def add_email(self, phone, email):
        self.emails.append((phone, email))
        return True


class FakeKnowledge:
    def __init__(self, passages: list[Passage] | None = None):
        self.passages = passages or []
        self.queries: list[tuple[str, str | None, str]] = []

    async def search(self, query, country, language):
        self.queries.append((query, country, language))
        return self.passages
