"""Test doubles for the agent: a scripted LLM, an in-memory scheduler and knowledge base, a fake embedder."""
import asyncio
import hashlib
import math
import re
from collections.abc import Callable

from langchain_core.messages import HumanMessage

from agent.normalize import normalize_text
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
    ``replies``: queue of reply drafts; when empty, replies in the prompt's language.
    ``extract_delay`` / ``generate_delay``: simulated call time in seconds; like the real provider, a call
    slower than its ``timeout`` raises ``TimeoutError`` once the timeout has passed."""

    def __init__(self, extract_map: dict[str, Extraction] | None = None, replies: list[str] | None = None,
                 fail_extract: bool = False, fail_generate: bool = False,
                 reply_fn: Callable[[str], str] | None = None,
                 extract_delay: float = 0.0, generate_delay: float = 0.0):
        self.extract_map = extract_map or {}
        self.replies = list(replies or [])
        self.fail_extract = fail_extract
        self.fail_generate = fail_generate
        self.reply_fn = reply_fn
        self.extract_delay = extract_delay
        self.generate_delay = generate_delay
        self.extract_calls: list[str] = []
        self.generate_systems: list[str] = []
        self.timeouts: list[tuple[str, float | None]] = []  # (call, timeout it was given)

    @staticmethod
    async def _take(delay: float, timeout: float | None) -> None:
        if timeout is not None and delay > timeout:
            await asyncio.sleep(timeout)
            raise TimeoutError("fake LLM: call timed out")
        if delay:
            await asyncio.sleep(delay)

    async def extract(self, system, messages, schema, timeout=None):
        text = _last_human(messages)
        self.extract_calls.append(text)
        self.timeouts.append(("extract", timeout))
        await self._take(self.extract_delay, timeout)
        if self.fail_extract or text not in self.extract_map:
            raise RuntimeError("fake LLM: extraction unavailable")
        return self.extract_map[text]

    async def generate(self, system, messages, timeout=None):
        self.generate_systems.append(system)
        self.timeouts.append(("generate", timeout))
        await self._take(self.generate_delay, timeout)
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
        self.prefetches = 0

    def prefetch(self):
        self.prefetches += 1

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


_STOP = {"the", "a", "an", "of", "in", "is", "are", "what", "for", "to", "and", "on", "it", "how", "much", "do",
         "i", "can", "my", "be", "with", "by", "or", "if", "as", "at", "from", "this", "that", "في", "من", "على",
         "هل", "كم", "ما", "عن"}


class FakeEmbedder:
    """Bag-of-words hashed into a small vector: texts sharing words are similar, unrelated ones are not."""

    def __init__(self, dim: int = 256, delay: float = 0.0):
        self.dim = dim
        self.delay = delay
        self.queries: list[str] = []
        self.documents = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"\w+", normalize_text(text)):
            if tok in _STOP or len(tok) < 2:
                continue
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed_documents(self, docs):
        self.documents += len(docs)
        return [self._vec(f"{title} {text}") for title, text in docs]

    async def embed_query(self, text):
        self.queries.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._vec(text)
