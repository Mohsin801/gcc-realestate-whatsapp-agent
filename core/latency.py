"""Latency budget and latency statistics (spec DoD: reply in < 3 s, English and Arabic).

``Deadline`` starts when the webhook receives the message (``IncomingMessage.received_at``) and travels with
the turn through the LangGraph config. Each external call asks it how long it may take; when too little of the
budget is left, the call is skipped and the agent uses its rule/template fallback, so the reply still goes out
in time. Writes (calendar bookings, attendee emails) are never cut short by the budget.
"""
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass


class Deadline:
    def __init__(self, seconds: float, *, start: float | None = None, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self.start = clock() if start is None else start
        self.at = self.start + seconds

    def remaining(self) -> float:
        return self.at - self._clock()

    def elapsed_ms(self) -> int:
        return int((self._clock() - self.start) * 1000)

    def allow(self, timeout: float, *, keep: float = 0.0, minimum: float = 0.0) -> float | None:
        """Seconds a call may take: at most ``timeout``, leaving ``keep`` seconds of the budget for later
        steps. ``None`` when less than ``minimum`` would be left: skip the call and use the fallback."""
        allowed = min(timeout, self.remaining() - keep)
        return allowed if allowed >= max(minimum, 0.001) else None


def percentile(values: Iterable[float], pct: float) -> float | None:
    """Nearest-rank percentile (``pct`` in 0-100); ``None`` for no values."""
    data = sorted(values)
    if not data:
        return None
    rank = max(1, math.ceil(pct / 100 * len(data)))
    return data[min(rank, len(data)) - 1]


@dataclass(frozen=True)
class LatencyStats:
    count: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None
    within_budget: float | None  # share of replies at or under the budget (0-1)

    @classmethod
    def of(cls, values_ms: Iterable[float], budget_ms: float) -> "LatencyStats":
        data = list(values_ms)
        if not data:
            return cls(0, None, None, None, None, None)
        return cls(
            count=len(data),
            p50_ms=percentile(data, 50),
            p95_ms=percentile(data, 95),
            p99_ms=percentile(data, 99),
            max_ms=max(data),
            within_budget=sum(v <= budget_ms for v in data) / len(data),
        )

    def line(self) -> str:
        if not self.count:
            return "no data"
        return (f"n={self.count}  p50={self.p50_ms:.0f} ms  p95={self.p95_ms:.0f} ms  p99={self.p99_ms:.0f} ms  "
                f"max={self.max_ms:.0f} ms  within budget={self.within_budget:.0%}")
