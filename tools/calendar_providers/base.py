"""Calendar provider interface. Google Calendar today; Cal.com / Outlook can implement the same protocol."""
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


class CalendarError(RuntimeError):
    """The calendar API failed (network, 5xx, unexpected response)."""


class CalendarNotConnected(CalendarError):
    """The owner has not connected a calendar, or the stored authorisation was revoked."""


@dataclass(frozen=True)
class CreatedEvent:
    event_id: str  # booking reference
    meet_link: str | None


class CalendarProvider(Protocol):
    async def free_busy(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """Busy intervals (aware datetimes) in the owner's calendar between start and end."""
        ...

    async def insert_event(
        self, *, summary: str, description: str, start: datetime, end: datetime, timezone: str, request_id: str
    ) -> CreatedEvent:
        """Create the meeting (with a video-call link) in the owner's calendar."""
        ...

    async def set_attendee(self, event_id: str, email: str) -> None:
        """Add the lead as attendee and send them the calendar invite email."""
        ...
