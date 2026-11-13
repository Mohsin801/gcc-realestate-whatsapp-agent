"""Process-wide runtime dependencies used by background tasks.

The webhook handler signature is fixed by the spec (§5) and has no `Request`, so the shared
WhatsApp client, DB session factory and agent runner are reached through this container,
which `main.py` fills during startup (and tests fill with fakes).
"""
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from services.inbound import IncomingMessage


class MessageSender(Protocol):
    async def send_text(self, to: str, body: str) -> str | None: ...
    async def mark_read(self, wamid: str, typing: bool = True) -> None: ...


# Takes the inbound message plus the resolved language ('en'/'ar'); returns 1+ reply bubbles.
AgentRunner = Callable[[IncomingMessage, str], Awaitable[list[str]]]


@dataclass
class Runtime:
    sender: MessageSender
    sessionmaker: async_sessionmaker[AsyncSession]
    run_agent: AgentRunner
    agent_timeout_seconds: float = 20.0


_runtime: Runtime | None = None


def set_runtime(runtime: Runtime | None) -> None:
    global _runtime
    _runtime = runtime


def get_runtime() -> Runtime:
    if _runtime is None:
        raise RuntimeError("Runtime not initialised (app lifespan has not run)")
    return _runtime
