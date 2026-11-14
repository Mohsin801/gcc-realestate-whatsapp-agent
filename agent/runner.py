"""Agent entrypoint used by the message processor: runs one turn of the LangGraph agent for a lead.

* Session state lives in the LangGraph checkpointer (Redis in production), keyed by phone number.
* If there is no checkpoint (new lead, or the session expired), the qualification fields are seeded
  from the Postgres ``leads`` row, so a returning lead is never asked the same questions twice.
* After each turn the qualification fields are written back to ``leads`` (the durable record).
* Latency: with ``budget_seconds`` set, a ``Deadline`` that started at webhook receipt (``msg.received_at``)
  goes into the graph config, so nodes cut slow LLM/search calls short and fall back in time. The checkpoint
  is written once, when the turn ends (``durability="exit"``), not after every node.
"""
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.latency import Deadline
from db import repository
from services.inbound import IncomingMessage

log = logging.getLogger(__name__)


class AgentService:
    def __init__(self, graph, sessionmaker: async_sessionmaker[AsyncSession] | None = None,
                 budget_seconds: float | None = None):
        self.graph = graph
        self.sessionmaker = sessionmaker
        self.budget_seconds = budget_seconds

    async def __call__(self, msg: IncomingMessage, language: str) -> list[str]:
        deadline = Deadline(self.budget_seconds, start=msg.received_at) if self.budget_seconds else None
        # The Deadline object is not a str/int/float, so LangGraph keeps it out of checkpoint metadata.
        config = {"configurable": {"thread_id": msg.phone, "lead_name": msg.profile_name, "deadline": deadline}}
        turn_input: dict[str, Any] = {
            "messages": [HumanMessage(msg.text or "")],
            "phone_number": msg.phone,
            "language": language,
            "turn": {},
        }
        snapshot = await self.graph.aget_state(config)
        if not snapshot.values:
            turn_input |= await self._seed(msg.phone)
        durability = {"durability": "exit"} if self.graph.checkpointer is not None else {}
        result = await self.graph.ainvoke(turn_input, config, **durability)
        await self._persist(msg.phone, result)
        turn = result.get("turn") or {}
        log.info(
            "agent turn",
            extra={
                "phone": msg.phone,
                "language": result.get("language"),
                "extraction_source": turn.get("extraction_source"),
                "message_intent": (turn.get("extraction") or {}).get("message_intent"),
                "parts": [p["kind"] for p in turn.get("parts") or []],
                "draft_source": turn.get("draft_source"),
                "reply_source": turn.get("reply_source"),
                "timings_ms": turn.get("timings"),
                "budget_skips": turn.get("budget_skips"),
                "agent_done_ms": deadline.elapsed_ms() if deadline else None,
                "booking_status": result.get("booking_status"),
                "is_qualified": result.get("is_qualified"),
            },
        )
        return list(turn.get("replies") or [])

    async def _seed(self, phone: str) -> dict[str, Any]:
        seed: dict[str, Any] = {
            "target_country": None, "intent": None, "budget_usd": None, "timeline": None,
            "is_qualified": False, "booking_status": "NONE", "selected_slot": None,
        }
        if self.sessionmaker is None:
            return seed
        try:
            async with self.sessionmaker() as session:
                lead = await repository.get_lead(session, phone)
                booking = await repository.latest_booking(session, lead.id) if lead else None
        except Exception:  # noqa: BLE001 - DB outage: start a fresh session
            log.warning("could not seed session from Postgres", exc_info=True)
            return seed
        if lead is not None:
            seed.update(
                target_country=lead.target_country, intent=lead.intent, budget_usd=lead.budget_usd,
                timeline=lead.timeline, is_qualified=bool(lead.is_qualified),
                # an offer's slots live in Redis and may be gone; re-offer instead of resuming SLOTS_OFFERED
                booking_status="BOOKED" if lead.booking_status == "BOOKED" else "NONE",
            )
            if booking is not None and lead.booking_status == "BOOKED":
                seed["selected_slot"] = booking.slot_iso
        return seed

    async def _persist(self, phone: str, state: dict) -> None:
        if self.sessionmaker is None:
            return
        budget = (state.get("turn") or {}).get("budget")
        fields = {
            "target_country": state.get("target_country"),
            "intent": state.get("intent"),
            "budget_usd": state.get("budget_usd"),
            "timeline": state.get("timeline"),
            "is_qualified": bool(state.get("is_qualified")),
            "booking_status": state.get("booking_status") or "NONE",
            "language": state.get("language"),
        }
        if budget:
            fields["budget_original_amount"] = budget["amount"]
            fields["budget_original_currency"] = budget["currency"]
        try:
            async with self.sessionmaker() as session:
                await repository.update_lead_profile(session, phone, **fields)
        except Exception:  # noqa: BLE001 - the reply still goes out; Redis keeps the session
            log.exception("failed to persist lead profile", extra={"phone": phone})
