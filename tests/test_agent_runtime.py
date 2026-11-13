"""Spec state contract, LLM provider wiring, and the runner's Postgres seeding/persistence."""
import asyncio
import operator
import time
import typing

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from agent.graph import build_graph
from agent.llm import LangChainLLM, LLMConfigError, _text_of, build_llm
from agent.runner import AgentService
from agent.state import GCCRealEstateState
from core.config import Settings
from db.models import Booking, ChatLog, Lead, Owner
from services.inbound import IncomingMessage
from services.processor import process_message
from services.runtime import Runtime
from tests.agent_fakes import FakeScheduler

PHONE = "971501234567"


# --- spec §3 contract -----------------------------------------------------------------------------
def test_state_matches_spec_exactly():
    hints = typing.get_type_hints(GCCRealEstateState, include_extras=True)
    assert list(hints) == [
        "phone_number", "language", "target_country", "intent", "budget_usd", "timeline", "messages",
        "is_qualified", "booking_status", "selected_slot",
    ]
    assert hints["phone_number"] is str and hints["language"] is str
    assert hints["budget_usd"] == typing.Optional[float]
    assert hints["is_qualified"] is bool and hints["booking_status"] is str
    for f in ("target_country", "intent", "timeline", "selected_slot"):
        assert hints[f] == typing.Optional[str]
    args = typing.get_args(hints["messages"])
    assert args[0] == typing.Sequence[BaseMessage] and args[1] is operator.add


# --- LLM provider -----------------------------------------------------------------------------------
def test_build_llm_requires_key_and_model_from_config():
    with pytest.raises(LLMConfigError):
        build_llm(Settings(_env_file=None, gemini_api_key="k", gemini_model=""))
    with pytest.raises(LLMConfigError):
        build_llm(Settings(_env_file=None, llm_provider="nope"))


def test_build_llm_gemini_uses_configured_model():
    llm = build_llm(Settings(_env_file=None, gemini_api_key="k", gemini_model="gemini-test-flash",
                             gemini_thinking_level="minimal", llm_timeout_seconds=3))
    assert isinstance(llm, LangChainLLM)
    assert llm._model.model.endswith("gemini-test-flash")
    assert llm._model.thinking_level == "minimal" and llm._model.max_retries == 0


async def test_langchain_llm_generate_and_timeout():
    llm = LangChainLLM(FakeListChatModel(responses=["hello there"]), timeout_seconds=1)
    assert await llm.generate("sys", [HumanMessage("hi")]) == "hello there"

    slow = LangChainLLM(FakeListChatModel(responses=["late"], sleep=0.5), timeout_seconds=0.05)
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await slow.generate("sys", [HumanMessage("hi")])


def test_text_of_content_blocks():
    assert _text_of([{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "Hi "}, "there"]) == "Hi there"


# --- runner: Postgres seeding and persistence ----------------------------------------------------
def _msg(text, wamid="wamid.R", language="en"):
    return IncomingMessage(wamid, PHONE, "Ahmed", "text", text, language, time.monotonic())


async def test_new_session_is_seeded_from_postgres_lead(sessionmaker):
    async with sessionmaker() as s:
        s.add(Lead(phone=PHONE, language="en", target_country="QATAR", intent="GOLDEN_VISA", budget_usd=800_000.0))
        await s.commit()
    graph = build_graph(llm=None, settings=Settings(_env_file=None), checkpointer=InMemorySaver())
    agent = AgentService(graph, sessionmaker)
    await agent(_msg("hello again"), "en")
    st = (await graph.aget_state({"configurable": {"thread_id": PHONE}})).values
    assert (st["target_country"], st["intent"], st["budget_usd"]) == ("QATAR", "GOLDEN_VISA", 800_000.0)
    assert st["turn"]["parts"][-1]["field"] == "timeline"  # not asked again for country/intent/budget


async def test_booked_lead_is_seeded_with_slot(sessionmaker):
    async with sessionmaker() as s:
        s.add(Owner(id="default", name="Owner"))
        lead = Lead(phone=PHONE, language="en", target_country="DUBAI", intent="INVESTMENT", budget_usd=1e6,
                    timeline="IMMEDIATE", is_qualified=True, booking_status="BOOKED")
        s.add(lead)
        await s.flush()
        s.add(Booking(lead_id=lead.id, owner_id="default", slot_iso="2026-11-20T08:00:00+00:00"))
        await s.commit()
    graph = build_graph(llm=None, settings=Settings(_env_file=None), checkpointer=InMemorySaver())
    await AgentService(graph, sessionmaker)(_msg("thanks"), "en")
    st = (await graph.aget_state({"configurable": {"thread_id": PHONE}})).values
    assert st["booking_status"] == "BOOKED" and st["selected_slot"] == "2026-11-20T08:00:00+00:00"


async def test_turn_results_are_persisted_to_lead(sessionmaker):
    graph = build_graph(llm=None, settings=Settings(_env_file=None), checkpointer=InMemorySaver())
    agent = AgentService(graph, sessionmaker)
    await agent(_msg("ميزانيتي مليونين ريال في الرياض", language="ar"), "ar")
    async with sessionmaker() as s:
        lead = await s.scalar(select(Lead).where(Lead.phone == PHONE))
    assert lead.target_country == "SAUDI" and lead.language == "ar"
    assert (lead.budget_original_amount, lead.budget_original_currency) == (2_000_000, "SAR")
    assert lead.budget_usd == pytest.approx(533_333.33, abs=0.01)
    assert lead.is_qualified is False and lead.booking_status == "NONE"


async def test_processor_runs_real_graph_end_to_end(fake_redis, sender, sessionmaker):
    graph = build_graph(llm=None, settings=Settings(_env_file=None), scheduler=FakeScheduler(),
                        checkpointer=InMemorySaver())
    rt = Runtime(sender=sender, sessionmaker=sessionmaker, run_agent=AgentService(graph, sessionmaker))
    await process_message(_msg("السلام عليكم", "wamid.1", "ar"), redis=fake_redis, runtime=rt)
    assert "أي سوق يهمك" in sender.sent[-1][1]
    async with sessionmaker() as s:
        logs = (await s.scalars(select(ChatLog).order_by(ChatLog.id))).all()
    assert [l.direction for l in logs] == ["in", "out"] and logs[1].latency_ms is not None


async def test_ai_message_is_appended_to_history():
    graph = build_graph(llm=None, settings=Settings(_env_file=None), checkpointer=InMemorySaver())
    await AgentService(graph)(_msg("hi"), "en")
    msgs = (await graph.aget_state({"configurable": {"thread_id": PHONE}})).values["messages"]
    assert isinstance(msgs[0], HumanMessage) and isinstance(msgs[1], AIMessage)
