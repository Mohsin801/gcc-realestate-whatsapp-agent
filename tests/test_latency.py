"""Phase 5 latency hardening: the per-turn budget, node timings, slot prefetch, warm-up."""
import time

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph
from agent.nodes import AgentNodes
from agent.prompts import Extraction
from agent.runner import AgentService
from agent.services import Passage
from core.config import Settings
from core.latency import Deadline, LatencyStats, percentile
from services.inbound import IncomingMessage
from services.warmup import warm_up
from tests.agent_fakes import FakeKnowledge, FakeLLM, FakeScheduler

PHONE = "971501234567"
CONFIG = {"configurable": {"thread_id": PHONE}}
# A small budget keeps the tests fast; the proportions match production (3 s, 0.5 s send, 1 s reply).
FAST = dict(latency_budget_seconds=0.6, latency_send_reserve_seconds=0.1, latency_reply_reserve_seconds=0.2,
            latency_min_call_seconds=0.05)


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def make_agent(llm=None, scheduler=None, knowledge=None, budget=True, **cfg):
    s = settings(**{**FAST, **cfg})
    graph = build_graph(llm=llm, settings=s, scheduler=scheduler, knowledge=knowledge, checkpointer=InMemorySaver())
    return AgentService(graph, budget_seconds=s.latency_budget_seconds if budget else None), graph


async def say(agent, text, language="en", received_at=None):
    msg = IncomingMessage("wamid.l", PHONE, "Ahmed", "text", text, language,
                          time.monotonic() if received_at is None else received_at)
    start = time.monotonic()
    replies = await agent(msg, language)
    return replies, time.monotonic() - start


async def turn_of(graph):
    return (await graph.aget_state(CONFIG)).values["turn"]


# --- core.latency -----------------------------------------------------------------------------------
class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def test_deadline_caps_calls_and_keeps_reserves():
    clock = Clock()
    d = Deadline(3.0, start=clock.t, clock=clock)
    assert d.allow(4.0) == pytest.approx(3.0)
    assert d.allow(1.0, keep=0.5) == 1.0
    clock.t += 2.0
    assert d.allow(4.0, keep=0.5) == pytest.approx(0.5)
    assert d.allow(4.0, keep=0.5, minimum=0.6) is None
    assert d.elapsed_ms() == 2000
    clock.t += 5
    assert d.allow(4.0) is None and d.remaining() < 0


def test_latency_stats():
    assert percentile([], 95) is None
    assert percentile([5, 1, 3, 2, 4], 50) == 3
    assert percentile(range(1, 101), 95) == 95
    stats = LatencyStats.of([1000, 2000, 2900, 3500], budget_ms=3000)
    assert stats.count == 4 and stats.max_ms == 3500 and stats.within_budget == 0.75
    assert "p95=3500 ms" in stats.line()
    assert LatencyStats.of([], 3000).line() == "no data"


# --- the agent under a latency budget ---------------------------------------------------------------
SCRIPT = {"Looking in Dubai, for investment": Extraction(message_intent="PROVIDE_INFO", target_country="DUBAI",
                                                          intent="INVESTMENT")}


async def test_slow_extraction_is_cut_short_and_rules_take_over():
    llm = FakeLLM(SCRIPT, extract_delay=5.0)
    agent, graph = make_agent(llm)
    replies, took = await say(agent, "Looking in Dubai, for investment")
    turn = await turn_of(graph)
    assert took < 0.6  # the reply is capped at budget - send reserve; the reserve absorbs graph overhead
    assert turn["extraction_source"] == "rules"  # the rules still read country + purpose
    state = (await graph.aget_state(CONFIG)).values
    assert state["target_country"] == "DUBAI" and state["intent"] == "INVESTMENT"
    (_, extract_timeout), (_, reply_timeout) = llm.timeouts
    assert extract_timeout == pytest.approx(0.3, abs=0.02)  # 0.6 budget - 0.1 send - 0.2 kept for the reply
    assert 0.05 <= reply_timeout <= 0.21
    assert turn["reply_source"] == "llm" and replies


async def test_slow_reply_falls_back_to_template_in_time():
    agent, graph = make_agent(FakeLLM(SCRIPT, generate_delay=5.0))
    replies, took = await say(agent, "Looking in Dubai, for investment")
    turn = await turn_of(graph)
    assert took < 0.6
    assert turn["extraction_source"] == "llm" and turn["reply_source"] == "template"
    assert replies == [turn["fallback"]]


async def test_budget_already_spent_skips_every_llm_call():
    llm = FakeLLM(SCRIPT)
    agent, graph = make_agent(llm)
    replies, _ = await say(agent, "Looking in Dubai, for investment", received_at=time.monotonic() - 10)
    turn = await turn_of(graph)
    assert llm.timeouts == []  # never called
    assert turn["budget_skips"] == ["extract", "reply"]
    assert turn["extraction_source"] == "rules" and turn["reply_source"] == "template" and replies


async def test_guard_retry_is_skipped_when_no_time_is_left():
    # the first draft leaves ~0.07 s after the send reserve: less than the 0.15 s minimum for a call
    llm = FakeLLM(SCRIPT, replies=["As an AI assistant, I'd love to help!"], generate_delay=0.43)
    agent, graph = make_agent(llm, latency_min_call_seconds=0.15)
    replies, took = await say(agent, "Looking in Dubai, for investment")
    turn = await turn_of(graph)
    assert "guard_retry" in turn["budget_skips"]
    assert turn["reply_source"] == "template" and "AI" not in replies[0]
    assert [c for c, _ in llm.timeouts].count("generate") == 1


async def test_slow_knowledge_search_defers_in_time():
    class SlowKnowledge(FakeKnowledge):
        async def search(self, query, country, language):
            import asyncio
            await asyncio.sleep(5)
            return [Passage(text="never used")]

    agent, graph = make_agent(knowledge=SlowKnowledge())
    replies, took = await say(agent, "What are the DLD fees in Dubai?")
    assert took < 0.6
    assert "DEFER" in [p["kind"] for p in (await turn_of(graph))["parts"]]


async def test_without_a_budget_calls_get_the_configured_timeout():
    llm = FakeLLM(SCRIPT)
    agent, graph = make_agent(llm, budget=False, llm_timeout_seconds=4.0)
    await say(agent, "Looking in Dubai, for investment", received_at=time.monotonic() - 10)
    assert llm.timeouts == [("extract", 4.0), ("generate", 4.0)]
    assert "budget_skips" not in await turn_of(graph)


async def test_every_node_is_timed():
    agent, graph = make_agent(FakeLLM(SCRIPT))
    await say(agent, "Looking in Dubai, for investment")
    timings = (await turn_of(graph))["timings"]
    assert set(timings) == {"detect_language", "extract", "normalize", "qualify", "respond", "persona_guard"}
    assert all(isinstance(ms, float) and ms >= 0 for ms in timings.values())


async def test_deadline_is_not_written_to_checkpoint_metadata():
    agent, graph = make_agent(FakeLLM(SCRIPT))
    await say(agent, "Looking in Dubai, for investment")
    snapshot = await graph.aget_state(CONFIG)
    assert "deadline" not in snapshot.metadata


async def test_slots_are_prefetched_when_one_answer_is_missing():
    cal = FakeScheduler()
    agent, _ = make_agent(scheduler=cal, budget=False)
    await say(agent, "Looking in Dubai, for investment")  # budget and timeline still missing
    assert cal.prefetches == 0
    await say(agent, "My budget is 2M AED")  # only the timeline left
    assert cal.prefetches == 1
    await say(agent, "Ready now")  # qualified: slots offered
    assert cal.offer_calls == 1


async def test_prefetch_errors_never_break_the_turn():
    class BrokenPrefetch(FakeScheduler):
        def prefetch(self):
            raise RuntimeError("redis down")

    agent, _ = make_agent(scheduler=BrokenPrefetch(), budget=False)
    await say(agent, "Looking in Dubai, for investment")
    replies, _ = await say(agent, "My budget is 2M AED")
    assert replies


def test_rules_treat_topic_questions_as_questions_even_with_a_purpose():
    ask = lambda text: AgentNodes._rule_extract(text, {}, []).message_intent  # noqa: E731
    assert ask("What is the minimum property value for residency in Qatar?") == "FAQ_QUESTION"
    assert ask("Can I get a golden visa if I buy in Dubai?") == "FAQ_QUESTION"
    assert ask("هل يحق لي الاقامه الذهبيه اذا اشتريت في دبي؟") == "FAQ_QUESTION"
    assert ask("Hello, I want a property in Doha to get residency") == "PROVIDE_INFO"
    assert ask("I want the golden visa, budget 2M AED?") == "PROVIDE_INFO"  # gives a budget: info


# --- warm-up ----------------------------------------------------------------------------------------
async def test_warm_up_calls_each_dependency_and_never_raises():
    class Cal:
        async def warm(self):
            return True

    llm, kb = FakeLLM(), FakeKnowledge()
    assert await warm_up(llm, kb, Cal()) == {"llm": True, "knowledge": True, "calendar": True}
    assert len(llm.generate_systems) == 1 and kb.queries

    assert await warm_up(FakeLLM(fail_generate=True), None) == {"llm": False}
    assert await warm_up(None, None) == {}


# --- production latency report (scripts/latency_report.py) ------------------------------------------
async def test_latency_report_reads_first_bubble_latency_from_chat_logs(sessionmaker):
    from datetime import UTC, datetime, timedelta

    from db import repository
    from scripts.latency_report import summarise

    async with sessionmaker() as s:
        lead = await repository.get_or_create_lead(s, PHONE, "Ahmed", "en")
        await repository.log_message(s, lead.id, "in", "hi", language="en", wamid="w1")
        for lang, ms in (("en", 1200), ("en", 2600), ("ar", 1900), ("ar", 3400)):
            await repository.log_message(s, lead.id, "out", "reply", language=lang, latency_ms=ms)
        await repository.log_message(s, lead.id, "out", "second bubble", language="en")  # no latency
        rows = await repository.reply_latencies(s, datetime.now(UTC) - timedelta(days=1))
        assert await repository.reply_latencies(s, datetime.now(UTC) + timedelta(days=1)) == []
    assert sorted(rows) == [("ar", 1900), ("ar", 3400), ("en", 1200), ("en", 2600)]
    stats = summarise(rows, budget_ms=3000)
    assert stats["all"].count == 4 and stats["all"].within_budget == 0.75
    assert stats["en"].max_ms == 2600 and stats["ar"].p95_ms == 3400
