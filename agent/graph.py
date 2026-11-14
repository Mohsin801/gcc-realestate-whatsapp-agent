"""Builds the LangGraph state machine (spec §6: bilingual agent with EN + Gulf Arabic intent classifiers).

    START -> detect_language -> extract -> normalize --+--> answer_question --+
                                                      |                      |
                                                      +----------+-----------+
                                                                 v
                                  qualify | offer_slots | book_slot | capture_email | follow_up
                                                                 v
                                                     respond -> persona_guard -> END

Every node is wrapped by ``timed`` so ``turn["timings"]`` holds each node's duration in ms (logged per turn,
used by the evaluation report to find slow stages).
"""
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agent.llm import LLMProvider
from agent.nodes import AgentNodes
from agent.services import Knowledge, NullKnowledge, NullScheduler, Scheduler
from agent.state import GCCRealEstateState
from core.config import Settings

STEP_NODES = ("qualify", "offer_slots", "book_slot", "capture_email", "follow_up")


class AgentGraphState(GCCRealEstateState, total=False):
    """The spec state plus ``turn``: per-turn working data, reset by the runner at the start of each turn."""

    turn: dict[str, Any]


def timed(name: str, fn: Callable[..., Awaitable[dict]]) -> Callable[[dict, RunnableConfig], Awaitable[dict]]:
    """Wrap a node so its wall time is added to ``turn["timings"][name]`` (ms)."""
    takes_config = len(inspect.signature(fn).parameters) > 1

    async def node(state: dict, config: RunnableConfig) -> dict:
        start = time.perf_counter()
        update = await (fn(state, config) if takes_config else fn(state))
        turn = dict(update.get("turn") or state.get("turn") or {})
        turn["timings"] = {**(turn.get("timings") or {}), name: round((time.perf_counter() - start) * 1000, 1)}
        return {**update, "turn": turn}

    # no functools.wraps: LangGraph reads the signature to decide whether to pass ``config``
    node.__name__ = node.__qualname__ = name
    return node


def build_graph(
    *,
    llm: LLMProvider | None,
    settings: Settings,
    scheduler: Scheduler | None = None,
    knowledge: Knowledge | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    nodes = AgentNodes(
        llm=llm, settings=settings, scheduler=scheduler or NullScheduler(), knowledge=knowledge or NullKnowledge()
    )
    g = StateGraph(AgentGraphState)
    for name in ("detect_language", "extract", "normalize", "answer_question", *STEP_NODES, "respond",
                 "persona_guard"):
        g.add_node(name, timed(name, getattr(nodes, name)))

    g.add_edge(START, "detect_language")
    g.add_edge("detect_language", "extract")
    g.add_edge("extract", "normalize")
    g.add_conditional_edges("normalize", AgentNodes.route_after_normalize, ["answer_question", *STEP_NODES])
    g.add_conditional_edges("answer_question", AgentNodes.next_step, list(STEP_NODES))
    for name in STEP_NODES:
        g.add_edge(name, "respond")
    g.add_edge("respond", "persona_guard")
    g.add_edge("persona_guard", END)
    return g.compile(checkpointer=checkpointer)
