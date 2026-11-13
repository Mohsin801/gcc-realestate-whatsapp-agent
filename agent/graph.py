"""Builds the LangGraph state machine (spec §6: bilingual agent with EN + Gulf Arabic intent classifiers).

    START -> detect_language -> extract -> normalize --+--> answer_question --+
                                                      |                      |
                                                      +----------+-----------+
                                                                 v
                                  qualify | offer_slots | book_slot | capture_email | follow_up
                                                                 v
                                                     respond -> persona_guard -> END
"""
from typing import Any

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
    g.add_node("detect_language", nodes.detect_language)
    g.add_node("extract", nodes.extract)
    g.add_node("normalize", nodes.normalize)
    g.add_node("answer_question", nodes.answer_question)
    for name in STEP_NODES:
        g.add_node(name, getattr(nodes, name))
    g.add_node("respond", nodes.respond)
    g.add_node("persona_guard", nodes.persona_guard)

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
