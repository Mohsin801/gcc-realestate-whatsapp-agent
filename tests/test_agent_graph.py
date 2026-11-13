"""The LangGraph agent end to end with a scripted LLM, in-memory checkpointer and fake calendar."""
import time

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph
from agent.language import detect_language
from agent.persona_guard import reveals_ai
from agent.prompts import Extraction
from agent.runner import AgentService
from agent.services import Passage
from core.config import Settings
from services.inbound import IncomingMessage
from tests.agent_fakes import AR_REPLY, EN_REPLY, FakeKnowledge, FakeLLM, FakeScheduler

PHONE = "971501234567"
CONFIG = {"configurable": {"thread_id": PHONE}}


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def make_agent(llm=None, scheduler=None, knowledge=None, **cfg):
    graph = build_graph(llm=llm, settings=settings(**cfg), scheduler=scheduler, knowledge=knowledge,
                        checkpointer=InMemorySaver())
    return AgentService(graph), graph


async def say(agent, text, language=None, name="Ahmed Al Mansoori"):
    lang = language or detect_language(text) or "en"
    msg = IncomingMessage("wamid.x", PHONE, name, "text", text, lang, time.monotonic())
    return await agent(msg, lang)


async def state_of(graph):
    return (await graph.aget_state(CONFIG)).values


def parts(state):
    return [p["kind"] for p in state["turn"]["parts"]]


EN_SCRIPT = {
    "Hi there": Extraction(message_intent="GREETING"),
    "Looking in Dubai, for investment": Extraction(message_intent="PROVIDE_INFO", target_country="DUBAI",
                                                    intent="INVESTMENT"),
    "Around 2M AED": Extraction(message_intent="PROVIDE_INFO", budget_text="Around 2M AED",
                                budget_amount=2_000_000, budget_currency="AED"),
    "Ready now": Extraction(message_intent="PROVIDE_INFO", timeline="IMMEDIATE"),
}


async def test_full_english_flow_qualify_offer_book_email():
    llm, cal = FakeLLM(EN_SCRIPT), FakeScheduler()
    agent, graph = make_agent(llm, cal)

    replies = await say(agent, "Hi there")
    st = await state_of(graph)
    assert replies == [EN_REPLY]
    assert parts(st) == ["GREETING", "ASK"] and st["turn"]["parts"][1]["field"] == "target_country"
    assert "Ahmed" in llm.generate_systems[-1]  # greets by first name

    await say(agent, "Looking in Dubai, for investment")
    st = await state_of(graph)
    assert (st["target_country"], st["intent"]) == ("DUBAI", "INVESTMENT")
    assert st["turn"]["parts"][-1] == {"kind": "ASK", "field": "budget_usd", "booking_requested": False}

    await say(agent, "Around 2M AED")
    st = await state_of(graph)
    assert st["budget_usd"] == pytest.approx(2_000_000 / 3.6725, abs=0.01)
    assert st["turn"]["parts"][-1]["field"] == "timeline"
    assert st["is_qualified"] is False

    replies = await say(agent, "Ready now")
    st = await state_of(graph)
    assert st["is_qualified"] is True and st["booking_status"] == "SLOTS_OFFERED"
    assert len(replies) == 2
    assert replies[1].startswith("1. Tomorrow at 4:00 PM GST\n2. Thursday at 11:00 AM GST\n3. Friday at 2:30 PM GST")

    n_extract_calls = len(llm.extract_calls)
    replies = await say(agent, "2")
    st = await state_of(graph)
    assert len(llm.extract_calls) == n_extract_calls  # lone digit: fast path, no LLM extraction
    assert st["booking_status"] == "BOOKED" and st["selected_slot"] == "2026-11-16T12:00:00+00:00"
    assert cal.booked == [(PHONE, "Ahmed Al Mansoori", 2)]
    assert "Thursday at 11:00 AM GST" in replies[-1] and "https://meet.google.com/abc-defg-hij" in replies[-1]

    await say(agent, "ahmed@example.com")
    st = await state_of(graph)
    assert cal.emails == [(PHONE, "ahmed@example.com")]
    assert parts(st) == ["EMAIL_SAVED"]
    # messages accumulate across turns (operator.add): 6 human + 6 AI
    assert len(st["messages"]) == 12


async def test_arabic_flow_without_llm_uses_rules_and_khaleeji_templates():
    agent, graph = make_agent(llm=None, scheduler=FakeScheduler())
    r = await say(agent, "السلام عليكم")
    assert "هلا والله Ahmed" in r[0] and "أي سوق يهمك" in r[0]
    r = await say(agent, "ابي شقة في دبي للاستثمار")
    assert "الميزانية" in r[0]
    r = await say(agent, "ميزانيتي مليونين درهم")
    assert "متى ناوي تشتري" in r[0]
    r = await say(agent, "الحين على طول")
    st = await state_of(graph)
    assert (st["target_country"], st["intent"], st["timeline"]) == ("DUBAI", "INVESTMENT", "IMMEDIATE")
    assert st["budget_usd"] == pytest.approx(2_000_000 / 3.6725, abs=0.01)
    assert st["booking_status"] == "SLOTS_OFFERED"
    assert "رد علي برقم الموعد" in r[1]
    r = await say(agent, "٢", language="ar")  # digits only: the processor passes the lead's last language
    assert (await state_of(graph))["booking_status"] == "BOOKED"
    assert "إيميلك" in r[0]


async def test_language_switch_mid_conversation():
    agent, graph = make_agent(FakeLLM({"Hello": Extraction(message_intent="GREETING")}))
    assert await say(agent, "Hello") == [EN_REPLY]
    assert await say(agent, "هلا، ابي أكمل بالعربي") == [AR_REPLY]
    assert (await state_of(graph))["language"] == "ar"


async def test_digits_only_keeps_previous_language():
    agent, graph = make_agent(llm=None)
    await say(agent, "السلام عليكم")
    await say(agent, "123", language="ar")  # processor passes the lead's last language
    assert (await state_of(graph))["language"] == "ar"


async def test_without_live_calendar_qualified_lead_gets_handoff():
    script = {"Dubai, investment, 2M AED, ready now": Extraction(
        message_intent="PROVIDE_INFO", target_country="DUBAI", intent="INVESTMENT",
        budget_text="2M AED", timeline="IMMEDIATE")}
    agent, graph = make_agent(FakeLLM(script), scheduler=None, )
    replies = await say(agent, "Dubai, investment, 2M AED, ready now")
    st = await state_of(graph)
    assert st["is_qualified"] and st["booking_status"] == "NONE"
    assert "NO_SLOTS" in parts(st) and len(replies) == 1


async def test_faq_without_knowledge_defers_and_never_states_figures():
    script = {"What are the DLD fees?": Extraction(message_intent="FAQ_QUESTION", target_country="DUBAI",
                                                   question="What are the Dubai Land Department fees?")}
    hallucinating = FakeLLM(script, replies=["DLD fee is 4% of the price.", "It's 4 percent plus admin fees."])
    knowledge = FakeKnowledge([])
    agent, graph = make_agent(hallucinating, knowledge=knowledge)
    replies = await say(agent, "What are the DLD fees?")
    st = await state_of(graph)
    assert "DEFER" in parts(st)
    assert knowledge.queries == [("What are the Dubai Land Department fees?", "DUBAI", "en")]
    assert "4" not in " ".join(replies) and "%" not in " ".join(replies)
    assert "consultant" in replies[0]


async def test_faq_with_knowledge_passes_passages_to_reply_prompt():
    script = {"Saudi RETT?": Extraction(message_intent="FAQ_QUESTION", target_country="SAUDI",
                                        question="What is the Saudi RETT?")}
    passage = Passage(text="RETT is 5% of the property value (ZATCA).", source="https://zatca.gov.sa")
    llm = FakeLLM(script, replies=["The RETT is 5% of the property value. Which purpose is the purchase for?"])
    agent, graph = make_agent(llm, knowledge=FakeKnowledge([passage]))
    replies = await say(agent, "Saudi RETT?")
    st = await state_of(graph)
    assert "ANSWER" in parts(st) and "ASK" in parts(st)
    assert "RETT is 5% of the property value (ZATCA)." in llm.generate_systems[-1]
    assert replies[0].startswith("The RETT is 5%")  # grounded figure is allowed


async def test_persona_guard_regenerates_ai_disclosure():
    llm = FakeLLM({"Hello": Extraction(message_intent="GREETING")},
                  replies=["As an AI assistant, I'd love to help!", "Hello! Which market are you looking at?"])
    agent, graph = make_agent(llm)
    replies = await say(agent, "Hello")
    assert replies == ["Hello! Which market are you looking at?"]
    assert "rejected (ai_disclosure)" in llm.generate_systems[-1]


async def test_persona_guard_falls_back_to_template_when_retry_also_fails():
    llm = FakeLLM({"هلا": Extraction(message_intent="GREETING")},
                  replies=["أنا مساعد آلي، كيف أخدمك؟", "أنا بوت هنا لخدمتك"])
    agent, _ = make_agent(llm)
    replies = await say(agent, "هلا")
    assert not reveals_ai(" ".join(replies))
    assert "أي سوق يهمك" in replies[0]


async def test_persona_conceal_flag_off_allows_disclosure():
    llm = FakeLLM({"Are you a bot?": Extraction(message_intent="OTHER")},
                  replies=["Yes, I'm an automated assistant; a human consultant handles the call."])
    agent, _ = make_agent(llm, persona_conceal_ai=False)
    replies = await say(agent, "Are you a bot?")
    assert replies[0].startswith("Yes, I'm an automated assistant")
    assert "automated assistant" in llm.generate_systems[0] and "Never say" not in llm.generate_systems[0]


async def test_conceal_flag_on_is_default_and_in_prompt():
    llm = FakeLLM({"Hi": Extraction(message_intent="GREETING")})
    agent, _ = make_agent(llm)
    await say(agent, "Hi")
    assert "Never say or imply that you are an AI" in llm.generate_systems[0]


async def test_wrong_language_reply_is_rejected():
    llm = FakeLLM({"السلام عليكم": Extraction(message_intent="GREETING")},
                  replies=["Hello! Which market?", "Hi again in English"])
    agent, _ = make_agent(llm)
    replies = await say(agent, "السلام عليكم")
    assert detect_language(replies[0]) == "ar"


async def test_llm_outage_uses_rules_and_templates():
    agent, graph = make_agent(FakeLLM(fail_extract=True, fail_generate=True))
    replies = await say(agent, "I want a villa in Riyadh for my family")
    st = await state_of(graph)
    assert (st["target_country"], st["intent"]) == ("SAUDI", "PERSONAL_USE")
    assert st["turn"]["extraction_source"] == "rules" and st["turn"]["draft_source"] == "template"
    assert "budget" in replies[0].lower()


async def test_long_reply_is_split_into_at_most_two_short_bubbles():
    long = " ".join(["Dubai has many great communities for families and investors alike."] * 12)
    agent, _ = make_agent(FakeLLM({"Hi": Extraction(message_intent="GREETING")}, replies=[long[:790]]))
    replies = await say(agent, "Hi")
    assert 1 <= len(replies) <= 2 and all(len(r) <= 400 for r in replies)


async def test_slot_taken_offers_new_times():
    agent, graph = make_agent(FakeLLM(EN_SCRIPT), FakeScheduler(taken=True))
    for text in ["Looking in Dubai, for investment", "Around 2M AED", "Ready now"]:
        await say(agent, text)
    replies = await say(agent, "1")
    st = await state_of(graph)
    assert st["booking_status"] == "SLOTS_OFFERED" and parts(st) == ["SLOT_TAKEN"]
    assert "Sunday at 10:00 AM GST" in replies[-1]


async def test_unclear_slot_reply_relists_times():
    script = {**EN_SCRIPT, "hmm let me think": Extraction(message_intent="OTHER")}
    agent, graph = make_agent(FakeLLM(script), FakeScheduler())
    for text in ["Looking in Dubai, for investment", "Around 2M AED", "Ready now"]:
        await say(agent, text)
    replies = await say(agent, "hmm let me think")
    st = await state_of(graph)
    assert parts(st) == ["SLOT_UNCLEAR"] and st["booking_status"] == "SLOTS_OFFERED"
    assert replies[-1].startswith("1. Tomorrow")


async def test_booking_failure_hands_off():
    agent, graph = make_agent(FakeLLM(EN_SCRIPT), FakeScheduler(fail_book=True))
    for text in ["Looking in Dubai, for investment", "Around 2M AED", "Ready now", "3"]:
        await say(agent, text)
    st = await state_of(graph)
    assert st["booking_status"] == "NONE" and parts(st) == ["BOOKING_FAILED"]


async def test_booking_request_before_qualification_still_qualifies_first():
    script = {"Can I talk to someone?": Extraction(message_intent="BOOKING_REQUEST")}
    llm = FakeLLM(script)
    agent, graph = make_agent(llm, FakeScheduler())
    await say(agent, "Can I talk to someone?")
    st = await state_of(graph)
    assert st["booking_status"] == "NONE"
    assert st["turn"]["parts"][-1] == {"kind": "ASK", "field": "target_country", "booking_requested": True}
    assert "They want a call" in llm.generate_systems[-1]
