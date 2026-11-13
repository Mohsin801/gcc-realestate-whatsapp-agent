"""services/processor.py: persistence, fallbacks, latency logging."""
import asyncio
import time

from sqlalchemy import select

from db.models import ChatLog, Lead
from services.inbound import IncomingMessage
from services.processor import process_message
from services.runtime import Runtime
from tests.conftest import RecordingSender


def _msg(text="Hello", wamid="wamid.P1", language="en", msg_type="text", name="Ahmed"):
    return IncomingMessage(
        wamid=wamid, phone="971501234567", profile_name=name, msg_type=msg_type,
        text=text, language=language, received_at=time.monotonic(),
    )


async def _agent(msg, language):
    return [f"reply-{language}"]


async def test_persists_lead_and_both_directions(fake_redis, sender, sessionmaker):
    rt = Runtime(sender=sender, sessionmaker=sessionmaker, run_agent=_agent)
    await process_message(_msg(), redis=fake_redis, runtime=rt)

    async with sessionmaker() as s:
        lead = await s.scalar(select(Lead))
        logs = (await s.scalars(select(ChatLog).order_by(ChatLog.id))).all()
    assert lead.phone == "971501234567" and lead.profile_name == "Ahmed" and lead.language == "en"
    assert [(l.direction, l.text) for l in logs] == [("in", "Hello"), ("out", "reply-en")]
    assert logs[0].wamid == "wamid.P1"
    assert logs[1].latency_ms is not None and logs[1].latency_ms >= 0
    assert sender.sent == [("971501234567", "reply-en")]


async def test_language_switch_updates_lead(fake_redis, sender, sessionmaker):
    rt = Runtime(sender=sender, sessionmaker=sessionmaker, run_agent=_agent)
    await process_message(_msg("Hi", "wamid.1", "en"), redis=fake_redis, runtime=rt)
    await process_message(_msg("هلا", "wamid.2", "ar"), redis=fake_redis, runtime=rt)
    assert sender.sent[-1][1] == "reply-ar"
    async with sessionmaker() as s:
        assert (await s.scalar(select(Lead))).language == "ar"


async def test_digits_only_keeps_previous_language(fake_redis, sender, sessionmaker):
    rt = Runtime(sender=sender, sessionmaker=sessionmaker, run_agent=_agent)
    await process_message(_msg("هلا", "wamid.1", "ar"), redis=fake_redis, runtime=rt)
    await process_message(_msg("2", "wamid.2", None), redis=fake_redis, runtime=rt)
    assert sender.sent[-1][1] == "reply-ar"


async def test_voice_note_gets_type_your_message_reply(fake_redis, sender, sessionmaker):
    async def must_not_run(msg, language):
        raise AssertionError("agent should not run for media")

    rt = Runtime(sender=sender, sessionmaker=sessionmaker, run_agent=must_not_run)
    await process_message(_msg("هلا", "wamid.1", "ar"), redis=fake_redis,
                          runtime=Runtime(sender=sender, sessionmaker=sessionmaker, run_agent=_agent))
    await process_message(_msg(None, "wamid.2", None, msg_type="audio"), redis=fake_redis, runtime=rt)
    assert "رسالتك" in sender.sent[-1][1]  # Arabic, because the lead's last language was Arabic


async def test_agent_error_sends_persona_safe_fallback(fake_redis, sender, sessionmaker):
    async def broken(msg, language):
        raise RuntimeError("LLM down")

    rt = Runtime(sender=sender, sessionmaker=sessionmaker, run_agent=broken)
    await process_message(_msg(), redis=fake_redis, runtime=rt)
    [(_, text)] = sender.sent
    assert "moment" in text.lower()
    for banned in ("ai", "bot", "assistant", "error"):
        assert banned not in text.lower().split()


async def test_agent_timeout_sends_fallback(fake_redis, sender, sessionmaker):
    async def slow(msg, language):
        await asyncio.sleep(5)
        return ["late"]

    rt = Runtime(sender=sender, sessionmaker=sessionmaker, run_agent=slow, agent_timeout_seconds=0.05)
    await process_message(_msg(), redis=fake_redis, runtime=rt)
    assert len(sender.sent) == 1 and sender.sent[0][1] != "late"


async def test_db_outage_still_replies(fake_redis, sender):
    def broken_sessionmaker():
        raise RuntimeError("postgres down")

    rt = Runtime(sender=sender, sessionmaker=broken_sessionmaker, run_agent=_agent)
    await process_message(_msg(), redis=fake_redis, runtime=rt)
    assert sender.sent == [("971501234567", "reply-en")]


async def test_send_failure_does_not_raise(fake_redis, sessionmaker):
    rt = Runtime(sender=RecordingSender(fail_send=True), sessionmaker=sessionmaker, run_agent=_agent)
    await process_message(_msg(), redis=fake_redis, runtime=rt)  # must not raise


async def test_same_phone_messages_are_serialised(fake_redis, sender, sessionmaker):
    active, max_active = 0, 0

    async def tracking(msg, language):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1
        return [msg.text]

    rt = Runtime(sender=sender, sessionmaker=sessionmaker, run_agent=tracking)
    await asyncio.gather(
        process_message(_msg("a", "wamid.a"), redis=fake_redis, runtime=rt),
        process_message(_msg("b", "wamid.b"), redis=fake_redis, runtime=rt),
    )
    assert max_active == 1
    assert sorted(t for _, t in sender.sent) == ["a", "b"]
