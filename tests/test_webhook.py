"""api/whatsapp.py: Meta verification, signature check, wamid dedup, fast 200 + background processing."""
import time

import httpx
import pytest
from fastapi import BackgroundTasks

from api.whatsapp import handle_incoming_msg
from main import app
from services.redis import get_redis
from services.runtime import Runtime, set_runtime
from tests.conftest import TEST_VERIFY_TOKEN, encode, make_text_payload, sign


@pytest.fixture
async def client(fake_redis, sender, sessionmaker):
    """ASGI client without the real lifespan: fake Redis/DB/sender, echo agent."""

    async def echo_agent(msg, language):
        return [f"echo:{language}:{msg.text}"]

    set_runtime(Runtime(sender=sender, sessionmaker=sessionmaker, run_agent=echo_agent))
    app.dependency_overrides[get_redis] = lambda: fake_redis
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    set_runtime(None)


async def _post(client, payload, signature=None):
    body = encode(payload)
    headers = {"Content-Type": "application/json", "X-Hub-Signature-256": signature or sign(body)}
    return await client.post("/whatsapp/webhook", content=body, headers=headers)


# --- GET verification handshake ---
async def test_verify_webhook_echoes_challenge(client):
    r = await client.get(
        "/whatsapp/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": TEST_VERIFY_TOKEN, "hub.challenge": "12345"},
    )
    assert r.status_code == 200 and r.text == "12345"


async def test_verify_webhook_rejects_wrong_token(client):
    r = await client.get(
        "/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "nope", "hub.challenge": "1"}
    )
    assert r.status_code == 403


# --- Signature ---
async def test_rejects_bad_signature(client, sender):
    r = await _post(client, make_text_payload("hi"), signature="sha256=deadbeef")
    assert r.status_code == 403
    assert sender.sent == []


async def test_rejects_missing_signature(client, sender):
    r = await client.post("/whatsapp/webhook", content=encode(make_text_payload("hi")),
                          headers={"Content-Type": "application/json"})
    assert r.status_code == 403
    assert sender.sent == []


# --- End-to-end through background task ---
async def test_valid_message_is_processed_and_answered(client, sender):
    r = await _post(client, make_text_payload("Hello, looking in Dubai", wamid="wamid.E2E"))
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert sender.read == ["wamid.E2E"]
    assert sender.sent == [("971501234567", "echo:en:Hello, looking in Dubai")]


async def test_arabic_message_detected_and_passed_to_agent(client, sender):
    await _post(client, make_text_payload("مطلوب شقة في دبي", wamid="wamid.AR"))
    assert sender.sent == [("971501234567", "echo:ar:مطلوب شقة في دبي")]


async def test_duplicate_wamid_answered_once(client, sender):
    payload = make_text_payload("hi", wamid="wamid.DUP")
    r1 = await _post(client, payload)
    r2 = await _post(client, payload)  # Meta retry
    assert r1.status_code == r2.status_code == 200
    assert len(sender.sent) == 1


async def test_status_callback_returns_200_without_reply(client, sender):
    payload = make_text_payload("x")
    value = payload["entry"][0]["changes"][0]["value"]
    del value["messages"]
    value["statuses"] = [{"id": "wamid.S", "status": "read"}]
    r = await _post(client, payload)
    assert r.status_code == 200 and sender.sent == []


# --- Meta SLA: handler itself must return fast and only *queue* the work ---
async def test_handler_returns_under_200ms_and_defers_processing(fake_redis, sender, sessionmaker):
    async def slow_agent(msg, language):  # would blow the SLA if run inline
        raise AssertionError("agent must not run inside the request handler")

    set_runtime(Runtime(sender=sender, sessionmaker=sessionmaker, run_agent=slow_agent))
    try:
        tasks = BackgroundTasks()
        start = time.perf_counter()
        result = await handle_incoming_msg(make_text_payload("hi", wamid="wamid.SLA"), tasks, redis=fake_redis)
        elapsed_ms = (time.perf_counter() - start) * 1000
    finally:
        set_runtime(None)
    assert result == {"status": "ok"}
    assert elapsed_ms < 200
    assert len(tasks.tasks) == 1 and sender.sent == []
