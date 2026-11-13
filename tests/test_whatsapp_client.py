"""services/whatsapp_client.py against a mocked Meta Graph API."""
import json

import httpx
import pytest
import respx

from services.whatsapp_client import WhatsAppAPIError, WhatsAppClient

URL = "https://graph.facebook.com/v23.0/PNID/messages"


@pytest.fixture
async def wa():
    c = WhatsAppClient(token="TOKEN", phone_number_id="PNID", api_version="v23.0", retry_delay=0)
    yield c
    await c.aclose()


@respx.mock
async def test_send_text_payload_and_auth(wa):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"messages": [{"id": "wamid.OUT"}]}))
    assert await wa.send_text("971501234567", "Hello") == "wamid.OUT"
    req = route.calls.last.request
    assert req.headers["Authorization"] == "Bearer TOKEN"
    assert json.loads(req.content) == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": "971501234567",
        "type": "text",
        "text": {"preview_url": False, "body": "Hello"},
    }


@respx.mock
async def test_mark_read_sends_typing_indicator(wa):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"success": True}))
    await wa.mark_read("wamid.IN")
    assert json.loads(route.calls.last.request.content) == {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": "wamid.IN",
        "typing_indicator": {"type": "text"},
    }


@respx.mock
async def test_retries_once_on_5xx(wa):
    route = respx.post(URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"messages": [{"id": "wamid.2"}]})]
    )
    assert await wa.send_text("1", "x") == "wamid.2"
    assert route.call_count == 2


@respx.mock
async def test_raises_after_retry_exhausted(wa):
    respx.post(URL).mock(return_value=httpx.Response(500))
    with pytest.raises(WhatsAppAPIError):
        await wa.send_text("1", "x")


@respx.mock
async def test_no_retry_on_4xx(wa):
    route = respx.post(URL).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(WhatsAppAPIError):
        await wa.send_text("1", "x")
    assert route.call_count == 1


@respx.mock
async def test_mark_read_failure_is_swallowed(wa):
    respx.post(URL).mock(return_value=httpx.Response(400))
    await wa.mark_read("wamid.X")  # must not raise
