"""Shared test fixtures: fake Redis, in-memory SQLite DB, recording WhatsApp sender, test settings."""
import hashlib
import hmac
import json

import fakeredis.aioredis
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.config import get_settings
from db.models import Base

TEST_APP_SECRET = "test-app-secret"
TEST_VERIFY_TOKEN = "verify-me"
TEST_PHONE_NUMBER_ID = "1234567890"


@pytest.fixture(autouse=True)
def test_settings(monkeypatch):
    """Deterministic settings for every test (ignores any local .env)."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", TEST_APP_SECRET)
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", TEST_VERIFY_TOKEN)
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", TEST_PHONE_NUMBER_ID)
    monkeypatch.chdir(__import__("pathlib").Path(__file__).parent)  # no .env here
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def fake_redis():
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest.fixture
async def sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


class RecordingSender:
    """Stand-in for WhatsAppClient that records calls."""

    def __init__(self, fail_send: bool = False):
        self.sent: list[tuple[str, str]] = []
        self.read: list[str] = []
        self.fail_send = fail_send

    async def send_text(self, to: str, body: str) -> str | None:
        if self.fail_send:
            raise RuntimeError("send failed")
        self.sent.append((to, body))
        return f"wamid.out.{len(self.sent)}"

    async def mark_read(self, wamid: str, typing: bool = True) -> None:
        self.read.append(wamid)


@pytest.fixture
def sender():
    return RecordingSender()


def make_text_payload(
    text: str,
    wamid: str = "wamid.TEST1",
    phone: str = "971501234567",
    name: str = "Ahmed",
    msg_type: str = "text",
) -> dict:
    message: dict = {"from": phone, "id": wamid, "timestamp": "1763000000", "type": msg_type}
    if msg_type == "text":
        message["text"] = {"body": text}
    elif msg_type == "audio":
        message["audio"] = {"id": "media-1", "mime_type": "audio/ogg"}
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "971400000000", "phone_number_id": TEST_PHONE_NUMBER_ID},
                            "contacts": [{"profile": {"name": name}, "wa_id": phone}],
                            "messages": [message],
                        },
                    }
                ],
            }
        ],
    }


def sign(body: bytes, secret: str = TEST_APP_SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def encode(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode()
