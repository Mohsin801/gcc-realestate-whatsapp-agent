"""Phase 0 smoke tests: config, redis helpers, app wiring."""
import fakeredis.aioredis
import pytest

from core.config import Settings
from services import redis as redis_svc


def test_settings_defaults_match_spec_decisions():
    s = Settings(_env_file=None)
    assert s.owner_timezone == "Asia/Dubai"
    assert s.working_days == [0, 1, 2, 3, 4]  # Mon–Fri
    assert (s.work_start, s.work_end) == ("09:00", "18:00")
    assert s.slot_duration_minutes == 30  # spec §4: 30-min slots
    assert s.slot_count == 3  # spec §4: next 3 slots


def test_working_days_parsed_from_env(monkeypatch):
    monkeypatch.setenv("WORKING_DAYS", "[6,0,1,2,3]")
    assert Settings(_env_file=None).working_days == [6, 0, 1, 2, 3]


@pytest.fixture
async def fake_redis():
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield r
    await r.aclose()


async def test_claim_wamid_is_idempotent(fake_redis):
    assert await redis_svc.claim_wamid(fake_redis, "wamid.ABC") is True
    assert await redis_svc.claim_wamid(fake_redis, "wamid.ABC") is False
    assert await redis_svc.claim_wamid(fake_redis, "wamid.XYZ") is True


async def test_offered_slots_roundtrip(fake_redis):
    assert await redis_svc.load_offered_slots(fake_redis, "971500000000") == []
    slots = ["2026-11-16T12:00:00+00:00", "2026-11-17T07:00:00+00:00"]
    await redis_svc.save_offered_slots(fake_redis, "971500000000", slots)
    assert await redis_svc.load_offered_slots(fake_redis, "971500000000") == slots


def test_app_routes_registered():
    from main import app

    assert app.url_path_for("health") == "/health"
