"""Google Calendar REST provider + OAuth routes, against mocked Google endpoints (respx)."""
import json
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import respx
from sqlalchemy import select

from core.config import get_settings
from core.crypto import TokenCipher, generate_key
from db.models import Owner
from main import app
from tools.calendar_providers.base import CalendarError, CalendarNotConnected
from tools.calendar_providers.google import API, TOKEN_URL, GoogleCalendarProvider, GoogleOAuth

SCOPES = ["https://www.googleapis.com/auth/calendar.events", "https://www.googleapis.com/auth/calendar.freebusy"]
EVENTS = f"{API}/calendars/primary/events"


@pytest.fixture
async def oauth():
    async with httpx.AsyncClient() as http:
        yield GoogleOAuth(http, "cid", "secret", SCOPES)


@pytest.fixture
def provider(oauth):
    return GoogleCalendarProvider(oauth, "refresh-1", "primary", retry_delay=0)


def _token_route():
    return respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={"access_token": "at-1", "expires_in": 3599}))


# --- provider -------------------------------------------------------------------------------------
@respx.mock
async def test_free_busy_parses_intervals_and_caches_access_token(provider):
    token = _token_route()
    fb = respx.post(f"{API}/freeBusy").mock(return_value=httpx.Response(200, json={
        "calendars": {"primary": {"busy": [{"start": "2026-11-17T05:00:00Z", "end": "2026-11-17T06:00:00Z"}]}}
    }))
    start, end = datetime(2026, 11, 17, 4, tzinfo=UTC), datetime(2026, 11, 17, 8, tzinfo=UTC)
    busy = await provider.free_busy(start, end)
    await provider.free_busy(start, end)
    assert busy == [(datetime(2026, 11, 17, 5, tzinfo=UTC), datetime(2026, 11, 17, 6, tzinfo=UTC))]
    assert token.call_count == 1  # cached
    req = fb.calls.last.request
    assert req.headers["Authorization"] == "Bearer at-1"
    assert json.loads(req.content) == {"timeMin": start.isoformat(), "timeMax": end.isoformat(),
                                       "items": [{"id": "primary"}]}
    refresh = parse_qs(token.calls.last.request.content.decode())
    assert refresh["grant_type"] == ["refresh_token"] and refresh["refresh_token"] == ["refresh-1"]


@respx.mock
async def test_insert_event_requests_google_meet(provider):
    _token_route()
    route = respx.post(EVENTS).mock(return_value=httpx.Response(200, json={
        "id": "evt-123", "hangoutLink": "https://meet.google.com/abc-defg-hij"}))
    start = datetime.fromisoformat("2026-11-17T09:00:00+04:00")
    end = datetime.fromisoformat("2026-11-17T09:30:00+04:00")
    created = await provider.insert_event(summary="S", description="D", start=start, end=end,
                                          timezone="Asia/Dubai", request_id="req-1")
    assert (created.event_id, created.meet_link) == ("evt-123", "https://meet.google.com/abc-defg-hij")
    req = route.calls.last.request
    assert req.url.params["conferenceDataVersion"] == "1"
    body = json.loads(req.content)
    assert body["start"] == {"dateTime": "2026-11-17T09:00:00+04:00", "timeZone": "Asia/Dubai"}
    assert body["conferenceData"]["createRequest"] == {
        "requestId": "req-1", "conferenceSolutionKey": {"type": "hangoutsMeet"}}


@respx.mock
async def test_meet_link_from_entry_points(provider):
    _token_route()
    respx.post(EVENTS).mock(return_value=httpx.Response(200, json={"id": "e", "conferenceData": {
        "entryPoints": [{"entryPointType": "phone", "uri": "tel:+1"},
                        {"entryPointType": "video", "uri": "https://meet.google.com/xyz"}]}}))
    t = datetime(2026, 11, 17, 5, tzinfo=UTC)
    created = await provider.insert_event(summary="S", description="D", start=t, end=t, timezone="UTC",
                                          request_id="r")
    assert created.meet_link == "https://meet.google.com/xyz"


@respx.mock
async def test_set_attendee_sends_invite(provider):
    _token_route()
    route = respx.patch(f"{EVENTS}/evt-123").mock(return_value=httpx.Response(200, json={"id": "evt-123"}))
    await provider.set_attendee("evt-123", "lead@example.com")
    req = route.calls.last.request
    assert req.url.params["sendUpdates"] == "all"
    assert json.loads(req.content) == {"attendees": [{"email": "lead@example.com"}]}


@respx.mock
async def test_revoked_refresh_token_means_not_connected(provider):
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(400, json={"error": "invalid_grant"}))
    with pytest.raises(CalendarNotConnected):
        await provider.free_busy(datetime(2026, 11, 17, tzinfo=UTC), datetime(2026, 11, 18, tzinfo=UTC))


@respx.mock
async def test_401_refreshes_token_and_retries(provider):
    token = respx.post(TOKEN_URL).mock(side_effect=[
        httpx.Response(200, json={"access_token": "old", "expires_in": 3599}),
        httpx.Response(200, json={"access_token": "new", "expires_in": 3599}),
    ])
    fb = respx.post(f"{API}/freeBusy").mock(side_effect=[
        httpx.Response(401), httpx.Response(200, json={"calendars": {"primary": {"busy": []}}})])
    assert await provider.free_busy(datetime(2026, 11, 17, tzinfo=UTC), datetime(2026, 11, 18, tzinfo=UTC)) == []
    assert token.call_count == 2 and fb.calls.last.request.headers["Authorization"] == "Bearer new"


@respx.mock
async def test_5xx_retried_once_then_error(provider):
    _token_route()
    respx.post(f"{API}/freeBusy").mock(return_value=httpx.Response(503))
    with pytest.raises(CalendarError):
        await provider.free_busy(datetime(2026, 11, 17, tzinfo=UTC), datetime(2026, 11, 18, tzinfo=UTC))


@respx.mock
async def test_freebusy_calendar_error_is_raised(provider):
    _token_route()
    respx.post(f"{API}/freeBusy").mock(return_value=httpx.Response(200, json={
        "calendars": {"primary": {"errors": [{"reason": "notFound"}]}}}))
    with pytest.raises(CalendarError):
        await provider.free_busy(datetime(2026, 11, 17, tzinfo=UTC), datetime(2026, 11, 18, tzinfo=UTC))


def test_authorization_url_asks_for_offline_refresh_token():
    url = GoogleOAuth(httpx.AsyncClient(), "cid", "secret", SCOPES).authorization_url("st", "https://x/cb")
    q = parse_qs(urlparse(url).query)
    assert q["access_type"] == ["offline"] and q["prompt"] == ["consent"] and q["state"] == ["st"]
    assert q["scope"] == [" ".join(SCOPES)] and q["redirect_uri"] == ["https://x/cb"]


def test_token_cipher_roundtrip_and_bad_key():
    c = TokenCipher(generate_key())
    assert c.decrypt(c.encrypt("refresh-xyz")) == "refresh-xyz"
    with pytest.raises(ValueError):
        TokenCipher("not-a-key")
    with pytest.raises(ValueError):
        TokenCipher(generate_key()).decrypt(c.encrypt("x"))


# --- OAuth routes ---------------------------------------------------------------------------------
@pytest.fixture
async def client(monkeypatch, fake_redis, sessionmaker):
    key = generate_key()
    monkeypatch.setenv("ADMIN_TOKEN", "admin-secret")
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", key)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://agent.example.com")
    get_settings.cache_clear()
    http = httpx.AsyncClient()
    app.state.redis = fake_redis
    app.state.sessionmaker = sessionmaker
    app.state.google_oauth = GoogleOAuth(http, "cid", "secret", get_settings().google_oauth_scopes)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        c.key = key
        yield c
    await http.aclose()
    get_settings.cache_clear()


async def test_start_requires_admin_token(client, monkeypatch):
    assert (await client.get("/auth/google/start")).status_code == 403
    assert (await client.get("/auth/google/start", params={"token": "wrong"})).status_code == 403
    monkeypatch.setenv("ADMIN_TOKEN", "")
    get_settings.cache_clear()
    assert (await client.get("/auth/google/start", params={"token": ""})).status_code == 403


async def test_start_redirects_to_google_with_state(client, fake_redis):
    r = await client.get("/auth/google/start", params={"token": "admin-secret"})
    assert r.status_code == 302
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert r.headers["location"].startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert q["redirect_uri"] == ["https://agent.example.com/auth/google/callback"]
    assert await fake_redis.get(f"oauth_state:{q['state'][0]}") == "1"


@respx.mock
async def test_callback_stores_encrypted_refresh_token(client, fake_redis, sessionmaker):
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={
        "access_token": "at", "refresh_token": "refresh-xyz", "expires_in": 3599, "scope": " ".join(SCOPES)}))
    await fake_redis.set("oauth_state:s1", "1")
    r = await client.get("/auth/google/callback", params={"state": "s1", "code": "c1"})
    assert r.status_code == 200 and "connected" in r.text
    async with sessionmaker() as s:
        owner = await s.scalar(select(Owner))
    assert owner.id == "default" and owner.timezone == "Asia/Dubai"
    assert owner.working_hours == {"days": [0, 1, 2, 3, 4], "start": "09:00", "end": "18:00"}
    assert "refresh-xyz" not in owner.google_refresh_token_enc
    assert TokenCipher(client.key).decrypt(owner.google_refresh_token_enc) == "refresh-xyz"
    # state is single-use
    assert (await client.get("/auth/google/callback", params={"state": "s1", "code": "c1"})).status_code == 400


async def test_callback_rejects_unknown_state(client, sessionmaker):
    r = await client.get("/auth/google/callback", params={"state": "forged", "code": "c"})
    assert r.status_code == 400
    async with sessionmaker() as s:
        assert await s.scalar(select(Owner)) is None


@respx.mock
async def test_callback_requires_all_scopes(client, fake_redis, sessionmaker):
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={
        "access_token": "at", "refresh_token": "r", "scope": SCOPES[0]}))
    await fake_redis.set("oauth_state:s2", "1")
    assert (await client.get("/auth/google/callback", params={"state": "s2", "code": "c"})).status_code == 400
    async with sessionmaker() as s:
        assert await s.scalar(select(Owner)) is None
