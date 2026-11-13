"""Google Calendar over its REST API (async httpx), authorised with the owner's OAuth refresh token.

* freeBusy:      POST /calendar/v3/freeBusy
* create event:  POST /calendar/v3/calendars/{id}/events?conferenceDataVersion=1  (Google Meet via createRequest)
* add attendee:  PATCH /calendar/v3/calendars/{id}/events/{eventId}?sendUpdates=all  (Google emails the invite)
"""
import asyncio
import logging
import time
from datetime import datetime
from urllib.parse import quote, urlencode

import httpx

from tools.calendar_providers.base import CalendarError, CalendarNotConnected, CreatedEvent

log = logging.getLogger(__name__)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://www.googleapis.com/calendar/v3"


class GoogleOAuth:
    """OAuth 2.0 web-server flow for the owner's Google account, plus access-token refresh with caching."""

    def __init__(self, http: httpx.AsyncClient, client_id: str, client_secret: str, scopes: list[str]):
        self.http = http
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = scopes
        self._cache: dict[str, tuple[str, float]] = {}  # refresh token -> (access token, expiry monotonic)

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def authorization_url(self, state: str, redirect_uri: str) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.scopes),
            "access_type": "offline",  # we need a refresh token
            "prompt": "consent",  # always return a refresh token, even on re-connect
            "include_granted_scopes": "true",
            "state": state,
        }
        return f"{AUTH_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> dict:
        resp = await self.http.post(TOKEN_URL, data={
            "code": code, "client_id": self.client_id, "client_secret": self.client_secret,
            "redirect_uri": redirect_uri, "grant_type": "authorization_code",
        })
        if resp.status_code != 200:
            raise CalendarError(f"Google token exchange failed: {resp.status_code} {resp.text[:200]}")
        return resp.json()

    async def access_token(self, refresh_token: str) -> str:
        cached = self._cache.get(refresh_token)
        if cached and cached[1] > time.monotonic() + 60:
            return cached[0]
        try:
            resp = await self.http.post(TOKEN_URL, data={
                "client_id": self.client_id, "client_secret": self.client_secret,
                "refresh_token": refresh_token, "grant_type": "refresh_token",
            })
        except httpx.TransportError as e:
            raise CalendarError(f"Google token refresh failed: {e}") from e
        if resp.status_code in (400, 401):
            # invalid_grant: revoked, expired (e.g. 7-day limit for apps in 'Testing'), or wrong client
            raise CalendarNotConnected(f"Google authorisation is no longer valid: {resp.text[:200]}")
        if resp.status_code != 200:
            raise CalendarError(f"Google token refresh failed: {resp.status_code}")
        data = resp.json()
        token = data["access_token"]
        self._cache[refresh_token] = (token, time.monotonic() + int(data.get("expires_in", 3600)))
        return token

    def forget(self, refresh_token: str) -> None:
        self._cache.pop(refresh_token, None)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class GoogleCalendarProvider:
    def __init__(self, oauth: GoogleOAuth, refresh_token: str, calendar_id: str = "primary",
                 retry_delay: float = 0.3):
        self.oauth = oauth
        self.refresh_token = refresh_token
        self.calendar_id = calendar_id
        self.retry_delay = retry_delay

    async def _request(self, method: str, url: str, **kwargs) -> dict:
        for attempt in (1, 2):
            token = await self.oauth.access_token(self.refresh_token)
            try:
                resp = await self.oauth.http.request(
                    method, url, headers={"Authorization": f"Bearer {token}"}, **kwargs
                )
            except httpx.TransportError as e:
                if attempt == 2:
                    raise CalendarError(f"Google Calendar unreachable: {e}") from e
                await asyncio.sleep(self.retry_delay)
                continue
            if resp.status_code == 401 and attempt == 1:  # access token revoked/expired early
                self.oauth.forget(self.refresh_token)
                continue
            if (resp.status_code == 429 or resp.status_code >= 500) and attempt == 1:
                await asyncio.sleep(self.retry_delay)
                continue
            if resp.status_code >= 400:
                raise CalendarError(f"Google Calendar {method} failed: {resp.status_code} {resp.text[:300]}")
            return resp.json()
        raise CalendarError("Google Calendar request failed")  # pragma: no cover

    def _events_url(self, event_id: str | None = None) -> str:
        url = f"{API}/calendars/{quote(self.calendar_id, safe='')}/events"
        return f"{url}/{quote(event_id, safe='')}" if event_id else url

    async def free_busy(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        data = await self._request("POST", f"{API}/freeBusy", json={
            "timeMin": start.isoformat(), "timeMax": end.isoformat(), "items": [{"id": self.calendar_id}],
        })
        cal = (data.get("calendars") or {}).get(self.calendar_id) or {}
        if cal.get("errors"):
            raise CalendarError(f"freeBusy error for calendar: {cal['errors']}")
        return [(_parse_dt(b["start"]), _parse_dt(b["end"])) for b in cal.get("busy", [])]

    async def insert_event(
        self, *, summary: str, description: str, start: datetime, end: datetime, timezone: str, request_id: str
    ) -> CreatedEvent:
        body = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start.isoformat(), "timeZone": timezone},
            "end": {"dateTime": end.isoformat(), "timeZone": timezone},
            "conferenceData": {
                "createRequest": {"requestId": request_id, "conferenceSolutionKey": {"type": "hangoutsMeet"}}
            },
        }
        data = await self._request(
            "POST", self._events_url(), params={"conferenceDataVersion": 1, "sendUpdates": "none"}, json=body
        )
        meet = data.get("hangoutLink")
        if not meet:
            for ep in (data.get("conferenceData") or {}).get("entryPoints") or []:
                if ep.get("entryPointType") == "video":
                    meet = ep.get("uri")
                    break
        return CreatedEvent(event_id=data["id"], meet_link=meet)

    async def set_attendee(self, event_id: str, email: str) -> None:
        await self._request(
            "PATCH", self._events_url(event_id), params={"sendUpdates": "all"},
            json={"attendees": [{"email": email}]},
        )
