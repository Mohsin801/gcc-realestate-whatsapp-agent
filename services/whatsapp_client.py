"""Meta WhatsApp Cloud API client: send text, mark as read + typing indicator."""
import asyncio
import logging

import httpx

log = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.facebook.com"
MAX_TEXT_LEN = 4096  # WhatsApp text body limit
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class WhatsAppAPIError(Exception):
    pass


class WhatsAppClient:
    def __init__(
        self,
        token: str,
        phone_number_id: str,
        api_version: str = "v23.0",
        http: httpx.AsyncClient | None = None,
        retry_delay: float = 0.5,
    ) -> None:
        self._url = f"{GRAPH_BASE}/{api_version}/{phone_number_id}/messages"
        self._headers = {"Authorization": f"Bearer {token}"}
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0))
        self._retry_delay = retry_delay

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, body: dict) -> dict:
        """POST with one retry on transient failures (network, 429, 5xx)."""
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                resp = await self._http.post(self._url, json=body, headers=self._headers)
                if resp.status_code in _RETRYABLE_STATUS and attempt == 0:
                    last_exc = WhatsAppAPIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                else:
                    if resp.is_error:
                        raise WhatsAppAPIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    return resp.json()
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt == 1:
                    break
            await asyncio.sleep(self._retry_delay)
        raise WhatsAppAPIError(f"WhatsApp API request failed: {last_exc}") from last_exc

    async def send_text(self, to: str, body: str) -> str | None:
        """Send a plain text message. Returns the outbound wamid."""
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": body[:MAX_TEXT_LEN]},
        }
        data = await self._post(payload)
        messages = data.get("messages") or [{}]
        return messages[0].get("id")

    async def mark_read(self, wamid: str, typing: bool = True) -> None:
        """Show blue ticks and (optionally) the 'typing…' indicator. Failures are logged, never raised."""
        payload: dict = {"messaging_product": "whatsapp", "status": "read", "message_id": wamid}
        if typing:
            payload["typing_indicator"] = {"type": "text"}
        try:
            await self._post(payload)
        except Exception:  # noqa: BLE001 - cosmetic only, must not block the reply
            log.warning("mark_read failed", extra={"wamid": wamid}, exc_info=True)
