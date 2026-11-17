"""Development only: wipe one lead's conversation so their next WhatsApp message starts fresh (see services/reset.py).

    .\\.venv\\Scripts\\python.exe scripts\\reset_lead.py 923001234567            # via the running app
    .\\.venv\\Scripts\\python.exe scripts\\reset_lead.py 923001234567 --direct   # straight to Redis + Postgres

By default it calls ``POST /admin/reset/{phone}`` on the running app with ``ADMIN_TOKEN`` from ``.env``, which
also clears the demo calendar's in-memory state. If the app isn't running it resets Redis and Postgres directly
(the demo calendar's memory went away with the app anyway). The token is never printed.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import get_settings  # noqa: E402
from services.reset import normalize_phone, reset_lead  # noqa: E402


async def reset_direct(phone: str) -> dict:
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    from db.session import dispose_engine, get_sessionmaker, init_engine
    from services.redis import create_redis

    settings = get_settings()
    init_engine(settings.database_url)
    redis = create_redis(settings.redis_url)
    try:
        async with AsyncRedisSaver.from_conn_string(settings.redis_url) as checkpointer:
            result = await reset_lead(phone, redis=redis, sessionmaker=get_sessionmaker(), checkpointer=checkpointer)
    finally:
        await redis.aclose()
        await dispose_engine()
    return result.as_dict()


def reset_via_app(phone: str, url: str) -> dict | None:
    """The app's answer, or None if it isn't reachable."""
    settings = get_settings()
    if not settings.admin_token:
        sys.exit("ADMIN_TOKEN is empty in .env")
    try:
        r = httpx.post(f"{url.rstrip('/')}/admin/reset/{phone}", headers={"X-Admin-Token": settings.admin_token},
                       timeout=40)
    except httpx.ConnectError:
        return None
    if r.status_code == 404:
        sys.exit("The app has no /admin/reset (restart it with the current code, and APP_ENV must be development). "
                 "Or use --direct.")
    if r.status_code != 200:
        sys.exit(f"Reset failed: HTTP {r.status_code} {r.text[:200]}")
    return r.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phone", help="WhatsApp number with country code, e.g. 923001234567")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="the running app (default: %(default)s)")
    parser.add_argument("--direct", action="store_true", help="skip the app and reset Redis + Postgres directly")
    args = parser.parse_args()
    try:
        phone = normalize_phone(args.phone)
    except ValueError as e:
        sys.exit(str(e))

    result = None if args.direct else reset_via_app(phone, args.url)
    if result is None:
        if not args.direct:
            print(f"App not reachable at {args.url}; resetting Redis and Postgres directly.")
        result = asyncio.run(reset_direct(phone))
    print(json.dumps(result, indent=2))
    if result.get("google_events"):
        print(f"Note: {result['google_events']} Google Calendar event(s) were not cancelled; remove them by hand.")


if __name__ == "__main__":
    main()
