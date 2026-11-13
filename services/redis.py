"""Redis client, FastAPI dependency, and small helpers (dedup, locks, offered slots)."""
import json

from fastapi import Request
from redis.asyncio import Redis

from core.config import get_settings


def create_redis(url: str | None = None) -> Redis:
    return Redis.from_url(url or get_settings().redis_url, decode_responses=True)


async def get_redis(request: Request) -> Redis:
    """FastAPI dependency: the shared Redis client created in the app lifespan."""
    return request.app.state.redis


# --- wamid deduplication (spec §5 step 2) ---
async def claim_wamid(redis: Redis, wamid: str) -> bool:
    """Atomically claim a WhatsApp message id. Returns False if it was already seen."""
    ttl = get_settings().whatsapp_dedup_ttl_seconds
    return bool(await redis.set(f"dedup:wamid:{wamid}", "1", nx=True, ex=ttl))


# --- per-phone processing lock ---
def phone_lock(redis: Redis, phone: str):
    """Serialises processing of messages from the same lead."""
    timeout = get_settings().phone_lock_timeout_seconds
    return redis.lock(f"lock:phone:{phone}", timeout=timeout, blocking_timeout=timeout)


# --- offered slots cache (ISO timestamps shown to the lead) ---
async def save_offered_slots(redis: Redis, phone: str, slots_iso: list[str]) -> None:
    ttl = get_settings().offered_slots_ttl_seconds
    await redis.set(f"offered:{phone}", json.dumps(slots_iso), ex=ttl)


async def load_offered_slots(redis: Redis, phone: str) -> list[str]:
    raw = await redis.get(f"offered:{phone}")
    return json.loads(raw) if raw else []
