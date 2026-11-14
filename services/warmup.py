"""Startup warm-up, so the first lead after a deploy doesn't pay for connection setup.

The first LLM call creates the provider client and its TLS connection, the first knowledge search does the
same for the embedding API and Qdrant, and the first calendar call refreshes the OAuth token. Together that
can add a second or more to the first reply. Each warm-up is best-effort, runs in the background, and never
raises.
"""
import asyncio
import logging
import time

from langchain_core.messages import HumanMessage

from agent.llm import LLMProvider
from agent.services import Knowledge

log = logging.getLogger(__name__)

WARMUP_TIMEOUT_SECONDS = 15.0


async def _timed(name: str, coro) -> tuple[str, bool, int]:
    start = time.perf_counter()
    try:
        await asyncio.wait_for(coro, timeout=WARMUP_TIMEOUT_SECONDS)
        ok = True
    except Exception:  # noqa: BLE001 - often a misconfiguration (bad model id, unsupported setting, quota)
        log.warning("warm-up of %s failed", name, exc_info=True)
        ok = False
    return name, ok, int((time.perf_counter() - start) * 1000)


async def warm_up(llm: LLMProvider | None, knowledge: Knowledge | None, scheduler=None) -> dict[str, bool]:
    jobs = []
    if llm is not None:
        jobs.append(_timed("llm", llm.generate("Reply with the single word OK.", [HumanMessage("ping")])))
    if knowledge is not None:
        jobs.append(_timed("knowledge", knowledge.search("DLD registration fee", "DUBAI", "en")))
    if scheduler is not None and hasattr(scheduler, "warm"):
        jobs.append(_timed("calendar", scheduler.warm()))
    results = await asyncio.gather(*jobs)
    for name, ok, ms in results:
        log.info("warm-up %s: %s in %d ms", name, "ok" if ok else "failed", ms)
    return {name: ok for name, ok, _ in results}
