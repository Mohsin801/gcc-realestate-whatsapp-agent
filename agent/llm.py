"""LLM provider abstraction. Gemini by default; Claude or OpenAI can be selected with LLM_PROVIDER.

The agent only needs two operations, so a provider is anything with:
  * ``extract(system, messages, schema, timeout=None)`` -> an instance of the Pydantic ``schema`` (structured output)
  * ``generate(system, messages, timeout=None)`` -> plain reply text
Each call has its own timeout (``LLM_TIMEOUT_SECONDS``, or a shorter ``timeout`` when the turn's latency budget
is running out); callers treat any exception as "LLM unavailable" and fall back to
rules/templates, so the lead is always answered.
"""
import asyncio
from collections.abc import Sequence
from typing import Protocol, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel

from core.config import Settings

T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    async def extract(self, system: str, messages: Sequence[BaseMessage], schema: type[T],
                      timeout: float | None = None) -> T: ...
    async def generate(self, system: str, messages: Sequence[BaseMessage], timeout: float | None = None) -> str: ...


class LLMConfigError(RuntimeError):
    pass


class LangChainLLM:
    """Adapts any LangChain chat model to ``LLMProvider``."""

    def __init__(self, chat_model: BaseChatModel, timeout_seconds: float):
        self._model = chat_model
        self._timeout = timeout_seconds
        self._structured: dict[type, object] = {}

    def _limit(self, timeout: float | None) -> float:
        return self._timeout if timeout is None else min(timeout, self._timeout)

    async def extract(self, system: str, messages: Sequence[BaseMessage], schema: type[T],
                      timeout: float | None = None) -> T:
        runnable = self._structured.get(schema)
        if runnable is None:
            runnable = self._structured[schema] = self._model.with_structured_output(schema)
        result = await asyncio.wait_for(
            runnable.ainvoke([SystemMessage(system), *messages]), timeout=self._limit(timeout)  # type: ignore[attr-defined]
        )
        if isinstance(result, dict):
            result = schema.model_validate(result)
        if not isinstance(result, schema):
            raise ValueError(f"structured output did not match {schema.__name__}")
        return result

    async def generate(self, system: str, messages: Sequence[BaseMessage], timeout: float | None = None) -> str:
        msg = await asyncio.wait_for(
            self._model.ainvoke([SystemMessage(system), *messages]), timeout=self._limit(timeout)
        )
        return _text_of(msg.content)


def _text_of(content) -> str:
    """Chat models may return a string or a list of content blocks (text, thoughts, ...)."""
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def build_llm(settings: Settings) -> LLMProvider:
    """Create the configured provider. Model ids always come from config, never from code."""
    provider = settings.llm_provider.lower()
    common = {"temperature": settings.llm_temperature}
    if provider == "gemini":
        if not settings.gemini_api_key or not settings.gemini_model:
            raise LLMConfigError("GEMINI_API_KEY and GEMINI_MODEL must be set for LLM_PROVIDER=gemini")
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {}
        if settings.gemini_thinking_level:
            kwargs["thinking_level"] = settings.gemini_thinking_level
        model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # we time-box each call and fall back ourselves
            **common,
            **kwargs,
        )
    elif provider == "anthropic":
        if not settings.anthropic_api_key or not settings.anthropic_model:
            raise LLMConfigError("ANTHROPIC_API_KEY and ANTHROPIC_MODEL must be set for LLM_PROVIDER=anthropic")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:  # optional dependency
            raise LLMConfigError("pip install langchain-anthropic to use LLM_PROVIDER=anthropic") from e
        model = ChatAnthropic(
            model=settings.anthropic_model, api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds, max_retries=0, **common,
        )
    elif provider == "openai":
        if not settings.openai_api_key or not settings.openai_model:
            raise LLMConfigError("OPENAI_API_KEY and OPENAI_MODEL must be set for LLM_PROVIDER=openai")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:  # optional dependency
            raise LLMConfigError("pip install langchain-openai to use LLM_PROVIDER=openai") from e
        model = ChatOpenAI(
            model=settings.openai_model, api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds, max_retries=0, **common,
        )
    else:
        raise LLMConfigError(f"Unknown LLM_PROVIDER {settings.llm_provider!r} (use gemini, anthropic or openai)")
    return LangChainLLM(model, settings.llm_timeout_seconds)
