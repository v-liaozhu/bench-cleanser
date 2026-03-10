"""Async-capable LLM client using CloudGPT Azure OpenAI endpoint.

Uses Azure AD token-based authentication via the cloudgpt module and
supports gpt-5.2 reasoning effort parameter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import os
from typing import Any

import openai
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from bench_cleanser.cache import ResponseCache
from bench_cleanser.models import PipelineConfig

logger = logging.getLogger(__name__)

# Errors that should trigger a retry with exponential back-off.
# InternalServerError (HTTP 500) is included because Azure OpenAI
# returns transient 500s under load.
_RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


def _create_async_client(config: PipelineConfig) -> openai.AsyncAzureOpenAI:
    """Create an AsyncAzureOpenAI client using CloudGPT token provider.

    Uses the cloudgpt module's Azure AD token provider with ``az`` CLI
    authentication.
    """
    # Add the project root to sys.path so cloudgpt.py can be imported
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from cloudgpt import get_openai_token_provider

    token_provider = get_openai_token_provider(
        use_azure_cli=True,
        skip_access_validation=True,
    )

    return openai.AsyncAzureOpenAI(
        api_version=config.llm_api_version,
        azure_endpoint=config.llm_base_url,
        azure_ad_token_provider=token_provider,
        max_retries=0,  # We handle retries ourselves with proper backoff
    )


class LLMClient:
    """Async wrapper around the CloudGPT Azure OpenAI chat-completions endpoint.

    Supports optional disk-based caching via :class:`ResponseCache` and
    automatic retries with exponential back-off on transient failures.
    Uses gpt-5.2 reasoning effort for thorough analysis.
    """

    def __init__(
        self,
        config: PipelineConfig,
        cache: ResponseCache | None = None,
    ) -> None:
        self._client = _create_async_client(config)
        self._model = config.llm_model
        self._max_tokens = config.llm_max_tokens
        self._reasoning_effort = config.llm_reasoning_effort
        self._retry_attempts = config.retry_attempts
        self._retry_delay = config.retry_delay_seconds
        self._cache = cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, system_prompt: str, user_prompt: str) -> str:
        """Build a deterministic cache key for a prompt pair."""
        return ResponseCache.make_key(system_prompt, user_prompt, self._model)

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Execute a single chat-completion request with retries.

        Returns the assistant content string.  Raises ``RuntimeError``
        after all retries are exhausted.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": self._max_tokens,
            "extra_body": {"reasoning_effort": self._reasoning_effort},
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_exc: BaseException | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                return content
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                delay = min(self._retry_delay * (2 ** (attempt - 1)), 60.0)
                logger.warning(
                    "LLM request failed (attempt %d/%d): %s – retrying in %.1fs",
                    attempt,
                    self._retry_attempts,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception as exc:  # noqa: BLE001
                logger.error("LLM request failed with non-retryable error: %s", exc)
                raise

        raise RuntimeError(
            f"LLM request failed after {self._retry_attempts} attempts. "
            f"Last error: {last_exc}"
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Parse *text* as JSON, handling optional markdown fences.

        If direct ``json.loads`` fails the method tries to extract a JSON
        object from a fenced code block (```json ... ```  or  ``` ... ```).
        Returns an empty dict on failure.
        """
        # Fast path
        text = text.strip()
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Try to pull JSON from a markdown code block.
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass

        # Last resort: find the first { ... } block.
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start : brace_end + 1])  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass

        logger.error("Failed to parse JSON from LLM response: %.200s", text)
        return {}

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def query(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str = "text",
    ) -> str:
        """Send a chat-completion request and return the response text.

        Parameters
        ----------
        system_prompt:
            The system message.
        user_prompt:
            The user message.
        response_format:
            ``"text"`` (default) or ``"json_object"``.

        Raises on API failure after retries.
        """
        key = self._cache_key(system_prompt, user_prompt)

        # Check cache
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("Cache hit for key %s", key[:12])
                return cached

        fmt = (
            {"type": "json_object"} if response_format == "json_object" else None
        )
        result = await self._call_api(system_prompt, user_prompt, response_format=fmt)

        # Store in cache
        if self._cache is not None and result:
            self._cache.put(key, result, model=self._model)

        return result

    async def query_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        skip_cache: bool = False,
    ) -> dict[str, Any]:
        """Send a chat-completion request and return the parsed JSON dict.

        The method first attempts to use the API-level
        ``response_format={"type": "json_object"}`` parameter.  If the
        model returns empty content, retries with an explicit JSON
        instruction injected into the system prompt.

        Parameters
        ----------
        skip_cache:
            If ``True``, bypass the cache for this request and make a
            fresh API call.  The result will still be stored in the cache.

        Raises on API failure (no silent fallback to ``{}``).
        """
        key = self._cache_key(system_prompt, user_prompt)

        # Check cache
        if not skip_cache and self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("Cache hit for key %s", key[:12])
                return self._extract_json(cached)

        # Try with native JSON mode first.
        result = await self._call_api(
            system_prompt,
            user_prompt,
            response_format={"type": "json_object"},
        )

        if not result:
            # Fallback: inject instruction into the system prompt.
            fallback_system = system_prompt.rstrip() + "\n\nRespond in valid JSON."
            result = await self._call_api(fallback_system, user_prompt)

        # Store in cache
        if self._cache is not None and result:
            self._cache.put(key, result, model=self._model)

        return self._extract_json(result)

    # ------------------------------------------------------------------
    # Synchronous convenience wrappers
    # ------------------------------------------------------------------

    def query_sync(self, system_prompt: str, user_prompt: str) -> str:
        """Blocking version of :meth:`query`."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self.query(system_prompt, user_prompt)
                ).result()

        return asyncio.run(self.query(system_prompt, user_prompt))

    def query_json_sync(
        self, system_prompt: str, user_prompt: str
    ) -> dict[str, Any]:
        """Blocking version of :meth:`query_json`."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self.query_json(system_prompt, user_prompt)
                ).result()

        return asyncio.run(self.query_json(system_prompt, user_prompt))
