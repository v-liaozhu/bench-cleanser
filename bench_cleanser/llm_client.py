"""Async-capable LLM client using CloudGPT Azure OpenAI endpoint.

Uses Azure AD token-based authentication via the cloudgpt module and
supports gpt-5.4 reasoning effort parameter.

All structured LLM calls use ``response_format={"type": "json_schema", ...}``
with strict Pydantic schemas — no regex JSON extraction, no silent fallbacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, TypeVar

import openai
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from bench_cleanser.cache import ResponseCache
from bench_cleanser.models import PipelineConfig

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

# Errors that should trigger a retry with exponential back-off.
# InternalServerError (HTTP 500) is included because Azure OpenAI
# returns transient 500s under load.
# BadRequestError (HTTP 400) is included because CloudGPT returns transient
# 400s ("unsupported operation") during model rollouts and capacity shifts.
# We also catch Azure credential errors (token expiry, az CLI hiccups)
# which are transient and resolve on retry.
_RETRYABLE_ERRORS: tuple[type[BaseException], ...] = [
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
]

# Attempt to include Azure credential errors so token-refresh
# failures are retried rather than treated as fatal.
try:
    from azure.identity import CredentialUnavailableError
    _RETRYABLE_ERRORS.append(CredentialUnavailableError)
except ImportError:
    pass
try:
    from azure.core.exceptions import ClientAuthenticationError
    _RETRYABLE_ERRORS.append(ClientAuthenticationError)
except ImportError:
    pass

_RETRYABLE_ERRORS = tuple(_RETRYABLE_ERRORS)


def _create_async_client(
    config: PipelineConfig,
) -> tuple[openai.AsyncAzureOpenAI, callable]:
    """Create an AsyncAzureOpenAI client using CloudGPT token provider.

    Uses the cloudgpt module's Azure AD token provider with ``az`` CLI
    authentication.  Wraps the provider with a caching layer so that
    ``az`` is not invoked on every single API call — the token is reused
    until it is close to expiry.

    Returns ``(client, invalidate_token)`` where *invalidate_token* is a
    callable that clears the cached Azure AD token, forcing the next API
    call to acquire a fresh one (useful on 401 errors).
    """
    import time

    from bench_cleanser._internal.cloudgpt import get_openai_token_provider

    raw_provider = get_openai_token_provider(
        use_azure_cli=True,
        skip_access_validation=True,
    )

    # Cache the token so az CLI is invoked at most once per token lifetime.
    # Azure AD tokens typically live 60-90 minutes; we refresh 5 min early.
    _cached_token: str | None = None
    _token_acquired_at: float = 0.0
    _TOKEN_LIFETIME = 50 * 60  # refresh every 50 min (conservative)

    def caching_token_provider() -> str:
        nonlocal _cached_token, _token_acquired_at
        now = time.monotonic()
        if _cached_token is not None and (now - _token_acquired_at) < _TOKEN_LIFETIME:
            return _cached_token
        logger.info("Acquiring fresh Azure AD token (cached token expired or missing)")
        _cached_token = raw_provider()
        _token_acquired_at = now
        return _cached_token

    def invalidate_token() -> None:
        nonlocal _cached_token, _token_acquired_at
        logger.info("Invalidating cached Azure AD token (401 received)")
        _cached_token = None
        _token_acquired_at = 0.0

    client = openai.AsyncAzureOpenAI(
        api_version=config.llm_api_version,
        azure_endpoint=config.llm_base_url,
        azure_ad_token_provider=caching_token_provider,
        max_retries=0,  # We handle retries ourselves with proper backoff
        timeout=None,   # No timeout — let long reasoning_effort=high calls finish
    )

    return client, invalidate_token


class LLMClient:
    """Async wrapper around the CloudGPT Azure OpenAI chat-completions endpoint.

    Supports optional disk-based caching via :class:`ResponseCache` and
    automatic retries with exponential back-off on transient failures.
    Uses gpt-5.4 reasoning effort for thorough analysis.
    """

    def __init__(
        self,
        config: PipelineConfig,
        cache: ResponseCache | None = None,
    ) -> None:
        self._client, self._invalidate_token = _create_async_client(config)
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
            "timeout": None,  # No per-request timeout — reasoning_effort=high takes long
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        last_exc: BaseException | None = None
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                return content
            except APIConnectionError as exc:
                # Pure network connectivity failure — could be ISP dropped,
                # DNS flake, or Azure edge down. Wait it out indefinitely.
                last_exc = exc
                delay = min(5.0 * (2 ** min(attempt - 1, 4)), 60.0)
                logger.warning(
                    "LLM connectivity error (attempt %d, will wait forever for network): %s — "
                    "retrying in %.1fs",
                    attempt, exc, delay,
                )
                await asyncio.sleep(delay)
                # No retry cap for connectivity errors — loop forever
                continue
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                # On 401, invalidate the cached token so the next retry
                # acquires a fresh one from az CLI.
                if isinstance(exc, AuthenticationError):
                    self._invalidate_token()
                if attempt >= self._retry_attempts:
                    break
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

    async def query_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        *,
        skip_cache: bool = False,
    ) -> T:
        """Send a chat-completion with Pydantic schema enforcement.

        Appends the JSON schema to the system prompt and uses
        ``response_format={"type": "json_object"}`` for API-level JSON
        guarantee. The response is then validated against the Pydantic
        model. No regex extraction, no silent fallbacks.

        Parameters
        ----------
        system_prompt:
            The system message (role, rules, schema description — NO data).
        user_prompt:
            The user message (all data, context, evidence).
        response_model:
            A Pydantic BaseModel subclass defining the expected output.
        skip_cache:
            If ``True``, bypass cache for this request.

        Returns
        -------
        An instance of *response_model* parsed from the LLM response.

        Raises
        ------
        RuntimeError
            After retries are exhausted or if schema validation fails
            on all attempts.
        """
        # Append schema to system prompt so the LLM knows the exact structure
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        augmented_system = (
            system_prompt.rstrip()
            + "\n\n## OUTPUT FORMAT\n\n"
            + "You MUST respond with valid JSON conforming EXACTLY to this schema. "
            + "Do NOT wrap in markdown fences. Do NOT include extra keys.\n\n"
            + f"```json\n{schema_json}\n```"
        )

        key = self._cache_key(augmented_system, user_prompt)

        # Check cache
        if not skip_cache and self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("Cache hit (structured) for key %s", key[:12])
                try:
                    return response_model.model_validate_json(cached)
                except ValidationError:
                    logger.info(
                        "Cached response failed schema validation for %s, re-querying",
                        response_model.__name__,
                    )

        max_validation_attempts = 2
        last_error: Exception | None = None

        for attempt in range(1, max_validation_attempts + 1):
            result = await self._call_api(
                augmented_system, user_prompt,
                response_format={"type": "json_object"},
            )

            if not result:
                raise RuntimeError(
                    f"LLM returned empty response for {response_model.__name__}"
                )

            # Store in cache (raw response) before validation
            if self._cache is not None:
                self._cache.put(key, result, model=self._model)

            try:
                return response_model.model_validate_json(result)
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "Schema validation failed for %s (attempt %d/%d): %s",
                    response_model.__name__, attempt, max_validation_attempts, exc,
                )
                if attempt < max_validation_attempts:
                    if self._cache is not None:
                        self._cache.delete(key)

        raise RuntimeError(
            f"Schema validation failed for {response_model.__name__} "
            f"after {max_validation_attempts} attempts: {last_error}"
        )

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
