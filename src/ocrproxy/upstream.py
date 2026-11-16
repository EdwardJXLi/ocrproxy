"""Chat-completions client for the single configured OpenAI-compatible upstream.

Structured output is the primary control mechanism: the model must satisfy a
JSON Schema. When the upstream rejects `response_format`, the client degrades
once to `json_object`, then to a schema-in-prompt request, and remembers the
working mode for the rest of the process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import (
    UPSTREAM_MAX_RETRIES,
    UPSTREAM_TIMEOUT_SECONDS,
    Settings,
)
from .errors import APIError, upstream_message

logger = logging.getLogger(__name__)

_BACKOFF_BASE_SECONDS = 0.5
_MAX_BACKOFF_SECONDS = 8.0
_STRUCTURED_OUTPUT_REJECTION = (
    "response_format",
    "response format",
    "json_schema",
    "json schema",
    "structured output",
    "structured outputs",
)


class UpstreamClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.default_base_url = settings.upstream_base_url
        self._modes: dict[str, str] = {}
        self._mode_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            base_url=settings.upstream_base_url,
            timeout=httpx.Timeout(UPSTREAM_TIMEOUT_SECONDS, connect=10.0),
            transport=transport,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete_json(
        self,
        *,
        api_key: str,
        model: str,
        messages: list[dict[str, Any]],
        schema_name: str,
        schema: dict[str, Any],
        strict: bool = True,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        target = base_url or self.default_base_url
        retries = 0
        while True:
            body, mode_used = self._build_body(
                model=model,
                messages=messages,
                schema_name=schema_name,
                schema=schema,
                strict=strict,
                mode=self._mode(target),
            )
            try:
                response = await self._client.post(
                    _chat_completions_url(target),
                    json=body,
                    headers={"authorization": f"Bearer {api_key}"},
                )
            except httpx.TimeoutException as exc:
                retries += 1
                if retries > UPSTREAM_MAX_RETRIES:
                    raise APIError(
                        504,
                        "upstream_timeout",
                        f"Upstream did not answer within {UPSTREAM_TIMEOUT_SECONDS:.0f}s.",
                    ) from exc
                await asyncio.sleep(_backoff(retries, None))
                continue
            except httpx.HTTPError as exc:
                raise APIError(
                    502,
                    "upstream_unreachable",
                    f"Upstream request failed: {exc.__class__.__name__}.",
                ) from exc

            if response.status_code >= 400:
                action = await self._structured_failure(response, mode_used, target)
                if action is not None:
                    continue
                retryable = response.status_code in (408, 409, 429) or response.status_code >= 500
                if retryable and retries < UPSTREAM_MAX_RETRIES:
                    retries += 1
                    await asyncio.sleep(_backoff(retries, response.headers.get("retry-after")))
                    continue
                raise self._error_from_response(response)

            return _extract_json(response, schema_name)

    def _build_body(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        schema_name: str,
        schema: dict[str, Any],
        strict: bool,
        mode: str,
    ) -> tuple[dict[str, Any], str]:
        body: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
        }
        if mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": strict},
            }
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}
            body["messages"].append(
                {
                    "role": "user",
                    "content": "Reply with one JSON object that validates against this JSON Schema:\n"
                    + json.dumps(schema, ensure_ascii=False),
                }
            )
        elif mode == "none":
            body["messages"].append(
                {
                    "role": "user",
                    "content": "Reply with one JSON object only, with no prose and no code fences. "
                    "It must validate against this JSON Schema:\n"
                    + json.dumps(schema, ensure_ascii=False),
                }
            )
        return body, mode

    def _mode(self, base_url: str) -> str:
        return self._modes.get(base_url, "json_schema")

    async def _structured_failure(
        self,
        response: httpx.Response,
        mode_used: str,
        base_url: str,
    ) -> str | None:
        """Detect a rejected structured-output mode.

        Returns "retry" when another request already changed the mode, and
        "degraded" after switching to the next weaker mode. Concurrent pages
        therefore degrade at most once each per step. Modes are tracked per
        base URL, so one provider's limits do not weaken another's requests.
        """
        if response.status_code != 400:
            return None
        text = response.text.lower()
        if not any(marker in text for marker in _STRUCTURED_OUTPUT_REJECTION):
            return None
        async with self._mode_lock:
            current = self._mode(base_url)
            if mode_used != current:
                return "retry"
            next_mode = {"json_schema": "json_object", "json_object": "none"}.get(current)
            if next_mode is None:
                return None
            logger.warning(
                "upstream %s rejected %s structured output; degrading to %s: %s",
                base_url,
                current,
                next_mode,
                upstream_message(response.content),
            )
            self._modes[base_url] = next_mode
            return "degraded"

    @staticmethod
    def _error_from_response(response: httpx.Response) -> APIError:
        message = upstream_message(response.content)
        status = response.status_code if response.status_code < 500 else 502
        if response.status_code in (401, 403):
            return APIError(
                response.status_code,
                "upstream_auth_error",
                f"Upstream rejected the API key: {message}",
            )
        if response.status_code == 404:
            return APIError(404, "upstream_model_not_found", message)
        if response.status_code == 429:
            return APIError(429, "upstream_rate_limited", message, code="rate_limit_exceeded")
        return APIError(status, "upstream_error", message)


def _chat_completions_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = f"{parsed.path.rstrip('/')}/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, ""))


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return max(0.0, min(float(retry_after), 30.0))
        except ValueError:
            pass
    return min(_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), _MAX_BACKOFF_SECONDS)


_CODE_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$")


def _extract_json(response: httpx.Response, schema_name: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise APIError(502, "upstream_invalid_output", "Upstream returned a non-JSON body.") from exc

    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        raise APIError(502, "upstream_invalid_output", "Upstream response contained no choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise APIError(
            502,
            "upstream_invalid_output",
            f"Upstream returned no content for schema {schema_name!r}.",
        )
    return parse_json_object(content)


def parse_json_object(content: str) -> dict[str, Any]:
    text = _CODE_FENCE.sub("", content.strip()).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise APIError(
                502,
                "upstream_invalid_output",
                "Upstream content was not a JSON object.",
            ) from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise APIError(
                502,
                "upstream_invalid_output",
                f"Upstream content was not valid JSON: {exc.msg}.",
            ) from None
    if not isinstance(value, dict):
        raise APIError(502, "upstream_invalid_output", "Upstream JSON was not an object.")
    return value
