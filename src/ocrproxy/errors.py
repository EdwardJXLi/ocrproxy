"""Error envelope shared by every failure path.

The body mirrors the OpenAI error shape, which is what upstream-compatible
clients already parse:

    {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        error_type: str,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.code = code
        self.param = param

    def as_body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def error_response(error: APIError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content=error.as_body())


async def api_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, APIError)
    return error_response(exc)


def upstream_message(body: bytes) -> str:
    """Pull a human-readable message out of an upstream error body."""
    text = body.decode("utf-8", "replace").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text[:2000] or "upstream returned an empty error body"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(payload.get("message"), str):
            return payload["message"]
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            parts = [
                item.get("msg", "") if isinstance(item, dict) else str(item)
                for item in detail
            ]
            joined = "; ".join(part for part in parts if part)
            if joined:
                return joined
    return text[:2000]
