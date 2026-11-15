"""Shared helpers: a real two-page PDF and a scripted OpenAI-compatible upstream."""

from __future__ import annotations

import contextlib
import json
from typing import Any, AsyncIterator

import httpx
import pymupdf

from ocrproxy.config import Settings
from ocrproxy.main import create_app

PAGE_MARKDOWN = "# {label}\n\n![](img-0)\n\nTotal: 1,200 EUR"


def make_pdf(pages: int = 2) -> bytes:
    document = pymupdf.open()
    for number in range(pages):
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 120), f"Invoice page {number + 1}", fontsize=24)
    data = document.tobytes()
    document.close()
    return data


class FakeUpstream:
    """Scripted `/v1/chat/completions` handler; records every request body."""

    def __init__(
        self,
        *,
        reject_json_schema: int = 0,
        status: int | None = None,
        error_body: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self.reject_json_schema = reject_json_schema
        self.status = status
        self.error_body = error_body
        self.content_override = content

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(404, json={"error": {"message": "unexpected path"}})
        payload = json.loads(request.content)
        self.requests.append(request)
        self.bodies.append(payload)

        if self.status is not None:
            return httpx.Response(self.status, json=self.error_body or {})
        if self.reject_json_schema and payload.get("response_format", {}).get("type") == "json_schema":
            self.reject_json_schema -= 1
            return httpx.Response(
                400,
                json={"error": {"message": "response_format `json_schema` is unsupported"}},
            )

        content = self.content_override or json.dumps(self._payload_for(payload))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "test-vision",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
        )

    def _payload_for(self, payload: dict[str, Any]) -> dict[str, Any]:
        system = payload["messages"][0]["content"]
        if system.startswith("You transcribe document page images"):
            label = payload["messages"][1]["content"][0]["text"].split(".")[0]
            return {
                "markdown": PAGE_MARKDOWN.format(label=label),
                "images": [{"id": "img-0", "bbox": [100, 100, 500, 400]}],
                "tables": [{"id": "table-0", "content": "| a | b |\n| - | - |\n| 1 | 2 |"}],
                "hyperlinks": ["https://example.com", "https://example.com"],
                "header": "Quarterly report",
                "footer": "Confidential",
            }
        if system.startswith("You extract structured data from document markdown"):
            return {"title": "Invoice", "total": 42}
        return {"kind": "chart"}


def settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "upstream_base_url": "http://upstream.test/v1",
        "upstream_model": "test-vision",
        "upstream_api_key": None,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@contextlib.asynccontextmanager
async def client_for(upstream: FakeUpstream, **overrides: Any) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        settings=settings(**overrides),
        upstream_transport=httpx.MockTransport(upstream.handler),
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


def data_uri(pdf: bytes) -> str:
    import base64

    return "data:application/pdf;base64," + base64.b64encode(pdf).decode("ascii")


def ocr_body(pdf: bytes, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": data_uri(pdf)},
    }
    body.update(overrides)
    return body
