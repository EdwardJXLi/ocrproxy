"""HTTP surface: Mistral-compatible `/v1/ocr` on top of one vision model."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import APIRouter, FastAPI, Request
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from .config import FETCH_TIMEOUT_SECONDS, MAX_DOCUMENT_BYTES, Settings, get_settings
from .documents import UploadedBytes
from .errors import APIError, api_error_handler
from .pipeline import run_ocr
from .schemas import OCROptions, OCRRequest, OCRResponse
from .upstream import UpstreamClient

logger = logging.getLogger(__name__)

router = APIRouter()

_FORM_OPTION_FIELDS = (
    "model",
    "base_url",
    "pages",
    "include_image_base64",
    "image_limit",
    "image_min_size",
    "table_format",
    "extract_header",
    "extract_footer",
    "document_annotation_prompt",
)
_FORM_JSON_FIELDS = ("document_annotation_format", "bbox_annotation_format")


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/v1/ocr",
    response_model=OCRResponse,
    response_model_exclude_none=False,
    responses={
        400: {"description": "Invalid request or document"},
        401: {"description": "Missing or rejected provider key"},
        413: {"description": "Document or request body too large"},
        415: {"description": "Unsupported document type"},
        502: {"description": "Upstream error or invalid upstream output"},
    },
)
async def create_ocr(request: Request) -> OCRResponse:
    settings: Settings = request.app.state.settings
    api_key, client_key = _api_key(request, settings)
    source, options = await _parse_request(request)
    if options.base_url is not None and not client_key:
        raise APIError(
            400,
            "invalid_request",
            "`base_url` requires a client-supplied provider key, so the configured "
            "fallback key is never sent to a request-selected host.",
            param="base_url",
        )
    return await run_ocr(
        source=source,
        options=options,
        settings=settings,
        upstream=request.app.state.upstream,
        http=request.app.state.http,
        api_key=api_key,
        base_url=options.base_url,
    )


def _api_key(request: Request, settings: Settings) -> tuple[str, bool]:
    header = request.headers.get("authorization", "")
    key = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not key:
        key = (request.headers.get("x-api-key") or "").strip()
    if key:
        return key, True
    if settings.upstream_api_key:
        return settings.upstream_api_key, False
    raise APIError(
        401,
        "missing_api_key",
        "Send your provider key as `Authorization: Bearer <key>`; it is forwarded upstream.",
    )


async def _parse_request(request: Request) -> tuple[Any, OCROptions]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        return await _parse_multipart(request)

    # Base64 inflates the payload by 4/3, plus JSON overhead.
    limit = int(MAX_DOCUMENT_BYTES * 1.4) + 64 * 1024
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise APIError(
            413,
            "document_too_large",
            f"Request body exceeds {limit // (1024 * 1024)} MiB. "
            "Upload the document as multipart instead.",
        )
    raw = await request.body()
    if not raw:
        raise APIError(400, "invalid_request", "Request body is empty.")
    if len(raw) > limit:
        raise APIError(
            413,
            "document_too_large",
            f"Request body exceeds {limit // (1024 * 1024)} MiB. "
            "Upload the document as multipart instead.",
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise APIError(400, "invalid_request", f"Request body is not valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise APIError(400, "invalid_request", "Request body must be a JSON object.")
    try:
        parsed = OCRRequest.model_validate(payload)
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    return parsed.document, _options_from(parsed)


async def _parse_multipart(request: Request) -> tuple[UploadedBytes, OCROptions]:
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile):
        raise APIError(
            400,
            "invalid_request",
            "Multipart requests must include the document in a `file` field.",
        )
    data = await upload.read(MAX_DOCUMENT_BYTES + 1)
    if not data:
        raise APIError(400, "invalid_request", "Uploaded file is empty.")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise APIError(
            413,
            "document_too_large",
            f"Document exceeds the {MAX_DOCUMENT_BYTES // (1024 * 1024)} MiB limit.",
        )

    values: dict[str, Any] = {}
    for name in _FORM_OPTION_FIELDS:
        value = form.get(name)
        if isinstance(value, str) and value.strip():
            values[name] = value.strip()
    for name in _FORM_JSON_FIELDS:
        value = form.get(name)
        if isinstance(value, str) and value.strip():
            try:
                values[name] = json.loads(value)
            except json.JSONDecodeError as exc:
                raise APIError(
                    400,
                    "invalid_request",
                    f"Form field `{name}` is not valid JSON: {exc.msg}.",
                    param=name,
                ) from exc

    try:
        options = OCROptions.model_validate(values)
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    source = UploadedBytes(
        data=data,
        name=upload.filename,
        content_type=upload.content_type,
    )
    return source, options


def _options_from(request: OCRRequest) -> OCROptions:
    return OCROptions(**{name: getattr(request, name) for name in OCROptions.model_fields})


def _validation_error(exc: ValidationError) -> APIError:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "body"
    return APIError(
        400,
        "invalid_request",
        f"`{location}`: {first.get('msg')}",
        param=str(first.get("loc", ("body",))[0]),
    )


def create_app(
    *,
    settings: Settings | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    cfg = settings or get_settings()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = cfg
        app.state.upstream = UpstreamClient(cfg, transport=upstream_transport)
        app.state.http = httpx.AsyncClient(
            timeout=httpx.Timeout(FETCH_TIMEOUT_SECONDS, connect=10.0),
            headers={"user-agent": "ocrproxy/0.1"},
        )
        try:
            yield
        finally:
            await app.state.upstream.aclose()
            await app.state.http.aclose()

    app = FastAPI(
        title="ocrproxy",
        version="0.1.0",
        summary="Mistral-compatible /v1/ocr backed by an OpenAI-compatible vision model",
        lifespan=lifespan,
    )
    app.add_exception_handler(APIError, api_error_handler)
    app.include_router(router)
    return app


app = create_app()
