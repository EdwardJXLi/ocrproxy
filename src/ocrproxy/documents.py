"""Document ingestion: resolve request input to bytes, then to page images."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import ipaddress
import re
from dataclasses import dataclass
from typing import Sequence
from urllib.parse import urljoin, urlsplit

import httpx
import pymupdf
from PIL import Image, UnidentifiedImageError

from .config import (
    MAX_DOCUMENT_BYTES,
    MAX_PAGES,
    PAGE_DPI,
    PAGE_IMAGE_QUALITY,
)
from .errors import APIError
from .schemas import DocumentInput, DocumentURLChunk, FileChunk, ImageURLChunk

_DATA_URI = re.compile(r"^data:(?P<mime>[^;,]*)(?P<params>;[^,]*)?,(?P<data>.*)$", re.S)
_NORMALIZED_BBOX_SIZE = 1000

_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"BM", "image/bmp"),
)

_SUPPORTED_IMAGE_MIMES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
}


@dataclass(frozen=True)
class UploadedBytes:
    """A document that arrived as a raw upload rather than a URL."""

    data: bytes
    name: str | None = None
    content_type: str | None = None


@dataclass(frozen=True)
class LoadedDocument:
    data: bytes
    mime: str
    name: str | None = None


@dataclass(frozen=True)
class RenderedPage:
    index: int
    data: bytes
    mime: str
    width: int
    height: int
    dpi: int


def parse_page_spec(spec: str | list[int] | None, *, limit: int) -> list[int] | None:
    """Parse Mistral's `pages` grammar: "0,2-4" or [0, 2, 3, 4]."""
    if spec is None:
        return None
    if isinstance(spec, str):
        indices: list[int] = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                bounds = chunk.split("-")
                if len(bounds) != 2:
                    raise APIError(400, "invalid_request", f"Invalid page range {chunk!r}.")
                start, end = (_parse_page_number(part, chunk) for part in bounds)
                if end < start:
                    raise APIError(400, "invalid_request", f"Invalid page range {chunk!r}.")
                if end - start + 1 > limit:
                    raise APIError(
                        400,
                        "invalid_request",
                        f"Page range {chunk!r} exceeds the {limit} page limit.",
                    )
                indices.extend(range(start, end + 1))
            else:
                indices.append(_parse_page_number(chunk, chunk))
    else:
        indices = [int(value) for value in spec]

    if any(index < 0 for index in indices):
        raise APIError(400, "invalid_request", "Page indices start at 0 and cannot be negative.")

    deduped: list[int] = []
    for index in indices:
        if index not in deduped:
            deduped.append(index)
    if len(deduped) > limit:
        raise APIError(
            400,
            "invalid_request",
            f"Requested {len(deduped)} pages, limit is {limit}. Raise OCR_MAX_PAGES to allow more.",
        )
    return deduped


def _parse_page_number(value: str, context: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise APIError(400, "invalid_request", f"Invalid page spec {context!r}.") from None


async def load_document(
    source: DocumentInput | UploadedBytes,
    *,
    http: httpx.AsyncClient,
) -> LoadedDocument:
    max_bytes = MAX_DOCUMENT_BYTES

    if isinstance(source, UploadedBytes):
        _check_size(len(source.data), max_bytes)
        return LoadedDocument(
            data=source.data,
            mime=sniff_mime(source.data, source.content_type),
            name=source.name,
        )

    if isinstance(source, FileChunk):
        raise APIError(
            400,
            "unsupported_document",
            "`file` chunks require Mistral's file store. Send a `document_url` "
            "(public URL or data URI) or upload the file as multipart/form-data.",
        )

    if isinstance(source, DocumentURLChunk):
        url = source.document_url
        name = source.document_name
    elif isinstance(source, ImageURLChunk):
        url = source.image_url if isinstance(source.image_url, str) else source.image_url.url
        name = None
    else:  # pragma: no cover - discriminated union is exhaustive
        raise APIError(400, "invalid_request", "Unsupported document chunk.")

    if url.startswith("data:"):
        data, declared_mime = _decode_data_uri(url)
        _check_size(len(data), max_bytes)
        return LoadedDocument(data=data, mime=sniff_mime(data, declared_mime), name=name)

    if not url.startswith(("http://", "https://")):
        raise APIError(
            400,
            "unsupported_document",
            "`document_url` must be an http(s) URL or a data URI.",
        )

    data = await _fetch(url, http=http, max_bytes=max_bytes)
    return LoadedDocument(data=data, mime=sniff_mime(data, None), name=name)


_MAX_REDIRECTS = 4


async def _fetch(url: str, *, http: httpx.AsyncClient, max_bytes: int) -> bytes:
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        await _ensure_fetchable(current)
        outcome = await _fetch_once(current, http=http, max_bytes=max_bytes)
        if isinstance(outcome, bytes):
            return outcome
        current = outcome
    raise APIError(400, "document_fetch_failed", "Document URL redirected too many times.")


async def _fetch_once(
    url: str,
    *,
    http: httpx.AsyncClient,
    max_bytes: int,
) -> bytes | str:
    """Return the body, or the next URL when the target redirects."""
    try:
        async with http.stream("GET", url) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise APIError(400, "document_fetch_failed", "Redirect without a location.")
                return urljoin(url, location)
            if response.status_code >= 400:
                raise APIError(
                    400,
                    "document_fetch_failed",
                    f"Could not fetch document: upstream returned {response.status_code}.",
                )
            declared = response.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                raise _too_large(max_bytes)
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise _too_large(max_bytes)
                chunks.append(chunk)
            return b"".join(chunks)
    except httpx.HTTPError as exc:
        raise APIError(
            400,
            "document_fetch_failed",
            f"Could not fetch document: {exc.__class__.__name__}.",
        ) from exc


async def _ensure_fetchable(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise APIError(
            400,
            "document_fetch_failed",
            "Document URL must use http or https.",
        )
    host = parsed.hostname
    if not host:
        raise APIError(400, "document_fetch_failed", "Document URL has no host.")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError as exc:
        raise APIError(
            400,
            "document_fetch_failed",
            f"Could not resolve {host!r}.",
        ) from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            raise APIError(
                400,
                "document_url_blocked",
                f"Refusing to fetch {host!r}: it resolves to the non-public address {address}.",
            )


def _too_large(max_bytes: int) -> APIError:
    return APIError(
        413,
        "document_too_large",
        f"Document exceeds the {max_bytes // (1024 * 1024)} MiB limit.",
    )


def _check_size(size: int, max_bytes: int) -> None:
    if size > max_bytes:
        raise _too_large(max_bytes)
    if size == 0:
        raise APIError(400, "invalid_request", "Document is empty.")


def _decode_data_uri(uri: str) -> tuple[bytes, str | None]:
    match = _DATA_URI.match(uri)
    if not match:
        raise APIError(400, "invalid_request", "Malformed data URI in document URL.")
    declared_mime = match.group("mime") or None
    params = match.group("params") or ""
    payload = match.group("data")
    if "base64" in params:
        try:
            data = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise APIError(400, "invalid_request", "Data URI payload is not valid base64.") from exc
    else:
        from urllib.parse import unquote_to_bytes

        data = unquote_to_bytes(payload)
    return data, declared_mime


def sniff_mime(data: bytes, declared: str | None) -> str:
    """Trust magic bytes; fall back to the declared content type."""
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data.lstrip()[:200].lower().startswith((b"<?xml", b"<svg")):
        return "image/svg+xml"
    if declared:
        return declared.split(";")[0].strip().lower()
    return "application/octet-stream"


def document_kind(mime: str) -> str:
    if mime == "application/pdf":
        return "pdf"
    if mime in _SUPPORTED_IMAGE_MIMES:
        return "image"
    raise APIError(
        415,
        "unsupported_document",
        f"Unsupported document type {mime!r}. Send a PDF or a "
        f"{', '.join(sorted(_SUPPORTED_IMAGE_MIMES))} image.",
    )


async def render_pages(
    document: LoadedDocument,
    *,
    requested: list[int] | None,
) -> list[RenderedPage]:
    kind = document_kind(document.mime)
    if kind == "pdf":
        return await asyncio.to_thread(_render_pdf, document.data, requested)
    return _render_image(document, requested)


def _render_pdf(data: bytes, requested: list[int] | None) -> list[RenderedPage]:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except pymupdf.FileDataError as exc:
        raise APIError(400, "invalid_document", "PDF could not be parsed.") from exc

    with doc:
        if doc.needs_pass:
            raise APIError(
                400,
                "unsupported_document",
                "Password-protected PDFs are not supported.",
            )
        count = doc.page_count
        if count == 0:
            raise APIError(400, "invalid_document", "PDF has no pages.")
        indices = _select_pages(requested, count)
        pages: list[RenderedPage] = []
        for index in indices:
            pixmap = doc.load_page(index).get_pixmap(dpi=PAGE_DPI, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            payload, mime, width, height = encode_image(image)
            pages.append(
                RenderedPage(
                    index=index,
                    data=payload,
                    mime=mime,
                    width=width,
                    height=height,
                    dpi=max(1, round(PAGE_DPI * width / max(pixmap.width, 1))),
                )
            )
        return pages


def _render_image(
    document: LoadedDocument,
    requested: list[int] | None,
) -> list[RenderedPage]:
    if not _select_pages(requested, 1):
        return []
    try:
        with Image.open(io.BytesIO(document.data)) as image:
            image.load()
            payload, mime, width, height = encode_image(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise APIError(400, "invalid_document", "Image could not be decoded.") from exc
    return [
        RenderedPage(
            index=0,
            data=payload,
            mime=mime,
            width=width,
            height=height,
            dpi=PAGE_DPI,
        )
    ]


def _select_pages(requested: list[int] | None, count: int) -> list[int]:
    if requested is None:
        if count > MAX_PAGES:
            raise APIError(
                400,
                "invalid_request",
                f"Document has {count} pages, limit is {MAX_PAGES}. "
                "Pass `pages` to select a range.",
            )
        return list(range(count))
    out_of_range = [index for index in requested if index >= count]
    if out_of_range:
        raise APIError(
            400,
            "invalid_request",
            f"Page index {out_of_range[0]} is out of range; document has {count} pages.",
        )
    return requested


def encode_image(
    image: Image.Image,
    *,
    quality: int = PAGE_IMAGE_QUALITY,
) -> tuple[bytes, str, int, int]:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue(), "image/jpeg", image.width, image.height


def normalized_bbox_to_pixels(
    page: RenderedPage,
    bbox: Sequence[float],
) -> tuple[int, int, int, int]:
    """Map a normalized 0-1000 bbox onto the rendered page image."""
    x1, y1, x2, y2 = bbox
    scan = max(page.width - 1, 1)
    height = max(page.height - 1, 1)
    left = round(x1 / _NORMALIZED_BBOX_SIZE * scan)
    top = round(y1 / _NORMALIZED_BBOX_SIZE * height)
    right = round(x2 / _NORMALIZED_BBOX_SIZE * scan)
    bottom = round(y2 / _NORMALIZED_BBOX_SIZE * height)
    return left, top, right, bottom


def crop_page(
    page: RenderedPage,
    pixels: tuple[int, int, int, int],
) -> tuple[bytes, str] | None:
    """Crop a pixel bbox out of a rendered page and re-encode it."""
    left, top, right, bottom = pixels
    if right - left < 8 or bottom - top < 8:
        return None
    try:
        with Image.open(io.BytesIO(page.data)) as image:
            cropped = image.crop((left, top, right, bottom))
            payload, mime, _, _ = encode_image(cropped, quality=90)
            return payload, mime
    except (UnidentifiedImageError, OSError):
        return None
