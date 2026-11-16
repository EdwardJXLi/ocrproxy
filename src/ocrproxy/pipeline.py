"""OCR pipeline: render pages, transcribe each one, assemble the response."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import time
from typing import Any, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import (
    MAX_PAGES,
    PAGE_CONCURRENCY,
    UPSTREAM_IMAGE_DETAIL,
    Settings,
)
from .documents import (
    LoadedDocument,
    RenderedPage,
    crop_page,
    load_document,
    normalized_bbox_to_pixels,
    parse_page_spec,
    render_pages,
)
from .errors import APIError
from .schemas import (
    OCROptions,
    OCRImageObject,
    OCRPageDimensions,
    OCRPageObject,
    OCRResponse,
    OCRTableObject,
    OCRUsageInfo,
    ResponseFormat,
)
from .upstream import UpstreamClient

logger = logging.getLogger(__name__)

_BBOX_SCALE = 1000


class _ExtractedImage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Any = ""
    bbox: Any = None


class _ExtractedTable(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Any = ""
    content: Any = ""


class PageExtraction(BaseModel):
    """Lenient view of the page schema: the upstream may relax strictness."""

    model_config = ConfigDict(extra="ignore")

    markdown: str = ""
    images: list[_ExtractedImage] = Field(default_factory=list)
    tables: list[_ExtractedTable] = Field(default_factory=list)
    hyperlinks: list[Any] = Field(default_factory=list)
    header: str | None = None
    footer: str | None = None


def page_extraction_schema(*, extract_header: bool, extract_footer: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "markdown": {
            "type": "string",
            "description": "Page content as GitHub-flavored Markdown in reading order.",
        },
        "images": {
            "type": "array",
            "description": "One entry per image placeholder used in the markdown.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "bbox"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Placeholder id exactly as written in the markdown, e.g. img-0.",
                    },
                    "bbox": {
                        "type": "array",
                        "description": "Bounding box [x1, y1, x2, y2] in integers normalized to 0-1000, "
                        "origin at the top-left corner of the page image.",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
            },
        },
        "tables": {
            "type": "array",
            "description": "Every table on the page, in reading order.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "content"],
                "properties": {
                    "id": {"type": "string", "description": "Table id, e.g. table-0."},
                    "content": {
                        "type": "string",
                        "description": "The complete table markup.",
                    },
                },
            },
        },
        "hyperlinks": {
            "type": "array",
            "description": "Every link target that appears on the page.",
            "items": {"type": "string"},
        },
    }
    required = list(properties)
    if extract_header:
        properties["header"] = {
            "type": ["string", "null"],
            "description": "The page header text, which must not repeat inside markdown.",
        }
        required.append("header")
    if extract_footer:
        properties["footer"] = {
            "type": ["string", "null"],
            "description": "The page footer text, which must not repeat inside markdown.",
        }
        required.append("footer")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _page_system_prompt(*, table_format: str, extract_header: bool, extract_footer: bool) -> str:
    if table_format == "html":
        table_rule = "Render every table as HTML <table> markup inside the Markdown."
    else:
        table_rule = "Render every table as a GitHub-flavored Markdown pipe table."
    lines = [
        "You transcribe document page images into Markdown for a document archive.",
        "Transcribe all visible text exactly as it appears. Do not summarize, translate, or correct it.",
        "Preserve reading order, headings, lists, code blocks, footnotes, and multi-column layout.",
        table_rule,
        "Write inline math as $...$ and display math as $$...$$.",
        "For each figure, chart, diagram, stamp, or photo, write a placeholder ![img-N](img-N) on its own line where it appears.",
        "\"images\" holds one entry per placeholder: the same id plus its bounding box.",
        "Bounding boxes are [x1, y1, x2, y2] integers normalized to 0-1000 of the page image, origin at the top left.",
        "Number image ids from img-0 in order of appearance.",
        "Keep hyperlinks inline as Markdown links and list every link target in \"hyperlinks\".",
        "For a blank page return an empty string and empty arrays.",
        "Reply with the JSON object only.",
    ]
    if extract_header:
        lines.append("Put the page header text in \"header\" and leave it out of \"markdown\".")
    if extract_footer:
        lines.append("Put the page footer text in \"footer\" and leave it out of \"markdown\".")
    return "\n".join(lines)


def _page_messages(
    page: RenderedPage,
    *,
    total: int,
    options: OCROptions,
    document_name: str | None,
) -> list[dict[str, Any]]:
    system = _page_system_prompt(
        table_format=options.table_format or "markdown",
        extract_header=options.extract_header,
        extract_footer=options.extract_footer,
    )
    label = f"Page {page.index + 1} of {total}."
    if document_name:
        label += f" Document: {document_name}."
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{label} Transcribe this page."},
                {
                    "type": "image_url",
                    "image_url": _image_url(page.data, page.mime),
                },
            ],
        },
    ]


def _image_url(data: bytes, mime: str) -> dict[str, Any]:
    encoded = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return {"url": encoded, "detail": UPSTREAM_IMAGE_DETAIL}


async def run_ocr(
    *,
    source: Any,
    options: OCROptions,
    settings: Settings,
    upstream: UpstreamClient,
    http: httpx.AsyncClient,
    api_key: str,
    base_url: str | None = None,
) -> OCRResponse:
    _reject_unsupported(options)
    # Validate annotation requests before spending upstream calls on pages.
    if options.bbox_annotation_format is not None:
        _annotation_spec(options.bbox_annotation_format, "bbox_annotation_format")
    if options.document_annotation_format is not None:
        _annotation_spec(options.document_annotation_format, "document_annotation_format")

    # The requested model name goes upstream unchanged; the env default only
    # applies when the client omits it.
    model = (options.model or "").strip() or settings.upstream_model

    started = time.monotonic()
    document: LoadedDocument = await load_document(source, http=http)
    requested = parse_page_spec(options.pages, limit=MAX_PAGES)
    pages = await render_pages(document, requested=requested)
    logger.info(
        "document %s mime=%s bytes=%d pages=%d",
        document.name or "<unnamed>",
        document.mime,
        len(document.data),
        len(pages),
    )

    semaphore = asyncio.Semaphore(PAGE_CONCURRENCY)
    page_objects = await asyncio.gather(
        *(
            _ocr_page(
                page,
                total=len(pages),
                model=model,
                options=options,
                document_name=document.name,
                upstream=upstream,
                api_key=api_key,
                base_url=base_url,
                semaphore=semaphore,
            )
            for page in pages
        )
    )

    annotation: str | None = None
    if options.document_annotation_format is not None:
        annotation = await _annotate_document(
            page_objects,
            model=model,
            options=options,
            upstream=upstream,
            api_key=api_key,
            base_url=base_url,
        )

    logger.info(
        "document %s done pages=%d elapsed=%.2fs",
        document.name or "<unnamed>",
        len(page_objects),
        time.monotonic() - started,
    )
    return OCRResponse(
        pages=list(page_objects),
        model=model,
        document_annotation=annotation,
        usage_info=OCRUsageInfo(
            pages_processed=len(page_objects),
            doc_size_bytes=len(document.data),
        ),
    )


async def _ocr_page(
    page: RenderedPage,
    *,
    total: int,
    model: str,
    options: OCROptions,
    document_name: str | None,
    upstream: UpstreamClient,
    api_key: str,
    base_url: str | None,
    semaphore: asyncio.Semaphore,
) -> OCRPageObject:
    schema = page_extraction_schema(
        extract_header=options.extract_header,
        extract_footer=options.extract_footer,
    )
    async with semaphore:
        payload = await upstream.complete_json(
            api_key=api_key,
            model=model,
            messages=_page_messages(
                page,
                total=total,
                options=options,
                document_name=document_name,
            ),
            schema_name="ocr_page",
            schema=schema,
            strict=True,
            base_url=base_url,
        )
    extraction = _validate_extraction(payload, page.index)

    markdown = extraction.markdown.strip()
    pending: list[tuple[OCRImageObject, str, bytes, str]] = []
    for candidate in extraction.images:
        bbox = _normalize_bbox(candidate.bbox)
        if bbox is None:
            logger.debug("page %d: dropping image with unusable bbox %r", page.index, candidate.bbox)
            continue
        pixels = normalized_bbox_to_pixels(page, bbox)
        if options.image_min_size and (
            pixels[2] - pixels[0] < options.image_min_size
            or pixels[3] - pixels[1] < options.image_min_size
        ):
            continue
        if options.image_limit is not None and len(pending) >= options.image_limit:
            break
        crop = crop_page(page, pixels)
        if crop is None:
            logger.debug("page %d: bbox %r cropped to nothing", page.index, bbox)
            continue
        image = OCRImageObject(
            id="",
            top_left_x=pixels[0],
            top_left_y=pixels[1],
            bottom_right_x=pixels[2],
            bottom_right_y=pixels[3],
        )
        pending.append((image, str(candidate.id or "").strip(), crop[0], crop[1]))

    images: list[OCRImageObject] = []
    for index, (image, source_id, crop_bytes, crop_mime) in enumerate(pending):
        target_id = f"img-{index}.jpeg"
        markdown = _rewrite_placeholder(markdown, source_id, target_id)
        image.id = target_id
        if options.include_image_base64:
            image.image_base64 = base64.b64encode(crop_bytes).decode("ascii")
        images.append(image)

    if options.bbox_annotation_format is not None and pending:
        name, annotation_schema, strict = _annotation_spec(
            options.bbox_annotation_format, "bbox_annotation_format"
        )
        annotations = await asyncio.gather(
            *(
                _annotate_crop(
                    crop_bytes,
                    crop_mime,
                    model=model,
                    schema_name=name,
                    schema=annotation_schema,
                    strict=strict,
                    upstream=upstream,
                    api_key=api_key,
                    base_url=base_url,
                    semaphore=semaphore,
                )
                for _, _, crop_bytes, crop_mime in pending
            )
        )
        for (image, _, _, _), annotation in zip(pending, annotations):
            image.image_annotation = annotation

    table_format = options.table_format or "markdown"
    tables = [
        OCRTableObject(id=f"table-{index}", content=str(table.content), format=table_format)
        for index, table in enumerate(extraction.tables)
        if str(table.content).strip()
    ]

    hyperlinks: list[str] = []
    for link in extraction.hyperlinks:
        if isinstance(link, str) and link.strip() and link.strip() not in hyperlinks:
            hyperlinks.append(link.strip())

    logger.info(
        "page %d transcribed chars=%d images=%d tables=%d",
        page.index,
        len(markdown),
        len(images),
        len(tables),
    )
    return OCRPageObject(
        index=page.index,
        markdown=markdown,
        images=images,
        tables=tables,
        hyperlinks=hyperlinks,
        header=extraction.header if options.extract_header else None,
        footer=extraction.footer if options.extract_footer else None,
        dimensions=OCRPageDimensions(dpi=page.dpi, height=page.height, width=page.width),
    )


async def _annotate_crop(
    data: bytes,
    mime: str,
    *,
    model: str,
    schema_name: str,
    schema: dict[str, Any],
    strict: bool,
    upstream: UpstreamClient,
    api_key: str,
    base_url: str | None,
    semaphore: asyncio.Semaphore,
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You extract structured data from an image cropped out of a document. "
            "Reply with JSON only.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the requested fields from this image."},
                {"type": "image_url", "image_url": _image_url(data, mime)},
            ],
        },
    ]
    async with semaphore:
        result = await upstream.complete_json(
            api_key=api_key,
            model=model,
            messages=messages,
            schema_name=schema_name,
            schema=schema,
            strict=strict,
            base_url=base_url,
        )
    return json.dumps(result, ensure_ascii=False)


async def _annotate_document(
    pages: Sequence[OCRPageObject],
    *,
    model: str,
    options: OCROptions,
    upstream: UpstreamClient,
    api_key: str,
    base_url: str | None,
) -> str:
    name, schema, strict = _annotation_spec(
        options.document_annotation_format, "document_annotation_format"
    )
    instruction = options.document_annotation_prompt or "Extract the requested fields from this document."
    markdown = "\n\n".join(
        f"<!-- page {page.index + 1} -->\n\n{page.markdown}" for page in pages
    )
    messages = [
        {
            "role": "system",
            "content": "You extract structured data from document markdown. Reply with JSON only.",
        },
        {"role": "user", "content": f"{instruction}\n\nDocument:\n\n{markdown}"},
    ]
    result = await upstream.complete_json(
        api_key=api_key,
        model=model,
        messages=messages,
        schema_name=name,
        schema=schema,
        strict=strict,
        base_url=base_url,
    )
    return json.dumps(result, ensure_ascii=False)


def _annotation_spec(format: ResponseFormat, field_name: str) -> tuple[str, dict[str, Any], bool]:
    if format.type != "json_schema" or format.json_schema is None:
        raise APIError(
            400,
            "invalid_request",
            f"`{field_name}` requires type \"json_schema\" with a `json_schema` definition.",
            param=field_name,
        )
    spec = format.json_schema
    schema = spec.schema_definition
    if not isinstance(schema, dict) or not schema:
        raise APIError(
            400,
            "invalid_request",
            f"`{field_name}.json_schema.schema` must be a non-empty JSON object.",
            param=field_name,
        )
    resolved = to_strict_schema(schema) if spec.strict else copy.deepcopy(schema)
    return spec.name.strip() or field_name, resolved, spec.strict


def to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a schema for OpenAI strict mode: all keys required, no extra keys."""
    clone = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict) and node.get("type", "object") == "object":
                node.setdefault("type", "object")
                node["required"] = list(properties)
                node["additionalProperties"] = False
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(clone)
    return clone


def _validate_extraction(payload: dict[str, Any], page_index: int) -> PageExtraction:
    try:
        return PageExtraction.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise APIError(
            502,
            "upstream_invalid_output",
            f"Upstream output for page {page_index} did not match the page schema "
            f"({first.get('loc')}: {first.get('msg')}).",
        ) from exc


def _normalize_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, str):
        value = [part for part in value.replace(";", ",").split(",") if part.strip()]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (
        min(max(x1, 0.0), _BBOX_SCALE),
        min(max(y1, 0.0), _BBOX_SCALE),
        min(max(x2, 0.0), _BBOX_SCALE),
        min(max(y2, 0.0), _BBOX_SCALE),
    )


def _rewrite_placeholder(markdown: str, source_id: str, target_id: str) -> str:
    if not source_id or source_id == target_id:
        return markdown
    variants = {
        source_id,
        f"{source_id}.jpeg",
        f"{source_id}.jpg",
        f"{source_id}.png",
        f"{source_id}.webp",
    }
    for variant in variants:
        markdown = markdown.replace(f"]({variant})", f"]({target_id})")
    return markdown


def _reject_unsupported(options: OCROptions) -> None:
    if options.include_blocks:
        raise APIError(
            400,
            "unsupported_option",
            "`include_blocks` is not supported by this proxy; omit it.",
            param="include_blocks",
        )
    if options.confidence_scores_granularity is not None:
        raise APIError(
            400,
            "unsupported_option",
            "`confidence_scores_granularity` is not supported by this proxy; omit it.",
            param="confidence_scores_granularity",
        )


__all__ = ["run_ocr", "page_extraction_schema", "to_strict_schema"]
