"""Wire models for the Mistral-compatible `/v1/ocr` endpoint.

Field names, optionality and response shape follow Mistral's OpenAPI
`OCRRequest` / `OCRResponse` definitions so existing clients keep working.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import normalize_base_url


class DocumentURLChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["document_url"] = "document_url"
    document_url: str
    document_name: str | None = None


class ImageURLRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    detail: str | None = None


class ImageURLChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"] = "image_url"
    image_url: str | ImageURLRef


class FileChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["file"]
    file_id: UUID


DocumentInput = Annotated[
    DocumentURLChunk | ImageURLChunk | FileChunk,
    Field(discriminator="type"),
]


class JsonSchemaSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    description: str | None = None
    schema_definition: dict = Field(alias="schema")
    strict: bool = False


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JsonSchemaSpec | None = None


class OCROptions(BaseModel):
    """Everything in `OCRRequest` except the document itself."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    base_url: str | None = None
    pages: str | list[int] | None = None
    include_image_base64: bool = False
    image_limit: int | None = Field(None, ge=0)
    image_min_size: int | None = Field(None, ge=0)
    bbox_annotation_format: ResponseFormat | None = None
    document_annotation_format: ResponseFormat | None = None
    document_annotation_prompt: str | None = None
    table_format: Literal["markdown", "html"] | None = None
    extract_header: bool = False
    extract_footer: bool = False
    # Accepted for schema compatibility, then rejected with a clear error.
    include_blocks: bool | None = None
    confidence_scores_granularity: Literal["word", "page", "block"] | None = None

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str | None) -> str | None:
        return None if value is None else normalize_base_url(value)


class OCRRequest(OCROptions):
    document: DocumentInput


class OCRPageDimensions(BaseModel):
    dpi: int
    height: int
    width: int


class OCRImageObject(BaseModel):
    id: str
    top_left_x: int | None = None
    top_left_y: int | None = None
    bottom_right_x: int | None = None
    bottom_right_y: int | None = None
    image_base64: str | None = None
    image_annotation: str | None = None


class OCRTableObject(BaseModel):
    id: str
    content: str
    format: Literal["markdown", "html"]


class OCRPageObject(BaseModel):
    index: int
    markdown: str
    images: list[OCRImageObject]
    tables: list[OCRTableObject] = Field(default_factory=list)
    hyperlinks: list[str] = Field(default_factory=list)
    header: str | None = None
    footer: str | None = None
    dimensions: OCRPageDimensions | None = None


class OCRUsageInfo(BaseModel):
    pages_processed: int
    doc_size_bytes: int | None = None


class OCRResponse(BaseModel):
    pages: list[OCRPageObject]
    model: str
    document_annotation: str | None = None
    usage_info: OCRUsageInfo
