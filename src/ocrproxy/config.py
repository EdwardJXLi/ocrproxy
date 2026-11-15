"""Process-wide configuration. One upstream provider, configured by environment."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Upstream: one OpenAI-compatible chat-completions endpoint.
    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_model: str = "gpt-4.1-mini"
    upstream_api_key: str | None = None
    upstream_timeout_seconds: float = Field(120.0, gt=0)
    upstream_max_retries: int = Field(2, ge=0, le=8)
    upstream_structured_output: Literal["json_schema", "json_object", "none"] = "json_schema"
    upstream_max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    upstream_image_detail: Literal["auto", "low", "high"] = "high"

    # Service behaviour.
    ocr_dpi: int = Field(200, ge=72, le=600)
    ocr_image_format: Literal["jpeg", "png"] = "jpeg"
    ocr_image_quality: int = Field(85, ge=1, le=100)
    ocr_max_image_side: int = Field(0, ge=0)
    ocr_max_pages: int = Field(100, ge=1)
    ocr_max_document_mb: float = Field(30.0, gt=0)
    ocr_page_concurrency: int = Field(4, ge=1, le=64)
    ocr_max_tokens: int | None = Field(8192, ge=1)
    ocr_temperature: float | None = Field(0.0, ge=0, le=2)
    ocr_fetch_timeout_seconds: float = Field(30.0, gt=0)
    ocr_allow_private_urls: bool = False

    @field_validator("upstream_base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        url = value.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError("UPSTREAM_BASE_URL must start with http:// or https://")
        return url

    @property
    def max_document_bytes(self) -> int:
        return int(self.ocr_max_document_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
