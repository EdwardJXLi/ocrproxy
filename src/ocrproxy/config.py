"""Upstream provider configuration and fixed service limits.

Only the upstream provider is configurable. Everything else is a constant, so
the proxy stays a thin, predictable layer.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Documents.
MAX_DOCUMENT_BYTES = 30 * 1024 * 1024
MAX_PAGES = 100
PAGE_DPI = 200
PAGE_IMAGE_QUALITY = 85
PAGE_CONCURRENCY = 4
FETCH_TIMEOUT_SECONDS = 30.0

# Upstream.
UPSTREAM_TIMEOUT_SECONDS = 120.0
UPSTREAM_MAX_RETRIES = 2
UPSTREAM_IMAGE_DETAIL = "high"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_model: str = "gpt-6-luna"
    upstream_api_key: str | None = None

    @field_validator("upstream_base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        url = value.strip().rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError("UPSTREAM_BASE_URL must start with http:// or https://")
        return url


@lru_cache
def get_settings() -> Settings:
    return Settings()
