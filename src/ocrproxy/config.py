"""Upstream provider configuration and fixed service limits.

Only the upstream provider is configurable, and requests may override its base
URL. Everything else is a constant, so the proxy stays a thin, predictable
layer.
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlsplit

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


def normalize_base_url(value: str) -> str:
    """Validate an upstream base URL and strip the trailing slash."""
    url = value.strip().rstrip("/")
    if not url:
        raise ValueError("base URL is empty")
    if any(character.isspace() for character in url):
        raise ValueError("base URL must not contain whitespace")
    try:
        parsed = urlsplit(url)
    except ValueError as exc:
        raise ValueError("base URL is not a valid URL") from exc
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("base URL must start with http:// or https://")
    if parsed.username or parsed.password:
        raise ValueError("base URL must not contain credentials")
    if parsed.fragment:
        raise ValueError("base URL must not contain a fragment")
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_model: str = "gpt-6-luna"
    upstream_api_key: str | None = None

    @field_validator("upstream_base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        return normalize_base_url(value)


@lru_cache
def get_settings() -> Settings:
    return Settings()
