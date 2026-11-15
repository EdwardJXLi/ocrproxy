# ocrproxy

Mistral-compatible `/v1/ocr` endpoint backed by a single OpenAI-compatible vision model.

Clients post documents in Mistral's OCR request format. The service renders each page to an image, asks the configured LLM for Markdown plus structured page metadata through JSON Schema outputs, and returns Mistral's OCR response format.

## Requirements

- Python 3.11 or newer
- An OpenAI-compatible chat-completions endpoint with vision support
- No external binaries: PDF rendering uses PyMuPDF

## Install

```sh
uv sync
```

## Configure

Configuration comes from environment variables or a `.env` file.

| Variable | Default | Meaning |
| --- | --- | --- |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Base URL of the one upstream provider; `/chat/completions` is appended |
| `UPSTREAM_MODEL` | `gpt-4.1-mini` | Vision model used for OCR |
| `UPSTREAM_API_KEY` | unset | Fallback key when a request carries none. Leave unset to require client keys |
| `UPSTREAM_TIMEOUT_SECONDS` | `120` | Per-request upstream timeout |
| `UPSTREAM_MAX_RETRIES` | `2` | Retries for 429/5xx/timeouts |
| `UPSTREAM_STRUCTURED_OUTPUT` | `json_schema` | `json_schema`, `json_object`, or `none`. Downgrades automatically if the upstream rejects a mode |
| `UPSTREAM_MAX_TOKENS_PARAM` | `max_tokens` | Use `max_completion_tokens` for OpenAI reasoning models |
| `UPSTREAM_IMAGE_DETAIL` | `high` | `high`, `low`, or `auto` (omit the field) |
| `OCR_MODEL_NAME` | `mistral-ocr-latest` | Model name echoed in responses |
| `OCR_DPI` | `200` | PDF render resolution |
| `OCR_IMAGE_FORMAT` | `jpeg` | `jpeg` or `png` page images sent upstream |
| `OCR_IMAGE_QUALITY` | `85` | JPEG quality |
| `OCR_MAX_IMAGE_SIDE` | `0` | Downscale page images above this pixel size; `0` disables |
| `OCR_MAX_PAGES` | `100` | Maximum pages per document or page selection |
| `OCR_MAX_DOCUMENT_MB` | `30` | Maximum document size |
| `OCR_PAGE_CONCURRENCY` | `4` | Concurrent page transcriptions |
| `OCR_MAX_TOKENS` | `8192` | Max output tokens per upstream call |
| `OCR_TEMPERATURE` | `0` | Set to an empty value to omit the field |
| `OCR_FETCH_TIMEOUT_SECONDS` | `30` | Timeout for fetching `document_url` over HTTP |
| `LOG_LEVEL` | `INFO` | Python log level |

## Run

```sh
UPSTREAM_BASE_URL=https://api.openai.com/v1 \
UPSTREAM_MODEL=gpt-4.1-mini \
uv run uvicorn ocrproxy.main:app --host 0.0.0.0 --port 8000
```

```sh
curl -s http://localhost:8000/v1/ocr \
  -H "Authorization: Bearer $PROVIDER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral-ocr-latest","document":{"type":"document_url","document_url":"https://example.com/report.pdf"}}'
```

`document_url` accepts an `http(s)` URL or a `data:application/pdf;base64,...` URI. The same path also accepts `multipart/form-data` with a `file` field and optional form fields (`pages`, `include_image_base64`, `table_format`, `extract_header`, `extract_footer`, `document_annotation_prompt`, `document_annotation_format`, `bbox_annotation_format` as JSON strings).

The client key is forwarded unchanged to the upstream as a bearer token. Keys are never stored or logged.

## Requests the proxy does not support

- `file_id` chunks, which point at Mistral's own file store
- `include_blocks` and `confidence_scores_granularity`
- Non-PDF, non-image documents (DOCX, HTML)

These return `400`, or `415` for an unsupported document type, with an explicit message rather than a partial response.

## Develop

```sh
uv run pytest
uv run python -m uvicorn ocrproxy.main:app --reload
```
