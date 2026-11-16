# ocrproxy

Mistral-compatible `/v1/ocr` endpoint backed by an OpenAI-compatible vision model.

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

Configuration comes from environment variables or a `.env` file. Only the upstream provider is configurable, and each request may override its base URL.

| Variable | Default | Meaning |
| --- | --- | --- |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Base URL of the one upstream provider; `/chat/completions` is appended |
| `UPSTREAM_MODEL` | `gpt-6-luna` | Fallback model when the request omits `model`; the request's model name is forwarded unchanged |
| `UPSTREAM_API_KEY` | unset | Fallback key when a request carries none. Leave unset to require client keys |

Fixed limits: 30 MiB documents, 100 pages per request, 200 DPI page renders, 4 concurrent page transcriptions, 120 s upstream timeout, and JPEG page images at quality 85. Requests set no `temperature`, `max_tokens`, or reasoning knobs, so model defaults apply.

## Run

```sh
UPSTREAM_BASE_URL=https://api.openai.com/v1 \
UPSTREAM_MODEL=gpt-6-luna \
uv run uvicorn ocrproxy.main:app --host 0.0.0.0 --port 8000
```

```sh
curl -s http://localhost:8000/v1/ocr \
  -H "Authorization: Bearer $PROVIDER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-6-luna","document":{"type":"document_url","document_url":"https://example.com/report.pdf"}}'
```

`document_url` accepts an `http(s)` URL or a `data:application/pdf;base64,...` URI. Remote fetches reject hosts that resolve to private, loopback, or link-local addresses. The same path also accepts `multipart/form-data` with a `file` field and optional form fields (`base_url`, `pages`, `include_image_base64`, `table_format`, `extract_header`, `extract_footer`, `document_annotation_prompt`, `document_annotation_format`, `bbox_annotation_format` as JSON strings).

Requests may set `base_url` in the JSON body or as a multipart field to send the OCR calls to another OpenAI-compatible endpoint. The proxy appends `/chat/completions` and preserves any query string. The override requires a client-supplied key, so the configured fallback key is never sent to a request-selected host.

The client key is forwarded unchanged to the upstream as a bearer token. Keys are never stored or logged.

## Requests the proxy does not support

- `file_id` chunks, which point at Mistral's own file store
- `include_blocks` and `confidence_scores_granularity`
- Non-PDF, non-image documents (DOCX, HTML)

These return `400`, or `415` for an unsupported document type, with an explicit message rather than a partial response.

## Container

```sh
docker build -t ocrproxy .
```

The `Container` workflow publishes `ghcr.io/<owner>/ocrproxy` with `latest`, branch, tag, and commit tags on pushes to `main` and on version tags. Pull requests build without pushing.

## Develop

```sh
uv run pytest
uv run python -m uvicorn ocrproxy.main:app --reload
```
