"""Contract tests for the Mistral-compatible `/v1/ocr` endpoint."""

from __future__ import annotations

import base64
import json

from helpers import FakeUpstream, client_for, ocr_body

AUTH = {"authorization": "Bearer user-key"}


async def test_data_uri_pdf_returns_mistral_shape(pdf_bytes):
    upstream = FakeUpstream()
    async with client_for(upstream) as client:
        response = await client.post("/v1/ocr", json=ocr_body(pdf_bytes), headers=AUTH)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "mistral-ocr-latest"
    assert body["document_annotation"] is None
    assert [page["index"] for page in body["pages"]] == [0, 1]
    assert body["usage_info"] == {"pages_processed": 2, "doc_size_bytes": len(pdf_bytes)}

    page = body["pages"][0]
    assert page["markdown"] == (
        "# Page 1 of 2\n\n![](img-0.jpeg)\n\nTotal: 1,200 EUR"
    )
    assert page["images"][0]["id"] == "img-0.jpeg"
    assert page["images"][0]["image_base64"] is None
    assert page["images"][0]["top_left_x"] >= 0
    assert page["dimensions"]["dpi"] == 200
    assert page["dimensions"]["width"] > 1600
    assert page["tables"][0]["format"] == "markdown"
    assert page["hyperlinks"] == ["https://example.com"]
    assert page["header"] is None

    first = upstream.bodies[0]
    assert upstream.requests[0].headers["authorization"] == "Bearer user-key"
    assert first["model"] == "mistral-ocr-latest"
    assert first["response_format"]["type"] == "json_schema"
    assert first["response_format"]["json_schema"]["name"] == "ocr_page"
    assert first["response_format"]["json_schema"]["strict"] is True
    # Reasoning-model defaults: sampling and token limits are left to the model.
    assert "max_tokens" not in first
    assert "max_completion_tokens" not in first
    assert "temperature" not in first
    image_part = first["messages"][1]["content"][1]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_image_crops_and_header_extraction(pdf_bytes):
    upstream = FakeUpstream()
    async with client_for(upstream) as client:
        response = await client.post(
            "/v1/ocr",
            json=ocr_body(pdf_bytes, include_image_base64=True, extract_header=True, extract_footer=True),
            headers=AUTH,
        )

    page = response.json()["pages"][0]
    crop = base64.b64decode(page["images"][0]["image_base64"])
    assert crop[:2] == b"\xff\xd8"
    assert page["header"] == "Quarterly report"
    assert page["footer"] == "Confidential"

    schema = upstream.bodies[0]["response_format"]["json_schema"]["schema"]
    assert schema["required"] == [
        "markdown",
        "images",
        "tables",
        "hyperlinks",
        "header",
        "footer",
    ]
    assert schema["additionalProperties"] is False


async def test_pages_selection_and_bounds(pdf_bytes):
    upstream = FakeUpstream()
    async with client_for(upstream) as client:
        selected = await client.post("/v1/ocr", json=ocr_body(pdf_bytes, pages="1"), headers=AUTH)
        too_far = await client.post("/v1/ocr", json=ocr_body(pdf_bytes, pages=[7]), headers=AUTH)
        bad_spec = await client.post("/v1/ocr", json=ocr_body(pdf_bytes, pages="0-x"), headers=AUTH)

    assert selected.status_code == 200
    assert [page["index"] for page in selected.json()["pages"]] == [1]
    assert selected.json()["usage_info"]["pages_processed"] == 1
    assert len(upstream.bodies) == 1
    assert too_far.status_code == 400
    assert bad_spec.status_code == 400


async def test_model_falls_back_to_configuration(pdf_bytes):
    upstream = FakeUpstream()
    body = ocr_body(pdf_bytes)
    body.pop("model")
    async with client_for(upstream) as client:
        response = await client.post("/v1/ocr", json=body, headers=AUTH)

    assert response.status_code == 200
    assert response.json()["model"] == "test-vision"
    assert upstream.bodies[0]["model"] == "test-vision"


async def test_missing_api_key(pdf_bytes):
    async with client_for(FakeUpstream()) as client:
        response = await client.post("/v1/ocr", json=ocr_body(pdf_bytes))

    assert response.status_code == 401
    assert response.json()["error"]["type"] == "missing_api_key"


async def test_env_key_fallback_is_forwarded(pdf_bytes):
    upstream = FakeUpstream()
    async with client_for(upstream, upstream_api_key="env-key") as client:
        response = await client.post("/v1/ocr", json=ocr_body(pdf_bytes))

    assert response.status_code == 200
    assert upstream.requests[0].headers["authorization"] == "Bearer env-key"


async def test_upstream_auth_error_passthrough(pdf_bytes):
    upstream = FakeUpstream(
        status=401,
        error_body={"error": {"message": "Incorrect API key provided"}},
    )
    async with client_for(upstream) as client:
        response = await client.post("/v1/ocr", json=ocr_body(pdf_bytes), headers=AUTH)

    assert response.status_code == 401
    error = response.json()["error"]
    assert error["type"] == "upstream_auth_error"
    assert "Incorrect API key" in error["message"]


async def test_structured_output_downgrade_is_sticky(pdf_bytes):
    upstream = FakeUpstream(reject_json_schema=1)
    async with client_for(upstream) as client:
        response = await client.post("/v1/ocr", json=ocr_body(pdf_bytes), headers=AUTH)

    assert response.status_code == 200, response.text
    modes = [body.get("response_format", {}).get("type") for body in upstream.bodies]
    assert modes[0] == "json_schema"
    assert set(modes[1:]) == {"json_object"}
    # The schema still reaches the model through the prompt.
    assert "JSON Schema" in upstream.bodies[1]["messages"][-1]["content"]


async def test_document_annotation_uses_user_schema(pdf_bytes):
    upstream = FakeUpstream()
    annotation_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "invoice",
            "schema": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "total": {"type": "number"}},
            },
            "strict": True,
        },
    }
    async with client_for(upstream) as client:
        response = await client.post(
            "/v1/ocr",
            json=ocr_body(
                pdf_bytes,
                document_annotation_format=annotation_format,
                document_annotation_prompt="Extract invoice fields.",
            ),
            headers=AUTH,
        )

    assert response.status_code == 200, response.text
    assert json.loads(response.json()["document_annotation"]) == {"title": "Invoice", "total": 42}

    annotation_request = upstream.bodies[-1]
    assert annotation_request["response_format"]["json_schema"]["name"] == "invoice"
    schema = annotation_request["response_format"]["json_schema"]["schema"]
    assert schema["required"] == ["title", "total"]
    assert schema["additionalProperties"] is False
    assert "Extract invoice fields." in annotation_request["messages"][1]["content"]


async def test_multipart_upload(pdf_bytes):
    async with client_for(FakeUpstream()) as client:
        response = await client.post(
            "/v1/ocr",
            files={"file": ("invoice.pdf", pdf_bytes, "application/pdf")},
            data={"pages": "0", "include_image_base64": "true", "extract_header": "true"},
            headers=AUTH,
        )

    assert response.status_code == 200, response.text
    pages = response.json()["pages"]
    assert len(pages) == 1
    assert pages[0]["images"][0]["image_base64"]
    assert pages[0]["header"] == "Quarterly report"


async def test_rejections(pdf_bytes):
    async with client_for(FakeUpstream()) as client:
        blocks = await client.post("/v1/ocr", json=ocr_body(pdf_bytes, include_blocks=True), headers=AUTH)
        docx = await client.post(
            "/v1/ocr",
            json={
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": "data:application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document;base64,UEsDBA==",
                },
            },
            headers=AUTH,
        )
        file_chunk = await client.post(
            "/v1/ocr",
            json={
                "model": "mistral-ocr-latest",
                "document": {"type": "file", "file_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301"},
            },
            headers=AUTH,
        )

    assert blocks.status_code == 400
    assert blocks.json()["error"]["type"] == "unsupported_option"
    assert docx.status_code == 415
    assert file_chunk.status_code == 400
    assert file_chunk.json()["error"]["type"] == "unsupported_document"


async def test_unknown_request_field_is_rejected(pdf_bytes):
    async with client_for(FakeUpstream()) as client:
        response = await client.post(
            "/v1/ocr",
            json=ocr_body(pdf_bytes, nonsense=True),
            headers=AUTH,
        )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "nonsense"
