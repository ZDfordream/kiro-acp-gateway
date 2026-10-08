"""Wire invisibility: nothing the gateway authors may name the relay.

Model prose is scrubbed separately (``sanitize.py``); this module covers the
strings the gateway itself writes — response keys, error codes, routes, the
index, health, metrics, the dashboard, and the docs surface. A client reading
any of these must not learn what sits between it and the model.
"""

from __future__ import annotations

import json
import re

import httpx
import pytest

from kiro_acp.gateway.app import create_app
from kiro_acp.gateway.backend import classify_kiro_error
from tests.test_gateway import FakeKiroBackend, make_settings

BRANDED = re.compile(r"kiro|gateway", re.IGNORECASE)


def unbranded(text: str) -> bool:
    return BRANDED.search(text) is None


def assert_unbranded(obj: object) -> None:
    blob = json.dumps(obj).lower()
    hit = BRANDED.search(blob)
    assert hit is None, f"branded token {hit.group(0)!r} in {blob[:400]}"


@pytest.fixture
async def client(workspace, engine):
    settings = make_settings(workspace, engine=engine)
    app = create_app(settings, backend=FakeKiroBackend(settings))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gw",
            headers={"Authorization": "Bearer secret"},
            timeout=30,
        ) as http:
            http.app = app
            yield http


PAYLOAD = {"model": "claude-opus-4.8", "messages": [{"role": "user", "content": "say hi"}]}


@pytest.mark.parametrize(
    "path,body",
    [
        ("/v1/chat/completions", PAYLOAD),
        ("/v1/completions", {"model": "claude-opus-4.8", "prompt": "say hi"}),
        ("/v1/responses", {"model": "claude-opus-4.8", "input": "say hi"}),
        (
            "/v1/messages",
            {"model": "claude-opus-4.8", "max_tokens": 64, "messages": PAYLOAD["messages"]},
        ),
    ],
)
async def test_response_bodies_are_unbranded(client: httpx.AsyncClient, path: str, body: dict):
    resp = await client.post(path, json=body)
    assert resp.status_code == 200, resp.text
    blob = resp.text.lower()
    assert "kiro" not in blob, blob[:400]


async def test_stream_usage_chunk_is_unbranded(client: httpx.AsyncClient):
    resp = await client.post(
        "/v1/chat/completions",
        json={**PAYLOAD, "stream": True, "stream_options": {"include_usage": True}},
    )
    assert resp.status_code == 200, resp.text
    assert "kiro" not in resp.text.lower(), resp.text[:400]


async def test_health_is_unbranded(client: httpx.AsyncClient):
    body = (await client.get("/health")).json()
    assert body["status"] == "ok"
    assert_unbranded(body)


async def test_index_is_unbranded(client: httpx.AsyncClient):
    body = (await client.get("/")).json()
    assert_unbranded(body)


async def test_models_list_is_unbranded(client: httpx.AsyncClient):
    body = (await client.get("/v1/models")).json()
    assert_unbranded(body)


async def test_docs_surfaces_are_absent(client: httpx.AsyncClient):
    for path in ("/docs", "/redoc", "/openapi.json"):
        resp = await client.get(path)
        assert resp.status_code == 404, path


async def test_error_codes_are_unbranded():
    for message in (
        "the model stalled for 30s",
        "Kiro failed to generate a response",
        "request timed out",
        "not signed in",
        "ECONNRESET",
        "something unexpected",
    ):
        _status, _type, code, _retry = classify_kiro_error(message)
        assert unbranded(code), f"branded error code {code!r}"


async def test_unsupported_endpoint_error_is_unbranded(client: httpx.AsyncClient):
    resp = await client.post("/v1/embeddings", json={"model": "m", "input": "x"})
    assert resp.status_code == 501
    assert_unbranded(resp.json())
