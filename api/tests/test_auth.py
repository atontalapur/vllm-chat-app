"""The auth boundary.

/health and /metrics must stay open: the Compose healthcheck and Prometheus
have no reason to hold the application key, and 401ing them breaks startup
ordering and leaves the dashboard silently empty.
"""

import pytest
from app.auth import require_api_key
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tests.conftest import VALID_KEY


def test_chat_requires_key(client: TestClient, valid_body: dict[str, object]) -> None:
    r = client.post("/chat/stream", json=valid_body)
    assert r.status_code == 401


def test_chat_rejects_wrong_key(client: TestClient, valid_body: dict[str, object]) -> None:
    r = client.post("/chat/stream", json=valid_body, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_health_is_open(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_is_open(client: TestClient) -> None:
    """Prometheus scrapes without a key. If this ever 401s, the dashboard dies."""
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "python_info" in r.text or "http_request" in r.text


def test_request_id_echoed(client: TestClient) -> None:
    r = client.get("/health")
    assert r.headers.get("X-Request-ID")


def test_valid_key_passes_auth(
    client: TestClient, valid_body: dict[str, object], respx_mock
) -> None:
    """A valid key gets past auth — proven by the failure being upstream, not 401."""
    import httpx

    from tests.conftest import UPSTREAM_URL

    respx_mock.post(UPSTREAM_URL).mock(side_effect=httpx.ConnectError("refused"))
    r = client.post("/chat/stream", json=valid_body, headers={"X-API-Key": VALID_KEY})
    assert r.status_code == 502


async def test_non_ascii_key_is_rejected_not_crashed() -> None:
    """A non-ASCII key must be rejected with 401, never raise.

    Starlette decodes inbound headers as latin-1, so a raw high byte arrives
    as a non-ASCII str. secrets.compare_digest raises TypeError on those,
    which surfaced as an unauthenticated 500 with a traceback in the log.
    Exercised at the dependency directly: HTTP clients refuse to encode a
    non-ASCII header value, so this path is unreachable through TestClient.
    """
    with pytest.raises(HTTPException) as exc:
        await require_api_key("caf\xe9")
    assert exc.value.status_code == 401
