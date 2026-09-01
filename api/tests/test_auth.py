"""The auth boundary.

/health and /metrics must stay open: the Compose healthcheck and Prometheus
have no reason to hold the application key, and 401ing them breaks startup
ordering and leaves the dashboard silently empty.
"""

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
