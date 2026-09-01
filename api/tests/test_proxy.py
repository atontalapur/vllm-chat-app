"""The streaming proxy and its failure modes.

This is the most bug-prone code in the service: it is async, it forwards a
stream it does not control, and both of its failure modes are invisible to a
happy-path test. The upstream is mocked, so none of this needs a GPU.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.conftest import UPSTREAM_URL, VALID_KEY, sse

AUTH = {"X-API-Key": VALID_KEY}
BODY: dict[str, object] = {"messages": [{"role": "user", "content": "hello"}]}

CHUNK_1 = '{"id":"cmpl-1","choices":[{"delta":{"content":"Hel"}}]}'
CHUNK_2 = '{"id":"cmpl-1","choices":[{"delta":{"content":"lo"}}]}'


def test_streams_chunks_in_order(client: TestClient, respx_mock) -> None:
    respx_mock.post(UPSTREAM_URL).mock(return_value=httpx.Response(200, text=sse(CHUNK_1, CHUNK_2)))

    with client.stream("POST", "/chat/stream", json=BODY, headers=AUTH) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())

    assert body.index("Hel") < body.index("lo"), "chunks must arrive in order"
    assert "[DONE]" in body


def test_sends_correct_upstream_payload(client: TestClient, respx_mock) -> None:
    route = respx_mock.post(UPSTREAM_URL).mock(return_value=httpx.Response(200, text=sse(CHUNK_1)))

    client.post(
        "/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32},
        headers=AUTH,
    )

    sent = route.calls.last.request
    import json

    payload = json.loads(sent.content)
    assert payload["stream"] is True, "streaming must be requested upstream"
    assert payload["max_tokens"] == 32
    assert payload["model"] == "test-model"


def test_connect_failure_returns_502_not_a_hang(client: TestClient, respx_mock) -> None:
    """Unreachable upstream must fail fast and visibly.

    A hang here is the worst outcome: it looks like a slow model, so the user
    waits indefinitely with no error and no way to recover.
    """
    respx_mock.post(UPSTREAM_URL).mock(side_effect=httpx.ConnectError("connection refused"))

    r = client.post("/chat/stream", json=BODY, headers=AUTH)

    assert r.status_code == 502
    payload = r.json()
    assert payload["error"] == "upstream_error"
    assert "cannot reach model server" in payload["detail"]
    assert payload["request_id"]


def test_connect_timeout_returns_502(client: TestClient, respx_mock) -> None:
    respx_mock.post(UPSTREAM_URL).mock(side_effect=httpx.ConnectTimeout("timed out"))
    r = client.post("/chat/stream", json=BODY, headers=AUTH)
    assert r.status_code == 502


def test_upstream_error_status_returns_502(client: TestClient, respx_mock) -> None:
    """A 500 from the model server (OOM, bad request) is not passed through raw."""
    respx_mock.post(UPSTREAM_URL).mock(return_value=httpx.Response(500, text="CUDA out of memory"))

    r = client.post("/chat/stream", json=BODY, headers=AUTH)

    assert r.status_code == 502
    assert "500" in r.json()["detail"]


def test_empty_stream_returns_502(client: TestClient, respx_mock) -> None:
    """A 200 that streams nothing is still a failed generation."""
    respx_mock.post(UPSTREAM_URL).mock(return_value=httpx.Response(200, text=""))
    r = client.post("/chat/stream", json=BODY, headers=AUTH)
    assert r.status_code == 502


@pytest.mark.parametrize("status_code", [400, 404, 429, 503])
def test_all_upstream_error_codes_become_502(
    client: TestClient, respx_mock, status_code: int
) -> None:
    respx_mock.post(UPSTREAM_URL).mock(return_value=httpx.Response(status_code, text="nope"))
    r = client.post("/chat/stream", json=BODY, headers=AUTH)
    assert r.status_code == 502
