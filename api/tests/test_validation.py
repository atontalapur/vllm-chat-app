"""Request validation: malformed input is rejected before it reaches the GPU."""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import VALID_KEY

AUTH = {"X-API-Key": VALID_KEY}


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"messages": []}, "empty conversation"),
        ({"messages": [{"role": "user", "content": ""}], "max_tokens": 10}, "empty content"),
        ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}, "max_tokens too low"),
        (
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4097},
            "max_tokens over cap",
        ),
        (
            {"messages": [{"role": "user", "content": "hi"}], "temperature": 2.1},
            "temperature over cap",
        ),
        (
            {"messages": [{"role": "user", "content": "hi"}], "temperature": -0.1},
            "temperature below zero",
        ),
        ({"messages": [{"role": "wizard", "content": "hi"}]}, "unknown role"),
        ({"messages": [{"role": "assistant", "content": "hi"}]}, "nothing to answer"),
    ],
)
def test_rejects_malformed(client: TestClient, body: dict[str, object], reason: str) -> None:
    r = client.post("/chat/stream", json=body, headers=AUTH)
    assert r.status_code == 422, f"should reject: {reason}"


def test_rejects_oversized_conversation(client: TestClient) -> None:
    """Unbounded history would overrun the context window mid-conversation.

    Caught here with a readable 413 rather than as an opaque upstream failure.
    """
    huge = "x" * 50_000
    r = client.post(
        "/chat/stream",
        json={"messages": [{"role": "user", "content": huge}]},
        headers=AUTH,
    )
    assert r.status_code == 413
    payload = r.json()
    assert payload["error"] == "conversation_too_large"
    assert payload["request_id"]
    # The UI shows `detail` verbatim, so it has to tell the user what to do.
    assert "Start a new chat" in payload["detail"]
