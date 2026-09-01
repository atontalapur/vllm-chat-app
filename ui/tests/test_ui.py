"""Smoke coverage for the chat UI.

The UI is a single Streamlit script, so it gets a smoke check rather than a
full suite: the module must import, and the error-mapping function must turn
every api failure into something a person can act on. That function is pure,
so it is worth testing directly — a silent or misleading error banner is the
difference between a user retrying and a user assuming the app is broken.
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("API_KEY", "test-key")


def test_module_imports() -> None:
    """Catches syntax errors and bad imports before the container starts."""
    import app

    assert callable(app.main)
    assert callable(app.stream_reply)


@pytest.mark.parametrize(
    ("status", "expected_fragment"),
    [
        (401, "Authentication failed"),
        (413, "too large"),
        (422, "rejected as invalid"),
        (502, "model server is unavailable"),
        (418, "Unexpected error 418"),
    ],
)
def test_error_messages_are_actionable(status: int, expected_fragment: str) -> None:
    import app

    response = Mock()
    response.status_code = status
    response.json.return_value = {"detail": "conversation is too large. Start a new chat."}

    message = app._describe_error(response)

    assert expected_fragment in message
    assert message, "an empty error message leaves the user with no idea what happened"


def test_error_message_survives_a_non_json_body() -> None:
    """A crashed upstream may return HTML or plain text, not JSON."""
    import app

    response = Mock()
    response.status_code = 500
    response.json.side_effect = ValueError("not json")
    response.text = "<html>502 Bad Gateway</html>"

    message = app._describe_error(response)

    assert "500" in message
