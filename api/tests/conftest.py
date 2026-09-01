"""Test configuration.

Environment is set before importing the app: config.py validates at import
time, so the settings have to exist first.
"""

import os

os.environ.setdefault("API_KEY", "test-key-12345")
os.environ.setdefault("MODEL_ID", "test-model")
os.environ.setdefault("VLLM_BASE_URL", "http://upstream-under-test:8000/v1")

import pytest  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

UPSTREAM_URL = "http://upstream-under-test:8000/v1/chat/completions"
VALID_KEY = "test-key-12345"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def valid_body() -> dict[str, object]:
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
        "temperature": 0.7,
    }


def sse(*chunks: str) -> str:
    """Build an SSE body the way the upstream server emits one."""
    return "".join(f"data: {c}\n\n" for c in chunks) + "data: [DONE]\n\n"
