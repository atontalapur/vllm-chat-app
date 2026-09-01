"""Startup configuration validation.

A misconfigured deployment must fail at boot with a readable error, not run
degraded and produce confusing 401s much later.
"""

import pytest
from app.config import Settings
from pydantic import ValidationError


def test_missing_api_key_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(model_id="m", api_key="", _env_file=None)  # type: ignore[call-arg]


def test_missing_model_id_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(api_key="k", model_id="", _env_file=None)  # type: ignore[call-arg]


def test_defaults_point_at_compose_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production default is the in-compose vLLM service; local dev overrides it.

    The ambient VLLM_BASE_URL must be cleared: pydantic-settings reads the
    environment regardless of _env_file, so leaving it set would test the
    conftest value rather than the default.
    """
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    s = Settings(api_key="k", model_id="m", _env_file=None)  # type: ignore[call-arg]
    assert s.vllm_base_url == "http://vllm:8000/v1"
    assert s.upstream_connect_timeout_s == 10.0
