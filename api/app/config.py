"""Runtime configuration, read from the environment.

Every required value is validated at import time. A misconfigured deployment
fails immediately with a readable error instead of running degraded and
producing confusing 401s or upstream errors much later.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Shared secret between the UI and this service. No default: an empty key
    # would silently authenticate nothing.
    api_key: str = Field(min_length=1)

    # Model name passed through to the upstream server.
    model_id: str = Field(min_length=1)

    # Upstream OpenAI-compatible server. Defaults to the in-compose vLLM
    # service; local development points this at any other implementation
    # (Ollama, etc.) without touching application code.
    vllm_base_url: str = "http://vllm:8000/v1"

    # Connect timeout only. The response body is deliberately unbounded:
    # generation legitimately takes minutes, and a read timeout would sever
    # long completions mid-stream.
    upstream_connect_timeout_s: float = 10.0

    # Guards against a client sending an unbounded conversation history that
    # would overrun the model's context window. Characters, not tokens: exact
    # token counting would need the model's tokenizer, and this only has to be
    # a sane upper bound, not a precise one.
    max_total_chars: int = 48_000


settings = Settings()  # type: ignore[call-arg]
