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

    # Ask the upstream server for per-token logprobs, which become the
    # confidence proxy on each trace. Costs roughly 120 bytes per token on the
    # vllm -> api hop (docs/spikes/s0-3-logprob-shape.md) and nothing on the
    # api -> UI hop, since the line is forwarded either way. Off means traces
    # are still written, with a null mean_logprob.
    capture_logprobs: bool = True

    # Where finished traces are written. Empty disables tracing entirely, which
    # is the right behaviour wherever there is no store: the local overlay, CI,
    # and a box whose migrations have not run yet. Points at the INSERT-only
    # role from migration 0002, never at POSTGRES_USER.
    trace_db_url: str = ""

    # How many finished traces may wait for the writer. Beyond this, traces are
    # dropped rather than queued: unbounded memory growth in the serving
    # container is a worse failure than losing training signal. ~50KB per trace
    # at the 48k-char cap, so 500 is a ceiling of roughly 25MB.
    trace_queue_size: int = 500

    # Ceiling on one write, and on opening the pool at startup. Short on
    # purpose: a store this slow is one the writer cannot keep up with anyway,
    # and the queue is a better place to notice that than a stuck connection.
    trace_write_timeout_s: float = 5.0

    # Guards against a client sending an unbounded conversation history that
    # would overrun the model's context window. Characters, not tokens: exact
    # token counting would need the model's tokenizer, and this only has to be
    # a sane upper bound, not a precise one.
    max_total_chars: int = 48_000


settings = Settings()  # type: ignore[call-arg]
