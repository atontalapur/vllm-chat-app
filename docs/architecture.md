# Architecture

Four services in the request path, plus a trace store and a metrics pair beside it. The
interesting part is not any one of them — it is what happens to a single request as it
crosses all of them.

## Request flow

```
  Browser
     |  HTTP over an SSH tunnel (nothing is published to the internet)
     v
+----------------------------------------------------------------+
|  ui — Streamlit                                    :8501        |
|  - holds conversation history in st.session_state               |
|  - trims to MAX_TURNS before sending, so the prompt cannot       |
|    grow until it overruns the context window                     |
|  - renders SSE chunks incrementally via st.write_stream          |
+----------------------------------------------------------------+
     |  POST /chat/stream          header: X-API-Key
     v
+----------------------------------------------------------------+
|  api — FastAPI                                     :8080        |
|                                                                  |
|   1. auth        X-API-Key, compared with secrets.compare_digest |
|                  -> 401                                          |
|   2. validate    Pydantic ChatRequest: role, content, max_tokens, |
|                  temperature bounds                 -> 422       |
|   3. size guard  total conversation characters      -> 413       |
|   4. proxy       httpx stream to the model server                |
|                  connect timeout 10s, no read timeout            |
|                  first chunk pulled before responding, so a      |
|                  connect failure is a real 502 rather than a     |
|                  200 whose body immediately errors  -> 502       |
|   5. log         one JSON line: request_id, path, status,        |
|                  latency_ms                                      |
|   6. trace       accumulate response text + logprobs while       |
|                  forwarding, then hand the finished trace to a   |
|                  bounded queue after the stream closes           |
|                  -> never blocks, never fails the request        |
+----------------------------------------------------------------+
     |  POST /v1/chat/completions   {"stream": true, "logprobs": true}
     v
+----------------------------------------------------------------+
|  vllm — vLLM OpenAI-compatible server              :8000        |
|  - PagedAttention: KV cache in blocks, allocated on demand       |
|  - continuous batching: new requests join the running batch      |
|    between decode steps rather than waiting for it to drain      |
|  - exports /metrics                                              |
+----------------------------------------------------------------+
     |
     v
    GPU
```

Tokens stream back up the same path. Nothing buffers a full response at any hop —
verified by measuring per-chunk arrival times, not by inspection.

Off to the side, and never in the path:

```
+----------------------------------------------------------------+
|  postgres — trace store                            :5432        |
|  - one row per completed request: messages, response,           |
|    mean/min logprob, token count, finish reason                 |
|  - written by a background task as api_writer, which holds      |
|    INSERT and nothing else                                      |
|  - read by the curation and eval jobs, never by the api         |
+----------------------------------------------------------------+
```

This is the raw material the fine-tuning loop learns from. See
[the trace store](trace-store.md) for what is captured, how it fails, and how to watch
it.

## Why the app layer exists

vLLM already serves an OpenAI-compatible API. The `api` service adds no capability the
model server lacks, and that is the point: it is where authentication, validation,
logging, and error shaping live in a real system. Routing the UI through it means those
concerns are exercised on every message instead of being decorative.

The UI never talks to vLLM directly. If it did, the app layer would be bypassable and
therefore untrustworthy.

## Failure behaviour

| Failure | Detected at | Response | What the user sees |
|---|---|---|---|
| No/wrong API key | auth dependency | 401 | "Authentication failed" banner |
| Malformed request | Pydantic | 422 | "rejected as invalid" banner |
| Conversation too large | size guard | 413 | "Start a new chat" banner |
| Model server unreachable | connect timeout | 502 | "model server is unavailable" |
| Model server 5xx | status check | 502 | "model server is unavailable" |
| Failure mid-stream | stream iteration | SSE error event | "connection lost mid-stream" |
| Model still loading | Compose health gate | api/ui do not start | stack not yet up |
| Trace store down, slow, or absent | background writer | none — request unaffected | nothing; the trace is dropped and counted |

The rule behind the table: **never hang, never fail silently.** A hung request is
indistinguishable from a slow model, so the user waits forever with nothing to act on.

The last row is the exception that proves it, and the reason the pipeline is safe to put
in the serving path at all. Everything above the line fails loudly to the user because
the user can act on it. A lost trace is not something the user can act on, so it fails
silently to them and loudly to the operator: `trace_write_failures_total` and
`trace_drops_total` on the dashboard. Inverting that — failing a chat because a training
pipeline is unhealthy — would trade a working conversation for a row in a table.

## Startup ordering

```
vllm (healthcheck: /health, 5m grace + 40 x 15s)     postgres (pg_isready)
  |  condition: service_healthy                        |  condition: service_healthy
  |                                                    v
  |                                                  db-migrate (one-shot, exits 0)
  |  condition: service_completed_successfully         |
  +<---------------------------------------------------+
  v
api (healthcheck: /health, liveness only)
  |  condition: service_healthy
  v
ui

prometheus, grafana — no ordering dependency; they scrape whatever is reachable
```

`db-migrate` applies `pipeline/db/migrations/*.sql` in order, records each file in
`schema_migrations`, and exits. A migration that fails rolls back and exits non-zero, which
keeps `api` from starting against a half-applied schema. Re-runs apply nothing. The
migration finishes in seconds, so it never adds to the cold start; vllm is always the
long pole.

The gate holds for `docker compose up` only. After a host reboot the daemon restarts
`api` and `postgres` under their `restart: unless-stopped` policies without re-running
`db-migrate`, so a data volume replaced between runs comes back empty and unnoticed until
the next `compose up`. The trace writer is fail-open for the same reason: a store that is
down or behind must cost traces, never chat.

One consequence of that gate is worth naming. The api opens its trace-store connection at
startup and, if the store is unreachable then, serves chat with tracing disabled until it
is restarted. Compose ordering makes that unlikely — `postgres` and `db-migrate` both
complete first — and an outage *after* startup is recovered automatically by the
connection pool. The boot log says which state it is in.

Cold start is 5-10 minutes: the image pull plus roughly 15GB of weights plus load time.
The healthcheck budget is deliberately generous, because a still-loading model is not an
unhealthy one, and a tight budget would fail the whole stack while everything was working
correctly.

`api`'s own `/health` reports liveness only and deliberately does not probe vLLM.
Startup ordering is Compose's job; conflating the two would make the api report unhealthy
— and get restarted — every time the model server hiccuped.

## Security boundary

Nothing binds to `0.0.0.0`. On a public-IP GPU box that would mean:

- **vLLM published** — anyone could hit the model directly, bypassing auth, validation,
  and logging entirely. The app layer would be decorative rather than load-bearing.
- **UI published** — the Streamlit page has no login of its own, so anyone who found the
  address would get free use of a GPU billed by the minute.

`ui` and `grafana` bind to `127.0.0.1`; `api`, `vllm`, `postgres`, and `prometheus` publish
nothing at all and are reachable only on the internal Compose network. Access is via SSH
tunnel.

`/health` and `/metrics` are exempt from the API key. The Compose healthcheck and
Prometheus have no reason to hold the application's secret, and 401ing them would break
startup ordering and leave every dashboard panel silently empty.
