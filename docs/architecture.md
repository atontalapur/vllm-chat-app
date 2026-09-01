# Architecture

Four services. The interesting part is not any one of them — it is what happens to a
single request as it crosses all four.

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
+----------------------------------------------------------------+
     |  POST /v1/chat/completions   {"stream": true}
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

The rule behind the table: **never hang, never fail silently.** A hung request is
indistinguishable from a slow model, so the user waits forever with nothing to act on.

## Startup ordering

```
vllm (healthcheck: /health, 5m grace + 40 x 15s)
  |  condition: service_healthy
  v
api (healthcheck: /health, liveness only)
  |  condition: service_healthy
  v
ui

prometheus, grafana — no ordering dependency; they scrape whatever is reachable
```

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

`ui` and `grafana` bind to `127.0.0.1`; `api`, `vllm`, and `prometheus` publish nothing at
all and are reachable only on the internal Compose network. Access is via SSH tunnel.

`/health` and `/metrics` are exempt from the API key. The Compose healthcheck and
Prometheus have no reason to hold the application's secret, and 401ing them would break
startup ordering and leave every dashboard panel silently empty.
