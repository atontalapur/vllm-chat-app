# The trace store

Every chat request leaves a row behind. Those rows are the raw material the
fine-tuning loop is built from: Sprint 3 curates them into a training set,
Sprint 4 trains on it, and Sprint 5 decides whether the result is better than
what is being served today. Without this layer the rest of the pipeline has
nothing to learn from.

This is also the first piece of the pipeline that sits **inside the serving
path**, which is what makes it delicate. It is written to a single rule:

> A chat request never waits on the trace store, and never fails because of it.

Everything below follows from that.

---

## The write path

```
  client                api                         queue            worker            postgres
    |                    |                            |                 |                  |
    |  POST /chat/stream |                            |                 |                  |
    |------------------->|                            |                 |                  |
    |                    |  chunk 1 ------------------>|                |                  |
    |<-- chunk 1 --------|  then observe(chunk 1)      |                 |                  |
    |<-- chunk 2 --------|  then observe(chunk 2)      |                 |                  |
    |        ...         |            ...             |                 |                  |
    |<-- [DONE] ---------|                            |                 |                  |
    |                    |  build() + submit() ------->|                 |                  |
    |  (response over)   |  returns instantly          |--- get() ------>|                  |
    |                    |                            |                 |--- INSERT ------>|
```

Three deliberate choices in that picture:

**Accumulation happens after each yield, not before.** `stream_chat` hands the
chunk to the client and only then feeds it to the accumulator. Resuming an
async generator runs the code after `yield` when the consumer asks for the next
line, so the parse happens in time already spent waiting on the upstream
socket. `api/tests/test_tracing.py::test_chunk_is_yielded_before_any_trace_work`
pins this: after the first chunk is in the caller's hands, the accumulator has
still seen nothing.

**`submit()` is synchronous.** It is not a coroutine, so no caller can
accidentally `await` it and no amount of database latency can reach the request
path through it. It puts the trace on a bounded queue and returns.

**The queue is bounded, and full means drop.** An unbounded queue turns a slow
store into unbounded memory growth in the api container — a worse failure than
losing traces, because the api is in the serving path and the trace store is
not. At `TRACE_QUEUE_SIZE` the newest trace is discarded and counted.

---

## What is captured

One row per completed request, in `traces`
(`pipeline/db/migrations/0001_traces.sql`):

| Column | Source | Notes |
|---|---|---|
| `request_id` | `X-Request-ID` | Primary key, and the join key to the JSON log line for the same request |
| `created_at` | `now()` | |
| `model` | `MODEL_ID` | Becomes the adapter name once Sprint 5 promotes one, so base and tuned traces are distinguishable |
| `messages` | the request | Full conversation as sent upstream, `jsonb` |
| `response` | the stream | Assistant text, accumulated from `delta.content` |
| `mean_logprob` | the stream | Confidence proxy, see below. Null when `CAPTURE_LOGPROBS=false` |
| `min_logprob` | the stream | Worst single token |
| `n_tokens` | the stream | Tokens that carried a logprob |
| `finish_reason` | the stream | `stop` or `length` |
| `judge_score` | — | Null until Sprint 2 scores it |
| `curation_status` | — | Null until Sprint 3 processes it. Null is how the curation query finds new work |

### What is deliberately *not* written

| Case | Why |
|---|---|
| 401, 413, 422 | Rejected before reaching the model; there is no response to trace |
| 502 (upstream unreachable) | No response either. An empty row would be noise in the curation query |
| Client disconnected mid-stream | A partial answer the user never saw, with no `finish_reason`, is not a training example |
| The prompt's own logprobs | `prompt_logprobs` is rejected by vLLM when streaming, and is not wanted |

---

## Reading the confidence proxy

`mean_logprob` is the mean of the per-token logprobs, which is the log of the
geometric-mean token probability. So `exp(mean_logprob)` reads as an average
per-token confidence between 0 and 1:

| `mean_logprob` | `exp()` | Reading |
|---|---|---|
| -0.1 | 0.90 | fluent, unhesitant |
| -0.5 | 0.61 | the starting threshold Sprint 3 will select below |
| -1.0 | 0.37 | visibly uncertain |

`min_logprob` sits beside it because the mean hides a single badly chosen token
in a long fluent answer. Both are kept, and Sprint 6 tunes thresholds on the
observed distribution rather than on these guesses.

**What this signal cannot do.** It measures hesitation, not correctness. The
worked example in `docs/spikes/s0-3-logprob-shape.md` is the whole argument: the
model answered a trick question about the 2011 World Cup final with the wrong
number at a mean confidence of 0.80. Confidently wrong looks exactly like
confidently right here. That gap is what the Sprint 2 judge exists to close;
logprobs are the cheap pre-filter, not the gate.

`mean_logprob` is `NULL`, never `0.0`, when nothing carried a logprob. The
difference matters: Sprint 3 selects traces *below* a threshold, and a `0.0`
would read as perfect confidence and silently exclude every such row from
curation forever.

---

## Failure behaviour

The store has no failure mode that reaches the user. Each one costs traces and
increments a counter instead.

| What happened | Counter | Effect on chat | Effect on traces |
|---|---|---|---|
| No `TRACE_DB_URL` | `trace_drops_total{reason="disabled"}` | none | none written |
| Store unreachable at api startup | `trace_drops_total{reason="disabled"}` | none | none written until the api restarts |
| Store down while running | `trace_write_failures_total{reason="database"}` | none | lost while down; the pool reconnects on its own |
| Store slower than `TRACE_WRITE_TIMEOUT_S` | `trace_write_failures_total{reason="timeout"}` | none | that trace lost |
| Traffic outrunning the writer | `trace_drops_total{reason="queue_full"}` | none | newest traces lost |
| Anything unforeseen in the write | `trace_write_failures_total{reason="unexpected"}` | none | that trace lost, worker survives |
| A malformed or unexpected SSE chunk | — | none | that chunk's tokens not counted |

Two of those are worth expanding, because they are the ones that bite.

**The worker must outlive a failed write.** An exception escaping the drain
loop would kill the task, and every subsequent trace would be dropped for the
life of the process with nothing in the logs to say so. Every write is wrapped,
including the `except Exception` catch-all, and
`test_database_error_is_counted_and_the_worker_survives` proves the next write
still lands after a failure.

**A store that appears late stays unused.** `start()` waits for the first
connection so the boot log says plainly whether tracing is on. The cost is that
a database which only becomes reachable *after* the api started stays disabled
until the api restarts. That trade is fine here because Compose already orders
`postgres` and `db-migrate` ahead of `api`; the case that actually happens in
practice is an outage *after* startup, which the pool recovers from by itself.

---

## Watching it

Two panels on the Grafana dashboard, beneath the serving ones.

**Trace capture: written vs lost.** `written` should track the request rate on
the panel above it. The two loss lines should sit flat at zero. A gap between
them is training data going missing, which no HTTP panel can show you: a request
that traces nothing is still a perfectly healthy 200.

**Trace writer queue depth.** Normally at or near zero. A sawtooth during a load
burst is expected. A depth that climbs and stays up means the writer is falling
behind, and drops begin once it reaches `TRACE_QUEUE_SIZE`.

| Symptom | Likely cause | First thing to check |
|---|---|---|
| `written` flat at zero, `dropped{disabled}` climbing | api never connected | `docker compose logs api \| grep "trace store"` |
| `failed{database}` climbing | store unhealthy or grants wrong | `docker compose logs postgres` |
| `failed{timeout}` climbing | store slow, or `TRACE_WRITE_TIMEOUT_S` too tight | queue depth panel |
| Queue depth climbing, then `dropped{queue_full}` | traffic beyond one writer | raise `TRACE_QUEUE_SIZE`, then look at the store |
| Everything zero including requests | nothing is being asked of the model | not a trace problem |

---

## Privileges

The api connects as `api_writer` (`pipeline/db/migrations/0002_api_writer_role.sql`),
which holds `INSERT` on `traces` and nothing else. Not `POSTGRES_USER`, which
owns the table.

What that buys: a bug in the serving path cannot read other users' conversations
back, cannot overwrite a judge score or a curation decision, and cannot drop the
table the training set is built from.

It has one non-obvious consequence, found by running the real writer against a
real database rather than a mock. The insert cannot name its conflict target:

```sql
-- Needs SELECT on request_id. Fails for api_writer with:
--   ERROR:  permission denied for table traces
INSERT INTO traces (...) VALUES (...) ON CONFLICT (request_id) DO NOTHING;

-- Needs INSERT alone. This is what the api runs.
INSERT INTO traces (...) VALUES (...) ON CONFLICT DO NOTHING;
```

The two are equivalent while the primary key is the table's only unique
constraint. Add a second one and the untargeted form starts swallowing those
conflicts too, and this decision has to be revisited.

CI asserts all of it: that the role exists, that its privilege set is exactly
`INSERT`, that it can insert, and that it cannot select.

---

## Configuration

All in `.env`; see `.env.example`.

| Variable | Default | What it does |
|---|---|---|
| `TRACE_DB_URL` | empty | Where traces go. Empty disables tracing entirely. Compose sets it to the `api_writer` DSN |
| `CAPTURE_LOGPROBS` | `true` | Ask the model server for per-token logprobs. False still writes traces, with a null `mean_logprob` |
| `TRACE_QUEUE_SIZE` | `500` | Traces that may wait for the writer. ~50KB each at the 48k-char request cap, so roughly 25MB at the ceiling |
| `TRACE_WRITE_TIMEOUT_S` | `5.0` | Ceiling on one write, and on the startup connect. libpq treats a connect timeout under 2s as 2s |
| `API_DB_PASSWORD` | — | Password for `api_writer`. Separate from `POSTGRES_PASSWORD`, which owns the table |

Turning tracing off entirely is unsetting `TRACE_DB_URL`. The api logs
`trace store disabled: no DSN configured` at boot and serves chat unchanged.

Costs, if you are deciding whether to leave `CAPTURE_LOGPROBS` on: roughly 120
bytes per token on the vllm→api hop, and nothing on the api→UI hop, because the
api forwards the line either way.

---

## Querying traces

The api's role cannot read. Query as the owner, from inside the Compose network
— `postgres` publishes no port.

```bash
# Traces from the last hour, newest first.
docker compose exec postgres psql -U traces -d traces -c \
  "SELECT request_id, created_at, n_tokens, round(mean_logprob::numeric,3) AS mean_lp,
          finish_reason, left(response, 60) AS preview
     FROM traces WHERE created_at > now() - interval '1 hour'
    ORDER BY created_at DESC LIMIT 20"

# The confidence distribution, which is what Sprint 6 tunes thresholds against.
docker compose exec postgres psql -U traces -d traces -c \
  "SELECT width_bucket(mean_logprob, -3, 0, 12) AS bucket,
          count(*), round(avg(mean_logprob)::numeric, 3) AS avg_lp
     FROM traces WHERE mean_logprob IS NOT NULL GROUP BY 1 ORDER BY 1"

# Truncated responses. Sprint 3 drops these from the training set.
docker compose exec postgres psql -U traces -d traces -c \
  "SELECT count(*) FILTER (WHERE finish_reason = 'length') AS truncated, count(*) AS total
     FROM traces"

# Join a trace to its log line.
docker compose logs api | grep "$(docker compose exec -T postgres psql -U traces -d traces \
  -tAc 'SELECT request_id FROM traces ORDER BY created_at DESC LIMIT 1')"
```

---

## How this was verified

Unit tests mock the pool, because a real database cannot be made to time out or
vanish mid-write on command. That leaves a gap: the SQL itself, the jsonb
adaptation, and the grants are exactly the things a mock will always agree with.
So they were also run for real against the Compose Postgres, end to end,
accumulator through writer to a row read back:

```
 request_id |          model           |                 messages                  |  response  | mean_lp | min_lp  | n_tokens | finish_reason
------------+--------------------------+-------------------------------------------+------------+---------+---------+----------+---------------
 e2e-1      | Qwen/Qwen2.5-7B-Instruct | [{"role": "user", "content": "who won?"}] | India won. | -0.3733 | -0.9000 |        3 | stop
```

That run is what found the `ON CONFLICT` privilege interaction above, and it
found it after the mocked tests were green. CI keeps the same ground covered:
it applies the migrations to a real Postgres and asserts the role's privileges
against `information_schema`.

---

## Known limits

**Conversations are stored in full, in plain text.** Whatever a user types into
the chat is what lands in `messages`. There is no redaction, no retention
policy, and no deletion path. For anything beyond a single-operator demo, that
is the first thing to change.

**No backpressure, by design.** The store never slows the request path down, so
under sustained overload the loss is silent apart from the counters. The
counters are the mitigation.

**One writer.** A single drain task serves the whole process. It has kept up
with everything this stack can produce on one GPU, and the queue-depth panel is
how you would find out if that stopped being true.

**Traces are not deduplicated.** Ten identical questions produce ten rows.
Near-duplicate collapsing is Sprint 3's job (S3-2), deliberately kept out of the
serving path.

**No schema for `judge_score` provenance.** The column records the number, not
which rubric version produced it. Sprint 2 has to add that, or scores from
different rubrics will be silently comparable.

---

## Related

- `docs/spikes/s0-3-logprob-shape.md` — where the chunk shapes and the proxy formula come from
- `docs/architecture.md` — where this sits in the request path
- `docs/pipeline-plan.md` — Sprint 1, and what consumes these rows next
- `api/app/tracing.py`, `api/app/trace_store.py` — the implementation, commented at the same level as this document
