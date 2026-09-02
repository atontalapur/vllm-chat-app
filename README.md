# vllm-chat-app

A self-hosted LLM chat stack you can read end to end. An open-source model served on a
GPU by [vLLM](https://docs.vllm.ai), a FastAPI application layer that owns auth and
validation and logging, a Streamlit chat UI, and Prometheus with Grafana watching all of
it. Four containers, one compose file.

Most tutorials stop at "call the API and print the tokens." The interesting part is
everything between a user pressing enter and a token coming back off a graphics card, so
that is what this repo is built to show.

```
Browser (via SSH tunnel)
   |
   v
ui (Streamlit)                    holds conversation history, renders tokens as they land
   |  POST /chat/stream + X-API-Key
   v
api (FastAPI)                     auth, validation, structured logging, SSE proxy
   |  POST /v1/chat/completions (stream=true)
   v
vllm (OpenAI-compatible server)   PagedAttention, continuous batching
   |
   v
GPU
   ^
   |  scrape /metrics every 15s
metrics (Prometheus + Grafana)    queue depth, TTFT, tokens/sec, KV cache
```

## Status

Built in reviewed steps, then run end to end on real hardware: an RTX 3090 (24GB) on a
rented Ubuntu VM, driver 580.95.05, vLLM 0.28.0, Qwen2.5-7B-Instruct.

What the hardware run measured, rather than assumed:

| Measurement | Result |
|---|---|
| Cold start, container up to serving | 3 min 16 s |
| Weight download | 117 s for 14.19 GiB |
| Model load into VRAM | 14.29 GiB, 123 s |
| `torch.compile` | 20.2 s |
| KV cache after weights | 5.79 GiB, 108,384 tokens |
| Theoretical concurrency at 8k context | 13.23 sequences |
| Configured batch cap | 4 sequences |
| Peak generation throughput under load | ~200 tok/s |

Every metric the Grafana dashboard queries was checked against the live `/metrics`
endpoint. That check was the one thing in this project that could not be done without a
GPU, so it was isolated down to a single command instead of being discovered panel by
panel while a rented box billed by the minute.

Remaining: the demo recording.

## Quick start on a GPU box

You need an NVIDIA GPU with 24GB or more of VRAM. An RTX 3090, RTX 4090, A10G, L4, or
A100 all work. Qwen2.5-7B-Instruct needs roughly 15GB in bf16 and the rest becomes KV
cache.

Two things about rented GPUs are worth knowing before spending money.

The instance has to be a real VM, not a container. Vast.ai and RunPod rent Docker
containers by default, so you SSH into a container with no init manager and
`docker compose` cannot run inside one. Vast.ai's documentation lists Docker Compose
among the features only their VM instances support. Filter for a VM image and expect to
pay roughly double the container rate. It is still under a dollar for a session.

Ask for 70GB of disk or more. Cheap listings default to 10 or 30GB, and the failure lands
partway through a 15GB model download, which is the slowest possible place to find out.

```bash
git clone https://github.com/atontalapur/vllm-chat-app.git
cd vllm-chat-app
cp .env.example .env

# API_KEY and GF_SECURITY_ADMIN_PASSWORD are required and ship empty on purpose
sed -i "s|^API_KEY=.*|API_KEY=$(openssl rand -hex 32)|" .env
sed -i "s|^GF_SECURITY_ADMIN_PASSWORD=.*|GF_SECURITY_ADMIN_PASSWORD=$(openssl rand -hex 16)|" .env

docker compose up -d --build
```

First boot pulls the vLLM image, downloads the weights, and loads them. Budget 5 to 10
minutes. It is not hung, and you can watch it:

```bash
docker compose logs -f vllm
docker compose ps            # vllm goes starting, then healthy
```

`api` and `ui` will not start until `vllm` reports healthy. That is
`depends_on: condition: service_healthy` doing its job, not a failure.

Then open the tunnel from your laptop and use it:

```bash
ssh -p <port> -L 8501:localhost:8501 -L 3000:localhost:3000 root@<host>
```

The chat is at `http://localhost:8501` and the dashboard is at `http://localhost:3000`,
where you log in as `admin` with the password you generated.

The full walkthrough, including fresh-VM setup for Docker and the NVIDIA container
toolkit, is in [docs/commands.md](docs/commands.md).

## Local development without a GPU

vLLM requires CUDA and cannot run on Apple Silicon. The compose file's
`deploy.resources.reservations.devices` block is rejected by the Docker daemon on a Mac,
and no virtual GPU exists that synthesizes CUDA where there is no NVIDIA hardware.
NVIDIA vGPU partitions physical cards inside a datacenter hypervisor, which is a
different thing. Renting a cloud GPU is the practical equivalent.

The application layer only speaks the OpenAI-compatible protocol, so anything
implementing that protocol can stand in for vLLM. On a Mac, [Ollama](https://ollama.com)
is Metal-accelerated and does exactly that:

```bash
ollama serve
ollama pull llama3.1

cp .env.example .env      # set API_KEY, then uncomment the two local lines
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build
```

In `.env`:

```
VLLM_BASE_URL=http://host.docker.internal:11434/v1
LOCAL_MODEL_ID=llama3.1:latest
```

Open `http://localhost:8501`. No tunnel needed, since it is already your machine.

Pick a model that does not do visible reasoning. Reasoning models stream deltas carrying
a `reasoning` field with empty `content`, so a content-only UI renders nothing until the
reasoning finishes. That looks exactly like a broken stream, and it cost an hour to
diagnose the first time.

The overlay drops the `vllm` service outright with Compose's `!reset` tag rather than
trying to disable its GPU reservation. Setting `deploy: {}` does not work: Compose merges
those maps deeply, the nested device reservation survives, and the daemon rejects the
container before it starts.

Everything above the serving layer develops and verifies on a laptop, including auth,
validation, logging, streaming, the UI, the tests, and CI. The queue-depth and KV-cache
behaviour is the only part that needs real hardware.

## How the pieces fit

### vllm

The official `vllm/vllm-openai` image, pinned to a tag, with no custom Dockerfile. There
is nothing to add. vLLM already exposes an OpenAI-compatible server, and a wrapper around
it would only be a place for bugs to live.

The flags that matter:

`--max-num-seqs=4` caps how many sequences run in a single forward pass. This is set
deliberately low. A realistic value would be 32 or higher, but at 32 a small burst never
queues and the dashboard stays flat, which hides the exact behaviour the project exists
to show. The startup log reports that the KV cache could hold 13.23 concurrent sequences
at this context length, so the queue that forms during a load test is a configuration
limit and not a memory limit. That distinction is the whole demonstration.

`--max-model-len=8192` caps the context window. Qwen2.5-7B defaults to 32k, which on a
24GB card leaves almost nothing for KV cache and can stop vLLM from starting at all. 8192
is plenty for chat and leaves 5.79 GiB of cache.

`--gpu-memory-utilization=0.90` is vLLM's own default, kept explicit so it is visible and
tunable when a box has something else holding VRAM.

`--disable-access-log-for-endpoints=/health,/metrics` stops a Prometheus scrape every
5 seconds from drowning the logs.

The service also gets `ipc: host`, which vLLM needs for its shared memory, and a named
volume for the Hugging Face cache so weights survive a restart.

### api

FastAPI, and the only component that talks to vLLM.

Auth is a single shared key in the `X-API-Key` header, compared with
`secrets.compare_digest` so a wrong key cannot be recovered by timing. One static key is
the right scope for a demo with one trusted client and the wrong scope for anything with
real users, which the limits section says plainly.

Auth is applied per route, not globally. `/health` and `/metrics` are deliberately open.
Putting the key on everything would have silently broken two things: Prometheus scraping
would 401 and leave every panel empty with no error anywhere, and the compose healthcheck
would fail, so `api` would never report healthy and `ui` would never start. Both failures
look like something else entirely.

Validation is Pydantic with real bounds on `max_tokens` and `temperature`, plus a
character cap on the whole conversation that returns 413 rather than forwarding an
oversized prompt to the GPU.

The streaming proxy uses `httpx.AsyncClient` with `connect=10, read=None`. A read timeout
would kill long generations mid-stream. The handler pulls the first chunk with
`await anext(stream)` before returning the response, so an upstream that is down produces
a real 502 instead of a 200 with an empty body. Failures after the first byte cannot
change the status code, so those are emitted as a structured SSE error event the UI knows
how to render.

### ui

Streamlit, calling the api service and never vLLM directly, so the auth and validation
and logging layer is always in the path. Conversation history lives in
`st.session_state` and is trimmed to a fixed number of turns before each request.

It fails fast on a missing API key at import rather than presenting a chat box that
cannot work, and it maps upstream status codes to specific messages: 401 says the key is
wrong, 413 says the conversation is too long, 502 says the model server is unreachable. A
generic "something went wrong" would be easier to write and useless to debug.

### metrics

Prometheus scrapes both vLLM and the api. Grafana's datasource and dashboard are both
provisioned from files on disk, so `docker compose up` produces a working dashboard with
nothing to click. A dashboard configured by hand fails silently on a fresh machine, since
every panel simply renders empty and there is no error to chase.

Five panels:

| Panel | Query subject | What it shows |
|---|---|---|
| Request queue depth | `num_requests_running`, `num_requests_waiting` | continuous batching, made visible |
| Time to first token | `time_to_first_token_seconds_bucket` | p50 and p95, rising under queueing |
| Token throughput | `generation_tokens_total`, `prompt_tokens_total` | tokens per second, generation and prefill |
| GPU KV cache utilisation | `kv_cache_usage_perc` | PagedAttention's block accounting |
| Application layer request rate | `http_requests_total` | 2xx, 4xx and 5xx from FastAPI |

The KV cache panel queries `vllm:kv_cache_usage_perc or vllm:gpu_cache_usage_perc`
because vLLM renamed that metric between versions, and the `or` operator makes the panel
survive either name.

## Watching it work

A single chat session never queues, so the dashboard stays flat and the interesting
behaviour is invisible. Force a queue:

```bash
docker compose exec api python3 /app/loadtest.py --concurrency 12 --requests 150 --max-tokens 400
```

Run it inside the api container. The api publishes no host port, so
`http://localhost:8080` does not resolve from the box's shell, and the script reads
`API_KEY` from the container's own environment, so passing `"$API_KEY"` from your shell
would expand to an empty string.

Set concurrency above `VLLM_MAX_NUM_SEQS` so more requests arrive than there are slots.
On the queue depth panel, `waiting` climbs while `running` pins at the cap. Throughput
should hold roughly flat through the burst instead of collapsing, which is continuous
batching absorbing concurrency instead of serialising it.

Use a run long enough to produce a plateau rather than a spike. A 30 second burst inside
a 15 minute window draws a narrow spike that cannot demonstrate flatness, and it leaves
the latency histogram with so few samples that Grafana draws a straight line between two
distant points. That line is interpolation, not measurement.

## Security model

No service is published to `0.0.0.0`. On a public-IP GPU box that would mean anyone who
finds the address can hit vLLM directly, bypassing the entire auth and validation layer,
and burn a GPU you are paying for by the minute. Streamlit has no login of its own, so
publishing it would be the same problem behind a nicer interface.

`docker compose ps` on a running stack shows the boundary directly. `vllm` and
`prometheus` show a bare `8000/tcp` and `9090/tcp` with no host binding at all, while
`ui` and `grafana` bind to `127.0.0.1` only. Everything is reached over an SSH tunnel.

`.env` is gitignored and never committed. `.env.example` ships with `API_KEY` and
`GF_SECURITY_ADMIN_PASSWORD` empty, and compose uses `${VAR:?}` so an empty value stops
startup with a readable message instead of booting a stack with no password. That guard
is also why CI supplies throwaway values for its compose validation job.

## Tests and CI

36 tests, 29 for the api layer and 7 for the UI. Five CI jobs on every push, covering
ruff, mypy in strict mode, both test suites, both image builds, and compose validation.

The auth tests include one that cannot be written with an HTTP client, because HTTP
clients refuse to send the header that triggers it. It calls the dependency directly:

```python
async def test_non_ascii_key_is_rejected_not_crashed() -> None:
    with pytest.raises(HTTPException) as exc:
        await require_api_key("caf\xe9")
    assert exc.value.status_code == 401
```

That test exists because of a real bug, described below.

## What broke, and what it taught

A repo that only shows the finished state teaches nothing about how it got there.

A wrong API key returned HTTP 500 instead of 401. `secrets.compare_digest` raises
`TypeError` on `str` arguments containing non-ASCII characters, with the message
"comparing strings with non-ASCII characters is not supported". Starlette decodes inbound
headers as latin-1 per the ASGI specification, so a client sending a raw high byte in
`X-API-Key` produced a decoded string that `compare_digest` refused to compare, and the
exception escaped as a 500 with a traceback. An attacker probing auth learns more from a
500 than from a 401. The fix compares bytes instead of strings, which carries no such
restriction and is still constant time. It was caught in code review and confirmed over a
raw socket before being fixed.

`deploy: {}` does not disable a GPU reservation. Compose merges override files deeply, so
the nested `devices` list survived and the daemon kept rejecting the container on a Mac.
Compose 2.24 added the `!reset` tag for this case, and the local overlay uses it to drop
the whole `vllm` service.

A YAML file can be broken by a colon inside a shell default. `${API_KEY:?generate with:
openssl rand -hex 32}` fails to parse, because `: ` inside an unquoted scalar starts a
mapping. Quoting the value fixes it.

The runbook was wrong in a way that would have cost real money. An earlier version told
you to run `docker compose up` on a standard Vast.ai instance. Those are containers, and
Compose is a VM-only feature there. The failure would have arrived after paying for a
15GB download.

Documentation drifts from code silently. Three load-test commands in this repo were wrong
at various points. One pointed at `/tmp/loadtest.py` when compose mounts the file at
`/app/loadtest.py`. One passed `--api-key "$API_KEY"` from a shell that never reads
`.env`. One targeted `http://localhost:8080` from a host where that port is deliberately
unpublished. All three would have failed at the exact moment of recording a demo.
Commands in documentation are code, and nothing in CI checks them.

## Known limits

Worth being direct about, since the demo does not prove any of it.

One shared static API key, with no per-user identity, no rotation, and no revocation.
This is the first thing that would need to change for anything real.

No rate limiting and no quotas. A single client can occupy every batch slot indefinitely.

No cost ceiling. Nothing in the stack stops a runaway loop from generating tokens until
the rented box is stopped by hand.

One model on one GPU, with no tensor parallelism and no horizontal scaling. The compose
file assumes a single machine.

Load has been synthetic and brief. There is no evidence here about sustained traffic,
memory behaviour over hours, or recovery from a mid-flight GPU fault.

Conversation history lives in browser session state, so it is lost on refresh and never
shared across devices.

## Cost

The box bills by the minute, so shutdown is part of the procedure.

| Action | Command | Effect |
|---|---|---|
| Start | `docker compose up -d` | first boot 5 to 10 minutes |
| Stop containers | `docker compose down` | box still billing |
| Free the disk | `docker compose down -v` | deletes cached weights, next start re-downloads 15GB |
| Stop paying | destroy or stop the instance in the provider console | the only step that stops billing |

Volume behaviour differs by provider, and getting it wrong means downloading the model
again. On Vast, destroying an instance destroys its disk. RunPod network volumes survive
a stop. On AWS, EBS survives a stop and dies on a terminate, and a stopped instance still
bills for storage but not for the GPU. Spot and preemptible instances can lose the volume
at any time.

A verification session on a community RTX 3090 costs well under two dollars, including
the first download.

### Gated models

Qwen2.5-7B-Instruct is ungated, so nothing extra is needed. To serve a gated model such
as Llama, create a Hugging Face account, accept the license on the model page, generate a
read token, and put it in `.env` as `HF_TOKEN`. vLLM pulls and caches the weights itself.

## Documentation

- [Architecture](docs/architecture.md), request flow, failure behaviour, security boundary
- [GPU box runbook](docs/gpu-box-runbook.md), provision, verify, record, tear down
- [Command reference](docs/commands.md), every command, grouped by where you run it
- [Published pages](docs/artifacts.md), hosted companions to the docs above

## License

MIT
