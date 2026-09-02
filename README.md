# vllm-chat-app

A self-hosted LLM chat stack: an open-source model served on GPU by
[vLLM](https://docs.vllm.ai), a FastAPI application layer, a Streamlit chat UI, and
Prometheus + Grafana watching the whole thing — packaged with Docker Compose.

The point of the project is the **request path**: what actually happens between a user
typing a message and tokens coming back off a GPU.

```
Browser (via SSH tunnel)
   |
   v
ui (Streamlit)                    holds conversation history, streams tokens in
   |  POST /chat/stream + X-API-Key
   v
api (FastAPI)                     auth, validation, structured logging, SSE proxy
   |  POST /v1/chat/completions (stream=true)
   v
vllm (OpenAI-compatible server)   paged attention, continuous batching
   |
   v
GPU
   ^
   |  scrape /metrics every 15s
metrics (Prometheus + Grafana)    queue depth, TTFT, tokens/sec, KV-cache %
```

## Status

Built in reviewed steps, then run end to end on real hardware: an RTX 3090 (24GB)
under vLLM 0.28.0, driver 580.95.05.

- [x] vLLM service (pinned image, GPU reservation, health-gated, weights cached)
- [x] CI — lint, types, tests, image builds, compose validation
- [x] FastAPI application layer (auth, validation, logging, streaming proxy)
- [x] Streamlit UI
- [x] Compose wiring for all four services, plus a GPU-free local overlay
- [x] Prometheus + Grafana, dashboard and datasource provisioned as code
- [x] Concurrent load script
- [x] Verified on GPU — 36 tests green, cold start to serving in ~3 minutes, all six
      dashboard metrics confirmed present in vLLM 0.28.0
- [ ] Demo recording

## Requirements

- An NVIDIA GPU box with **24GB+ VRAM** (A10G / L4 / A100-40GB class; an AWS
  `g5.xlarge` or the RunPod/Lambda equivalent). Qwen2.5-7B-Instruct needs ~16GB in
  bf16.
- **~60GB free disk.** The vLLM image plus ~15GB of weights plus the Prometheus and
  Grafana images overrun the 30-50GB default boot volume some cheaper rental tiers
  ship — and it fails *partway through the model download*, which is the slowest
  possible place to discover it.
- Docker with the [NVIDIA Container
  Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed (`docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi`
  should print your GPU).

## Setup

```bash
cp .env.example .env
# Required: API_KEY and GF_SECURITY_ADMIN_PASSWORD.
#   openssl rand -hex 32
# HF_TOKEN only if you switch to a gated model.

docker compose up -d --build
```

First boot pulls the image and downloads ~15GB of weights: **expect 5-10 minutes**
before the service reports healthy. That's normal, not a hang. Watch it:

```bash
docker compose ps                      # STATUS goes  starting -> healthy
docker compose logs -f vllm
```

Verify the OpenAI-compatible endpoint once it's healthy (run on the box — vLLM has no
published port, by design):

```bash
docker compose exec vllm python3 -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/v1/models').read().decode())"
```

Then open the tunnel from your laptop and use it:

```bash
ssh -L 8501:localhost:8501 -L 3000:localhost:3000 user@your-gpu-box
```

- chat UI — http://localhost:8501
- dashboard — http://localhost:3000 (user `admin`, password from `.env`)

## Watching it work

A single chat session never queues, so the dashboard stays flat and the interesting
behaviour is invisible. Force a queue:

```bash
docker compose exec api python3 /app/loadtest.py --concurrency 12 --requests 60
```

Run it **inside the api container**, not on the box's shell: the api has no published
port, so `http://localhost:8080` does not resolve from the host. The script also picks
up `API_KEY` from the container's own environment, so no `--api-key` flag is needed —
passing `"$API_KEY"` from the host shell expands to an empty string, because `.env` is
read by Compose, not by your shell.

Set `--concurrency` above `VLLM_MAX_NUM_SEQS` (default 4) so more requests arrive than
there are slots. On the **Request queue depth** panel, `waiting` climbs above zero while
`running` pins at the cap, then drains. **Token throughput** should hold roughly flat
through the burst rather than collapsing — that is continuous batching absorbing
concurrency instead of serialising it, and it is the thing worth recording.

## Dashboard

Five panels, provisioned from `metrics/grafana/dashboards/vllm.json`:

| Panel | Shows |
|---|---|
| Request queue depth | `running` vs `waiting` — continuous batching, visible |
| Time to first token | p50 / p95, rises under queueing |
| Token throughput | generation and prompt tokens per second |
| GPU KV cache utilisation | PagedAttention's block accounting |
| Application layer request rate | 2xx / 4xx / 5xx from the FastAPI layer |

Both the dashboard and its Prometheus datasource are provisioned from disk, so
`docker compose up` yields a working dashboard with nothing to click. A dashboard whose
datasource was configured by hand fails silently on a fresh machine — every panel simply
renders empty.

### Gated models

Qwen2.5-7B-Instruct is ungated, so nothing extra is needed. To serve a gated model
(Llama, etc.): create a Hugging Face account, accept the model's license on its model
page, generate a **read** access token, and put it in `.env` as `HF_TOKEN`. vLLM pulls
and caches the weights itself — you never download them by hand.

## Local development (no GPU)

vLLM requires CUDA, so it cannot run on a Mac — `deploy.resources.reservations.devices:
driver: nvidia` is rejected by the daemon, and no "virtual GPU" exists that synthesizes
CUDA where there is no NVIDIA hardware. (NVIDIA vGPU partitions *physical* cards in a
datacenter hypervisor; a rented cloud GPU instance is the practical equivalent.)

The application layer only speaks the **OpenAI-compatible protocol**, so anything that
implements it can stand in for vLLM during development. On Apple Silicon,
[Ollama](https://ollama.com) is Metal-accelerated and does exactly that:

```bash
ollama serve                      # if not already running
ollama pull llama3.1              # or any model you prefer

cp .env.example .env              # set API_KEY, then uncomment the two lines below
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build
```

In `.env`:

```
VLLM_BASE_URL=http://host.docker.internal:11434/v1
LOCAL_MODEL_ID=llama3.1:latest
```

The overlay drops the `vllm` service entirely (`!reset`) rather than trying to disable
its GPU reservation — `deploy: {}` does not clear a nested device reservation, and the
daemon rejects the container before it starts. Open http://localhost:8501.

The api reaches the host from inside its container via `host.docker.internal`. Leave
`VLLM_BASE_URL` unset in production and the api defaults to the in-compose `vllm`
service.

**Pick a non-reasoning model locally.** Reasoning models (qwen3.5, etc.) stream deltas
carrying a `reasoning` field with empty `content`, so a content-only UI renders nothing
until reasoning completes — which looks identical to a broken stream and will waste an
hour of your life.

**What this does not cover:** Ollama is not vLLM. No PagedAttention, no continuous
batching, and a completely different `/metrics` shape. The queue-depth and KV-cache
dashboard — the centerpiece of the project — is only demonstrable on the real GPU box.
Everything above the serving layer (auth, validation, logging, streaming, UI, CI,
tests) develops and verifies fully on your laptop.

## Security model

**No service is published to `0.0.0.0`.** On a public-IP GPU box that would mean anyone
who finds the address can (a) hit vLLM directly and bypass the entire auth/validation
layer, and (b) burn a GPU you are paying for by the minute. Reach the UIs over an SSH
tunnel instead:

```bash
ssh -L 8501:localhost:8501 -L 3000:localhost:3000 user@your-gpu-box
```

Then open `http://localhost:8501` (chat) and `http://localhost:3000` (Grafana) in your
own browser. This is also how the dashboard gets screen-recorded for the demo.

## Ops and cost

The box bills by the minute. Treat shutdown as part of the run.

| Action | Command | Note |
|---|---|---|
| Start | `docker compose up -d` | first boot 5-10min |
| Stop containers | `docker compose down` | box still billing |
| Free the disk | `docker compose down -v` | **deletes cached weights** — next start re-downloads 15GB |
| Stop paying | stop/terminate the instance in your provider console | see below |

**Stop vs terminate**, because volume behavior differs and getting it wrong means
re-downloading the model:

- **RunPod** — network volumes persist across pod stops. Safe to stop.
- **AWS** — EBS persists if you **stop** the instance; **terminating** destroys it
  (unless the volume is explicitly marked to survive). Stopped instances still bill for
  EBS storage, just not for the GPU.
- **Spot / preemptible** — some configurations lose the volume entirely. Assume a
  re-download after any interruption.

Rough budget: allow ~2-4 GPU-hours per build session, plus ~15-30 minutes of pure
download time on the very first boot.

## Documentation

- [Architecture](docs/architecture.md) — request flow, failure behaviour, security boundary
- [GPU box runbook](docs/gpu-box-runbook.md) — provision, verify, record, tear down
- [Command reference](docs/commands.md) — every command, grouped by where you run it

## License

MIT
