# GPU box runbook

Everything in this repo is built and verified except the parts that require an NVIDIA
GPU. This is the checklist for those. Budget one session of roughly 2-3 hours; the box
bills by the minute, so the teardown step is part of the procedure, not an afterthought.

## 1. Provision

**The instance must be a real VM, not a container.** This is the single easiest way to
waste an hour. Vast.ai and RunPod rent *Docker containers* by default: you SSH into a
container, not a machine, and there is no init manager, so `docker compose` cannot run
inside one. Vast.ai's own documentation lists Docker Compose among the things only their
VM instances support.

- **Vast.ai** — use the VM filter and an Ubuntu 22.04/24.04 VM image. Fewer machines
  qualify and boot is slower, but the stack then runs as written.
- **RunPod** — same container limitation.
- **Lambda Labs / AWS g5 / any normal cloud VM** — real VMs, nothing special needed.

Do not pick a template that pre-runs a model server (vLLM, TGI, Ollama,
text-generation-webui). It will start its own server and compete for VRAM.

Other requirements:

- **24GB+ VRAM** — RTX 3090/4090, A10G, L4, or A100. A community-cloud RTX 3090 at
  $0.10-0.15/hr is the cheapest thing that works: Ampere is compute capability 8.6, well
  above vLLM's 7.0 floor, with native bf16. A 2-3 hour session costs well under a dollar.
  Turing (RTX 2080, T4) works but lacks native bf16, so expect fp16 instead.
- **~60GB disk.** This is the trap on cheap listings. Community and interruptible hosts
  often default to 10-30GB, and the failure lands partway through the model download,
  which is the slowest possible place to find out. Disk is usually a cheap add-on;
  confirm it before starting, not after.
- **Prefer on-demand over interruptible** for a recording session. Spot pricing around
  $0.07/hr is real, but a preemption can take the volume with it and cost you a 15GB
  re-download — more expensive in billed minutes than the on-demand premium.
- **Check network speed.** A slow community host turns a 10-minute cold start into 40
  minutes of billed time, which outweighs any GPU-hour saving.
- **NVIDIA Container Toolkit** installed. Most "GPU + Docker" images ship it. Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi
```

If that prints your GPU, everything in this repo will work. If it does not, stop and fix
it — nothing else will succeed until it does.

Check the driver's CUDA version in that output. `VLLM_IMAGE_TAG` defaults to
`v0.28.0-cu129` (CUDA 12.9); if the driver is r580+ you may use the bare `v0.28.0` tag
(CUDA 13.0) instead.

## 2. Start

```bash
git clone <your repo> && cd vllm-chat-app
cp .env.example .env

# Required:
#   API_KEY                     openssl rand -hex 32
#   GF_SECURITY_ADMIN_PASSWORD  openssl rand -hex 16
# Leave VLLM_BASE_URL and LOCAL_MODEL_ID commented out — those are for laptop dev.

docker compose up -d --build
```

**Expect 5-10 minutes before anything is usable.** Image pull, then ~15GB of weights,
then load. This is the step where people assume something has hung. Watch it:

```bash
watch docker compose ps          # vllm: starting -> healthy
docker compose logs -f vllm
```

`api` and `ui` will not start until `vllm` reports healthy. That is the health gate
working, not a failure.

## 3. Verify

```bash
# vLLM is serving
docker compose exec vllm python3 -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/v1/models').read().decode())"

# vLLM metrics exist (these drive four of the five dashboard panels)
docker compose exec vllm python3 -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/metrics').read().decode())" | grep -E '^vllm:(num_requests|generation_tokens|.*cache_usage)'

# the app layer reaches it
docker compose exec api python3 -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8080/health').read().decode())"
```

The `grep` matters: it is the one thing that cannot be checked without a GPU. If the KV
cache metric is named something other than `vllm:kv_cache_usage_perc` or
`vllm:gpu_cache_usage_perc` in your vLLM version, panel 4 will be empty and the fix is a
one-line edit to `metrics/grafana/dashboards/vllm.json`.

## 4. Open the tunnel

From your laptop, not the box:

```bash
ssh -L 8501:localhost:8501 -L 3000:localhost:3000 user@your-gpu-box
```

- chat — http://localhost:8501
- dashboard — http://localhost:3000 (`admin` / your `GF_SECURITY_ADMIN_PASSWORD`)

Nothing is exposed to the internet. This is deliberate: the Streamlit page has no login,
and an open GPU on a public IP is someone else's free compute at your expense.

## 5. Record

Send a few chat messages first and confirm tokens stream in smoothly.

Then, with the chat and dashboard side by side, start the burst:

```bash
docker compose exec api python3 /app/loadtest.py --concurrency 12 --requests 60
```

No `--api-key` flag: the script falls back to the `API_KEY` already in the api
container's environment. Passing `"$API_KEY"` from the host shell instead expands to
an empty string, because `.env` is read by Compose, not by your shell.

Concurrency 12 against the default `VLLM_MAX_NUM_SEQS=4` means three times as many
requests as slots. What to watch, in order of how well it demonstrates the point:

1. **Request queue depth** — `waiting` climbs above zero, `running` pins flat at 4.
   That flat line is the batch size cap; the climbing line is the queue.
2. **Token throughput** — holds roughly flat through the burst. This is the payoff:
   three times the load does not mean a third of the throughput, because continuous
   batching keeps the GPU busy instead of serialising requests.
3. **Time to first token** — p95 rises while queued, then recovers.
4. **KV cache utilisation** — climbs with concurrent sequences, since PagedAttention
   allocates blocks on demand rather than reserving a worst-case block per request.

If the queue never forms, `--max-num-seqs` is too high relative to concurrency. Raise
`--concurrency` or lower `VLLM_MAX_NUM_SEQS` in `.env` and restart `vllm`.

## 6. Tear down

```bash
docker compose down          # stops containers, keeps cached weights
```

Then **stop or terminate the instance in your provider's console.** `docker compose
down` does not stop the billing.

- **RunPod** — network volumes persist across stops. Safe to stop.
- **AWS** — EBS persists if you *stop*; *terminating* destroys it unless the volume is
  marked to survive. Stopped instances still bill for storage, not for the GPU.
- **Spot/preemptible** — may lose the volume entirely; assume a 15GB re-download.

`docker compose down -v` also deletes the cached weights. Only use it if you are
finished with the box, otherwise the next start re-downloads everything.
