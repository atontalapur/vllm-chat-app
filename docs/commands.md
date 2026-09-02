# Command reference

Every command needed to run this stack, grouped by where you type it.

**The single most common mistake is running a command in the wrong place.** Check first:

```bash
hostname
```

`ubuntu` (or whatever the box is called) means you are on the GPU box. Your Mac's name
means you are local. Nothing in the Docker sections works on a Mac.

---

## 1. SSH

Every rental gets a new IP and port, so put them in variables once per session rather
than retyping. On your Mac:

```bash
export BOX=root@77.104.167.148     # from the SSH icon on the instance card
export BOXPORT=41494
```

### Connect

```bash
ssh -p $BOXPORT $BOX
```

### Connect with the tunnels (this is the one you usually want)

```bash
ssh -p $BOXPORT -L 8501:localhost:8501 -L 3000:localhost:3000 $BOX
```

Then in your browser:

- chat — <http://localhost:8501>
- dashboard — <http://localhost:3000> (`admin` / `GF_SECURITY_ADMIN_PASSWORD` from `.env`)

The tunnel lives only as long as that terminal tab. Close it and both pages go dead.
Nothing is exposed to the internet, which is deliberate: Streamlit has no login, and an
open GPU on a public IP is someone else's free compute at your expense.

If port 3000 is already taken on your Mac, ssh prints `bind: Address already in use`.
Use a different local port; only the left-hand number changes:

```bash
ssh -p $BOXPORT -L 8501:localhost:8501 -L 3001:localhost:3000 $BOX
```

### Leave

`exit`, or Ctrl+D. **This does not stop billing** — see section 7.

### Keys

Vast has no password login. Public key first, or SSH is refused.

```bash
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -C "vast" -f ~/.ssh/id_ed25519
pbcopy < ~/.ssh/id_ed25519.pub
```

Paste it into the Vast console:

- **before renting** — Keys → SSH Keys → + New. Applies to every future instance.
- **already rented** — Instances → the SSH key button on the instance card. Account
  keys do not reach instances that already exist, which is the usual cause of
  `Permission denied (publickey)`.

### Copy files off the box

```bash
scp -P $BOXPORT $BOX:/root/vllm-chat-app/somefile.log .     # note: capital -P
```

---

## 2. First-time setup on a fresh box

Only needed once per rental. A bare Ubuntu VM has neither Docker nor the NVIDIA
container toolkit.

```bash
apt-get update && apt-get install -y git curl
curl -fsSL https://get.docker.com | sh

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
```

If apt fails with `Could not get lock /var/lib/dpkg/lock-frontend`, Ubuntu's
`unattended-upgrades` is holding it. Wait it out instead of forcing anything:

```bash
for i in $(seq 1 60); do
  apt-get install -y nvidia-container-toolkit && break
  echo "locked, retry $i in 10s"; sleep 10
done
```

### The gate

```bash
nvidia-smi                                                        # host sees the GPU
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi   # containers do too
df -h /                                                           # need 70G+ free
```

If the container `nvidia-smi` prints your GPU, everything downstream works. If it does
not, stop and fix that first — nothing else will succeed.

Note the driver's **CUDA Version**. `VLLM_IMAGE_TAG=v0.28.0-cu129` (CUDA 12.9) works on
any r525+ driver. The bare `v0.28.0` tag is CUDA 13.0 and needs r580+.

### Clone and configure

```bash
git clone -b feat/vllm-chat-stack https://github.com/atontalapur/vllm-chat-app.git
cd vllm-chat-app
cp .env.example .env

sed -i "s|^API_KEY=.*|API_KEY=$(openssl rand -hex 32)|" .env
sed -i "s|^GF_SECURITY_ADMIN_PASSWORD=.*|GF_SECURITY_ADMIN_PASSWORD=$(openssl rand -hex 16)|" .env

grep -E '^(API_KEY|GF_SECURITY_ADMIN_PASSWORD)=' .env      # save the Grafana password
```

---

## 3. Running the stack

All from `~/vllm-chat-app` on the box.

| What | Command |
|---|---|
| Start (first time, builds images) | `docker compose up -d --build` |
| Start (later) | `docker compose up -d` |
| Stop, keep cached weights | `docker compose down` |
| Stop and **delete** cached weights | `docker compose down -v` |
| Restart one service | `docker compose restart vllm` |
| Rebuild one service after a code change | `docker compose up -d --build api` |
| What is running | `docker compose ps` |

First boot takes **5-10 minutes**: image pull, then ~15GB of weights, then load. It is
not hung. `api` and `ui` stay down until `vllm` reports healthy — that is the health
gate, not a failure.

Only use `down -v` when you are finished with the box for good. It throws away the
model and the next start re-downloads 15GB.

---

## 4. Watching and verifying

```bash
docker compose logs -f vllm          # follow one service, Ctrl+C to stop following
docker compose logs -f api
docker compose logs --tail=50 ui     # last 50 lines, no follow
docker compose ps                    # STATUS: starting -> healthy
watch docker compose ps              # same, refreshing
```

Health and reachability. These run *inside* containers because neither `vllm` nor `api`
publishes a port to the host:

```bash
# vLLM is serving the model
docker compose exec vllm python3 -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/v1/models').read().decode())"

# the app layer is alive
docker compose exec api python3 -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8080/health').read().decode())"

# the metrics that drive the dashboard
docker compose exec vllm python3 -c "import urllib.request;print(urllib.request.urlopen('http://localhost:8000/metrics').read().decode())" | grep -E '^vllm:(num_requests|generation_tokens|.*cache_usage|time_to_first_token)'
```

Expected metric names, all five dashboard panels:

```
vllm:num_requests_running
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
vllm:generation_tokens_total
vllm:prompt_tokens_total
vllm:time_to_first_token_seconds_bucket
```

Verified present in vLLM 0.28.0. If a future version renames one, the fix is a one-line
edit to the matching `expr` in `metrics/grafana/dashboards/vllm.json`.

Shell inside a container when you need to poke around:

```bash
docker compose exec api sh
docker compose exec vllm bash
```

GPU usage live:

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

---

## 5. The load test

```bash
docker compose exec api python3 /app/loadtest.py --concurrency 12 --requests 60
```

Longer run for a video take:

```bash
docker compose exec api python3 /app/loadtest.py --concurrency 12 --requests 120
```

Two things that look like bugs but are not:

- **No `--api-key` flag.** The script reads `API_KEY` from the api container's own
  environment. Passing `"$API_KEY"` from the box's shell expands to an empty string,
  because `.env` is read by Compose, not by your shell.
- **It must run inside the api container.** The api publishes no host port, so
  `http://localhost:8080` does not resolve from the box's shell.

Before recording, in Grafana: time range **Last 15 minutes**, refresh **5s**. The
default 6-hour window makes a 90-second burst invisible.

What the burst demonstrates, given `--concurrency 12` against `VLLM_MAX_NUM_SEQS=4`:

| Panel | What happens | Why |
|---|---|---|
| Request queue depth | `running` pins flat at 4, `waiting` climbs to ~8 | the batch cap, and the queue behind it |
| Token throughput | holds roughly flat | continuous batching absorbs concurrency instead of serialising it |
| Time to first token | p95 rises, then recovers | requests waiting their turn |
| KV cache | climbs with concurrent sequences | PagedAttention allocates blocks on demand |

If no queue forms, raise `--concurrency` or lower `VLLM_MAX_NUM_SEQS` in `.env` and
`docker compose restart vllm`.

---

## 6. When something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied (publickey)` | key not attached to *this* instance | Instances → SSH key button on the card |
| `bind: Address already in use` | local port taken | change the left-hand port: `-L 3001:localhost:3000` |
| `Could not get lock` on apt | unattended-upgrades | the retry loop in section 2 |
| `nvidia-ctk: command not found` | toolkit install lost the apt lock | rerun the install |
| vLLM exits with a KV-cache error | context window too large for the card | lower `VLLM_MAX_MODEL_LEN` in `.env`, restart |
| vLLM OOMs at startup | other processes hold VRAM | lower `VLLM_GPU_MEMORY_UTILIZATION` to `0.85` |
| Dashboard panels empty | no traffic in the last minute, or wrong time range | send a chat message; set range to Last 15 minutes |
| UI shows a 502 banner | vLLM down or still loading | `docker compose ps`, then `logs -f vllm` |
| Chat page dead in browser | tunnel tab closed | reopen the `-L` ssh command |

`.env` changes only take effect on restart:

```bash
docker compose up -d          # picks up changed env for affected services
docker compose restart vllm   # or just the one
```

---

## 7. Teardown

```bash
docker compose down
```

Then **destroy or stop the instance in the Vast console**. Exiting SSH does not stop
billing. `docker compose down` does not stop billing. Only the console does.

Volume behaviour differs by provider, and getting it wrong costs a 15GB re-download:

- **Vast** — destroying the instance destroys its disk.
- **RunPod** — network volumes survive a stop.
- **AWS** — EBS survives *stop*, dies on *terminate*. A stopped instance still bills for
  storage, not for the GPU.
- **Spot / preemptible** — assume the volume can vanish at any time.

---

## 8. Local development on a Mac (no GPU)

vLLM needs CUDA and cannot run on Apple Silicon. The api only speaks the
OpenAI-compatible protocol, so Ollama stands in for vLLM locally.

```bash
ollama serve                # if not already running
ollama pull llama3.1

cp .env.example .env        # set API_KEY, then uncomment the two local lines
docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --build
```

In `.env`:

```
VLLM_BASE_URL=http://host.docker.internal:11434/v1
LOCAL_MODEL_ID=llama3.1:latest
```

Open <http://localhost:8501>. No tunnel needed — it is already your machine.

Use a **non-reasoning** model. Reasoning models stream deltas with a `reasoning` field
and empty `content`, so a content-only UI renders nothing until reasoning finishes,
which is indistinguishable from a broken stream.

Everything above the serving layer develops and verifies fully on the laptop. The
queue-depth and KV-cache dashboard is the one thing that needs real hardware.

### Tests and checks

```bash
cd api && uv run pytest && uv run mypy app && uv run ruff check .
cd ui  && uv run pytest && uv run ruff check .
docker compose config                  # validate compose without starting anything
```

---

## 9. Git

```bash
git status
git switch -c feat/some-change         # never commit on main
git add -p
git commit
git push -u origin feat/some-change
gh pr create --fill
gh pr checks                           # CI status
```
