# S0-1: runtime LoRA loading on vLLM v0.28.0

**Jira:** CFT-8. **Status:** desk research done, GPU confirmation pending.

## Question

Does the pinned image (`vllm/vllm-openai:v0.28.0-cu129`) let us load a LoRA adapter into a
*running* server and serve it under its own model name? Sprint 5 (promotion without a
restart) depends on the answer being yes.

## What the docs say

Checked against https://docs.vllm.ai/en/stable/features/lora and
https://docs.vllm.ai/en/stable/usage/security, not memory.

Two independent switches are needed:

| Switch | Where | Effect |
|---|---|---|
| `--enable-lora` | server flag | Allocates LoRA slots on the GPU. Without it nothing LoRA-related works. |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` | environment variable | Turns on `POST /v1/load_lora_adapter` and `POST /v1/unload_lora_adapter`. Off by default. |

Sizing flags, all fixed at boot:

| Flag | Default | Why we care |
|---|---|---|
| `--max-lora-rank` | 16 | Must be >= the rank (`r` in `adapter_config.json`) of every adapter we ever load. Too high wastes VRAM; too low rejects the load. |
| `--max-loras` | 1 | How many adapters can be active in one batch. Sprint 5 serves base and candidate side by side, but base needs no slot, so 1 may do; 2 leaves room for a rollback target. |
| `--max-cpu-loras` | = max-loras | Adapters parked in host RAM, swapped in on demand. |

Endpoints:

```bash
# load: lora_path is a path *inside the vllm container*
curl -X POST http://vllm:8000/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "candidate-001", "lora_path": "/adapters/candidate-001"}'

# unload
curl -X POST http://vllm:8000/v1/unload_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "candidate-001"}'
```

Once loaded, the adapter is addressed by putting its `lora_name` in the `model` field of a
normal `/v1/chat/completions` request. The base model stays available under its own name.
This is exactly the shape Sprint 5 needs: the api swaps one string per request.

Adapters can also be preloaded at boot with `--lora-modules name=path`, which is what the
compose file should do so a restart comes back with the active adapter already served.

## Security note

vLLM's docs call runtime loading insecure: it reads arbitrary files off disk. In this stack
the vllm service has no `ports:` and is only reachable from the api container and the
compose network (`docs/architecture.md`), so the endpoints are not exposed. The api must
never proxy them.

## Implications for later sprints

- **S4-2** must write adapters onto a volume the vllm service mounts, and the path passed to
  `load_lora_adapter` is the container-side path.
- **S4-2** training config must cap LoRA rank at whatever `--max-lora-rank` is set to.
  Proposal: rank 16 for training, `--max-lora-rank 32` on the server for headroom.
- **S5-3** resolves a `lora_name`, not a path. Promotion = load adapter, then flip the name.
  Rollback = flip the name back; the previous adapter is still loaded if `--max-loras >= 2`.

## GPU-box confirmation (to run)

Uses a public rank-32 adapter for `Qwen/Qwen2.5-7B-Instruct` so this does not wait on
Sprint 4: `zjudai/flowertune-medical-lora-qwen2.5-7b-instruct`
(`adapter_config.json` + `adapter_model.safetensors`, targets `q_proj`, `v_proj`).

```bash
# 1. fetch the adapter onto the host, then into a volume the container sees
mkdir -p adapters/medical
for f in adapter_config.json adapter_model.safetensors; do
  curl -sL -o adapters/medical/$f \
    https://huggingface.co/zjudai/flowertune-medical-lora-qwen2.5-7b-instruct/resolve/main/$f
done

# 2. start vllm with LoRA on (compose override, not a permanent change yet)
cat > docker-compose.lora.yml <<'YAML'
services:
  vllm:
    command:
      - --model=${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
      - --max-num-seqs=${VLLM_MAX_NUM_SEQS:-4}
      - --max-model-len=${VLLM_MAX_MODEL_LEN:-8192}
      - --gpu-memory-utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
      - --disable-access-log-for-endpoints=/health,/metrics
      - --enable-lora
      - --max-lora-rank=32
      - --max-loras=2
    environment:
      VLLM_ALLOW_RUNTIME_LORA_UPDATING: "True"
    volumes:
      - ./adapters:/adapters:ro
YAML
docker compose -f docker-compose.yml -f docker-compose.lora.yml up -d vllm
docker compose logs -f vllm   # wait for "Application startup complete"

# 3. load at runtime and check it appears as a model
docker compose exec api python3 - <<'PY'
import json, urllib.request
def post(path, body):
    req = urllib.request.Request(f"http://vllm:8000{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req).read().decode()
print(post("/v1/load_lora_adapter", {"lora_name": "medical", "lora_path": "/adapters/medical"}))
print(urllib.request.urlopen("http://vllm:8000/v1/models").read().decode())
print(post("/v1/chat/completions", {"model": "medical", "max_tokens": 40,
      "messages": [{"role": "user", "content": "What is hypertension?"}]}))
PY

# 4. record VRAM delta for S0-2
nvidia-smi --query-gpu=memory.used --format=csv
```

## Result

_To fill in on the box:_

- vLLM version string from `/version`:
- `/v1/models` lists `medical` after load: yes / no
- Chat completion with `"model": "medical"` returns 200: yes / no
- VRAM before / after load:
- Anything that did not match the docs:
