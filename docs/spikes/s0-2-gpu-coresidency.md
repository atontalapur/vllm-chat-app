# S0-2: can the trainer share the GPU with vLLM

**Jira:** CFT-9. **Status:** desk estimate done, measurement on the box pending.

## Question

vLLM claims 0.90 of VRAM (`docker-compose.yml`). Can a 7B LoRA train run alongside a
shrunk vLLM on the 24GB cards the runbook targets, or must serving stop? This decides
whether the loop is continuous (co-resident) or batched (stop-train-restart), and whether
S4-3 is a trivial script or the sprint's hardest story.

## Estimate

Model: `Qwen/Qwen2.5-7B-Instruct`, bf16, 15.2GB of safetensors on the Hub. 28 layers,
hidden 3584, 4 KV heads of 128 dims, so the KV cache costs 56KB per token.

**Serving (vLLM), current settings: 4 sequences, 8192 context**

| Item | GB |
|---|---|
| Weights, bf16 | 15.2 |
| KV cache, 4 x 8192 tokens | 1.9 |
| CUDA context, graphs, activations | ~1.5 |
| **Minimum** | **~18.5** |
| Allocated at 0.90 of 24GB | 21.6 |

Shrunk as far as it usefully goes (1 sequence, 2048 context): ~17GB. Weights dominate;
shrinking the KV cache buys under 2GB.

**Training (PEFT LoRA on the same base), batch 1, seq 1024, gradient checkpointing**

| Variant | Base weights | Activations + optimizer | Total |
|---|---|---|---|
| LoRA, bf16 base | 15.2 | 3-5 | 18-20 |
| QLoRA, 4-bit base | ~5 | 3-5 | 8-10 |

**Co-residency on 24GB:** best case is shrunk vLLM (17) + QLoRA (8-10) = 25-27GB. Does
not fit. The reason is structural: both processes need their own copy of the base
weights, and two copies do not fit in 24GB regardless of tuning.

**Co-residency on 48GB (A6000, L40S, A40):** 21.6 + 10 = 31.6GB, fits with room. This is
the card class where a continuous loop becomes possible.

## Prediction

Stop-train-restart on 24GB. The loop is batched: accumulate traces while serving, stop
serving, train, restart, evaluate, promote or discard.

Consequences if confirmed:

- **S4-3** is a real story: stop `vllm`, run the trainer, restart, wait for the health
  gate, and leave the stack recoverable if training dies halfway. The `api` service has
  `depends_on: vllm: condition: service_healthy`, so a stopped vllm takes the api down
  with it on restart; the script must bring the stack back with `docker compose up -d`,
  not just `start vllm`.
- **S4-2** should use QLoRA anyway. With serving stopped the full 24GB is free and bf16
  LoRA fits, but QLoRA leaves 14GB of headroom for larger batches and longer sequences,
  and trains faster per step on consumer cards. Note: an adapter trained against a 4-bit
  base is then served on the bf16 base by vLLM. This works and is common practice, but
  the small mismatch is one more reason the eval gate, not the training loss, decides
  promotion.
- **Cost per cycle** includes the serving gap. Record it in S6-4: downtime minutes are
  part of "GPU cost per cycle".
- **Not a blocker for any sprint.** Sprint 1-3 and 5 are unaffected.

## Measurement on the box (to run)

Three numbers, then a yes/no.

```bash
# 1. serving footprint at current settings, after warm-up
docker compose up -d
curl -s -H "X-API-Key: $API_KEY" localhost:8080/health   # via tunnel, or exec into api
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 2. serving footprint shrunk
VLLM_MAX_NUM_SEQS=1 VLLM_MAX_MODEL_LEN=2048 VLLM_GPU_MEMORY_UTILIZATION=0.72 \
  docker compose up -d vllm
# wait for healthy, then:
nvidia-smi --query-gpu=memory.used --format=csv

# 3. QLoRA footprint alone (serving stopped)
docker compose stop vllm api ui
docker run --rm --gpus all -v hf-cache:/root/.cache/huggingface \
  -e HF_TOKEN pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime bash -c '
pip -q install transformers peft bitsandbytes accelerate && python3 - <<PY
import torch, time
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    device_map="cuda")
m.gradient_checkpointing_enable()
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM"))
print("loaded, GB:", torch.cuda.memory_allocated()/1e9)
x = torch.randint(0, 150000, (1, 1024), device="cuda")
m(input_ids=x, labels=x).loss.backward()
print("after one step, peak GB:", torch.cuda.max_memory_allocated()/1e9)
PY'
docker compose up -d

# 4. the actual test: shrunk vllm up, then run step 3 alongside it. Expect OOM.
```

## Result

_To fill in on the box:_

| Measurement | GB |
|---|---|
| Serving, default settings | |
| Serving, shrunk (1 seq, 2048 ctx) | |
| QLoRA, one step, peak | |
| Shrunk serving + QLoRA together | fits / OOM |

**Decision:** co-resident / stop-train-restart
