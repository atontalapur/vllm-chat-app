# bench/ — measurements that produce contribution numbers

Everything in this directory exists to answer one question: **which numbers in
this project measure a difference my decisions made, rather than a property of
vLLM, Qwen2.5-7B, or an RTX 3090?**

An audit of the repo (no benchmarks, just sourcing every figure) found that
almost every number already published here is the second kind. "Cold start
3 min 16 s", "14.29 GiB model load", "~200 tok/s" — anyone running the same
compose file on the same card gets the same values. They describe the hardware
and the inference server, not the work. Two figures survived as genuine
contributions, and both belong to the trace-store subsystem rather than to the
serving stack that is the headline of the project.

These three measurements are designed to close that gap by adding the thing
every one of those numbers is missing: **a baseline to be different from.**

> **Nothing here has been run yet.** Every script is written and syntax-checked;
> no measurement in this directory has produced a number. Anything you see
> described below as a claim is a *template to fill from a results file*, never
> a result.

---

## Before anything: the box will not start without this

`API_DB_PASSWORD` became a required variable this week. Without it,
`db-migrate` exits 1, and because `api` waits on
`db-migrate: condition: service_completed_successfully`, the api never starts
and the gateway benchmarks have nothing to talk to. This will bite on a fresh
box, because `.env.example` ships it empty on purpose.

```bash
# On the GPU box, in the repo root, BEFORE docker compose up
grep -q '^API_DB_PASSWORD=.\+' .env || \
  sed -i "s|^API_DB_PASSWORD=.*|API_DB_PASSWORD=$(openssl rand -hex 16)|" .env
grep '^API_DB_PASSWORD=' .env   # must print a value, not an empty assignment
```

Keep it hex. Compose interpolates it into a connection URL, where `@ : / ? #`
would need percent-encoding.

---

## What gets measured

| # | Measurement | Baseline it is different from | Script |
|---|---|---|---|
| 1 | Full stack vs naive single-request serving, same 12-request workload | Hugging Face transformers, one request at a time, same model and GPU | `bench_hf_baseline.py` + `bench_stream.py --mode gateway` |
| 2 | Cost of the application layer | Direct vLLM on :8000, bypassing the gateway | `bench_stream.py --mode vllm` vs `--mode gateway` |
| 3 | Local verification cycle, no GPU | GPU-box cold-start floor of 3 min 16 s (README.md:44) — **weaker, read the caveat** | `bench_local_verify.py` |

### How tokens are counted, and why that matters

The existing `scripts/loadtest.py` records TTFT and wall-clock latency and
**never counts a token** (`Results`, `scripts/loadtest.py:39-45`). That single
gap is why "~200 tok/s" in README.md:50 cannot be traced to any run: no tool in
this repo has ever counted one.

`bench_stream.py` counts tokens itself, from entries in
`choices[0].logprobs.content` on each streamed chunk. Counting *chunks* would
be wrong — `docs/spikes/s0-3-logprob-shape.md` records a real chunk carrying 11
tokens under a single `delta.content`. On the direct path the server's own
`usage.completion_tokens` is recorded alongside as an independent cross-check.

**This makes gateway-mode token counting depend on `CAPTURE_LOGPROBS=true`,**
because the gateway builds its own upstream payload
(`api/app/vllm_client.py:44`) and will not forward a client's logprobs request.
The script refuses to report throughput rather than guess if logprobs are
absent.

---

## Run order

Steps 1-2 are on your Mac. Steps 3-9 are on the GPU box. `hostname` tells you
which you are in: the box prints `ubuntu` or similar, your Mac prints its own
name.

### 1. Mac — measurement 3 (no GPU, do this first, it is free)

```bash
cd ~/Documents/Projects/vllm-chat-app
python3 bench/bench_local_verify.py --repeats 3
```

Writes `bench/results/local-verify-<timestamp>.json`. Takes roughly as long as
three full test cycles. If docker is not running, add `--skip-docker`; the two
compose-validation steps are then recorded as skipped rather than silently
dropped.

### 2. Mac — get bench/ onto the box

`bench/` is not committed yet, so copy it directly:

```bash
export BOX=root@<ip> BOXPORT=<port>     # from the Vast instance card
scp -P $BOXPORT -r bench $BOX:/root/vllm-chat-app/
```

### 3. Box — start the stack

```bash
ssh -p $BOXPORT $BOX
cd /root/vllm-chat-app
# the API_DB_PASSWORD step above, if this is a fresh box
docker compose up -d
docker compose ps        # wait for vllm and api to read healthy
```

Cold start is 5-10 minutes on a fresh box (image pull + ~15GB weights). On a
box with cached weights it was observed once at 3 min 16 s (README.md:44).

### 4. Box — measurement 2, direct vLLM (the baseline half)

```bash
docker compose exec api python3 /app/bench/bench_stream.py \
  --mode vllm --url http://vllm:8000 --repeats 3 --warmup 1 --label direct
```

Run it from inside the `api` container: `vllm` publishes no host port, so
`http://vllm:8000` only resolves on the compose network.

> `bench/` is not mounted into the api container by default, and **mounting it
> would mean editing `docker-compose.yml`, which is out of scope for this
> directory.** Use one of these instead, neither of which modifies a file:
>
> ```bash
> # copy bench/ into the running container
> docker compose cp bench api:/app/bench
> # or run a throwaway api container with bench/ mounted
> docker compose run --rm -v "$PWD/bench:/app/bench" --entrypoint python3 \
>   api /app/bench/bench_stream.py --mode vllm --url http://vllm:8000 --label direct
> ```
>
> Results land inside the container, so copy them back:
> `docker compose cp api:/app/bench/results bench/results`

### 5. Box — measurement 2, through the gateway (and measurement 1b)

```bash
docker compose exec api python3 /app/bench/bench_stream.py \
  --mode gateway --url http://localhost:8080 --repeats 3 --warmup 1 --label gateway
```

`API_KEY` is already in the container's environment, so no `--api-key` is
needed. Passing `"$API_KEY"` from the box's shell would expand to an empty
string, because the shell does not read `.env` (`docs/commands.md:131`).

This same result file is the "full stack" side of measurement 1.

### 6. Box — optional: tracing on vs off

A runtime env override, no file edited:

```bash
CAPTURE_LOGPROBS=false docker compose up -d api      # shell env beats .env
docker compose exec api python3 /app/bench/bench_stream.py \
  --mode gateway --url http://localhost:8080 --repeats 3 \
  --allow-untokenised --label gateway-nologprobs
docker compose up -d api                             # back to the .env value
```

`--allow-untokenised` is required here: with capture off there are no logprobs
in the stream, so tokens are recorded as `null` and only latency and TTFT are
reported. That is the honest output, not a degraded one.

### 7. Box — free the GPU for the baseline

The naive baseline needs ~15 GiB for bf16 weights and the stack holds ~21 GB
(`docs/spikes/s0-2-gpu-coresidency.md:113`). Spike S0-2 established the two do
not co-reside on a 24 GB card, so serving stops first. The script refuses to
start if less than 16 GiB is free, with this instruction:

```bash
docker compose stop vllm api ui
nvidia-smi --query-gpu=memory.free --format=csv    # confirm the card is free
```

### 8. Box — measurement 1a, the naive baseline

torch and transformers are not installed by any script here. Install them
deliberately, in an isolated venv, so nothing lands in the system Python or the
containers:

```bash
python3 -m venv bench/.venv-hf
bench/.venv-hf/bin/pip install torch transformers accelerate
bench/.venv-hf/bin/python bench/bench_hf_baseline.py --repeats 3 --warmup 1
```

The download is large (torch is roughly 2.5 GB); budget for it on a metered
box. Weights themselves are already cached from the vLLM run only if the HF
cache path matches — if not, this re-downloads ~15 GB, so check
`~/.cache/huggingface` before assuming.

### 9. Box — restart serving, then compare

```bash
docker compose up -d
python3 bench/compare.py \
  --baseline  bench/results/hf-baseline-<timestamp>.json \
  --candidate bench/results/stream-gateway-<timestamp>.json
python3 bench/compare.py \
  --baseline  bench/results/stream-vllm-direct-<timestamp>.json \
  --candidate bench/results/stream-gateway-<timestamp>.json
```

`compare.py` refuses if the two sides used different models, prompt-set
versions, `max_tokens`, temperature, or prompt counts — the easiest way to
manufacture an impressive and false speedup is to subtract two different
workloads.

Finally, copy the results off the box before destroying it:

```bash
# from the Mac
scp -P $BOXPORT -r $BOX:/root/vllm-chat-app/bench/results bench/
```

---

## Time and cost

**These are planning estimates, not measurements.** The per-request times are
exactly what the benchmark exists to discover, so the baseline figure is
deliberately generous.

| Step | Estimate | Basis |
|---|---|---|
| 1. Local verify (Mac) | 3-6 min | no GPU, no cost |
| 3. Stack cold start | 3-10 min | 3 min 16 s observed with cached weights (README.md:44); 5-10 min on a fresh box (README.md) |
| 4-5. Two stream runs | ~10 min | 2 targets × (1 warmup + 3 waves) × 12 requests at 200 max_tokens |
| 6. Tracing on/off (optional) | ~6 min | one more target plus an api restart |
| 7. Stop stack | ~1 min | ~60 s stack recovery observed (`docs/spikes/s0-2-gpu-coresidency.md:131`) |
| 8. HF install + baseline | 25-40 min | torch download plus 37 sequential requests; the longest and least predictable step |
| 9. Restart + compare | ~5 min | |
| **Total GPU time** | **~1.25-1.5 h** | sum of the box-side rows |

Cost, as arithmetic, from the vendor list price in `docs/gpu-box-runbook.md:26`
of $0.10-0.15/hr for a community RTX 3090:

```
1.5 h × $0.10/hr = $0.15
1.5 h × $0.15/hr = $0.225
```

**ESTIMATE: $0.15 - $0.23.** This is a list price multiplied by a predicted
duration. It is not a receipt, and it must never be cited as one — the existing
"under $2 per session" claim in README.md:393 has exactly this problem, and
disagrees with `docs/gpu-box-runbook.md:27` ("well under a dollar") by 2×. If
you want a real cost number, screenshot the Vast billing page after the session
and cite that.

---

## What each measurement would let you claim

Templates. `<...>` is filled from the named field of the named result file —
never from memory, and never rounded up.

### Measurement 1 — the headline

> Served a 12-request concurrent workload at `<candidate tokens_per_second_per_wave.p50>`
> tokens/sec against `<baseline …p50>` for naive single-request Hugging Face
> transformers serving on the same RTX 3090 and the same model
> (`<ratio>`× throughput), cutting p95 end-to-end latency from `<baseline
> total_latency_s.p95>`s to `<candidate …p95>`s, by serving through vLLM with
> continuous batching behind a FastAPI gateway.

**The honest framing, which must survive into the bullet.** This credits an
*architecture and configuration decision* — choosing a paged-attention server
with continuous batching, setting the batch cap, and putting an application
layer in front — not code written inside vLLM. Phrase it as "designed and
configured a stack that achieves X against the naive alternative", never as
"optimised inference by X". An interviewer who asks "what did you actually
write?" should get a clean answer, and the inflated version does not have one.

It is still a real contribution number: the baseline is the obvious thing a
competent engineer does first, and the delta is what the decision bought.

### Measurement 2 — the application layer's price

> Quantified the cost of the API gateway at `<added ms>` ms added p50 latency
> (`<pct>`%) over direct vLLM access on an identical 12-request workload, the
> price of auth, Pydantic validation, a conversation size guard, structured
> request logging, and per-request trace capture.

This is a genuine contribution number in both directions. A small overhead
justifies the layer; a large one is a finding worth acting on. Do not run it
only if you expect a flattering answer.

### Measurement 3 — the GPU-free path

> Kept the full verification cycle — 84 tests, strict type checking, lint, and
> compose validation — at `<cycle_wall_s_mean>`s on a laptop with no GPU, by
> mocking the model server in tests and shipping an OpenAI-compatible stand-in
> for local runs, so no change needs the 3 min 16 s GPU cold start
> (README.md:44) before it can be verified.

**This one is the weakest and its caveat is load-bearing.** There is no
historical GPU-CI baseline in this repo — all five CI jobs have always run on
`ubuntu-latest` — so **no "CI got N% faster" claim is available**, and inventing
one would be fabrication. The defensible claim is about the *iteration floor*,
and the two sides are not like-for-like: the local cycle runs gates the box does
not, and the box exercises a real model that the local tests mock. The script
records this caveat inside its own output file so it cannot be quietly dropped
later.

---

## Files

| File | What it does |
|---|---|
| `common.py` | Provenance capture (commit, dirty-tree flag, GPU, host), nearest-rank percentiles, result-file writing, the shared `refuse()` |
| `prompts.json` | The fixed 12-prompt set, versioned. Changing it invalidates comparisons against older results |
| `bench_stream.py` | Concurrent SSE benchmark for gateway or direct vLLM. Counts tokens from `logprobs.content`. Stdlib only |
| `bench_hf_baseline.py` | Naive sequential Hugging Face baseline. The only script needing torch; installs nothing itself |
| `bench_local_verify.py` | Times the local GPU-free verification cycle, and records the honest-framing caveat in its output |
| `compare.py` | Turns two result files into a delta table; refuses on workload mismatch |
| `results/` | Timestamped JSON, one file per run. Raw per-request samples, never just summaries |

## Rules these scripts follow

- **Refuse rather than mislead.** Every precondition failure exits non-zero with
  an instruction. A benchmark that silently measures the wrong configuration
  produces a number indistinguishable from a real one.
- **Raw samples always go to disk.** Every percentile can be rechecked against
  the array it came from.
- **Provenance in every file.** Commit SHA, whether the tree was dirty, GPU
  name, driver, model, vLLM version, and every workload parameter. A number
  whose own file cannot say where it came from is unusable three weeks later,
  which is the exact fate of "~200 tok/s".
- **Nothing outside `bench/` is touched.** No source, config, test, doc, or CI
  file is modified by anything here, including at run time — the one env
  override in step 6 is set in the shell, not written to `.env`.
- **Percentiles are nearest-rank, no interpolation**, so every reported
  percentile is a sample that actually occurred.
