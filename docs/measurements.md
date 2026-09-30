# Measurements

Every quantitative claim this project makes, where it came from, and what to do
about it. One place, so a number never has to be re-derived or taken on trust.

**Last audited:** 2026-09-21, against commit `8e6d88a`.

---

## How to read this

Two labels, and the distinction is the whole point of the document.

**SPEC** describes vLLM, Qwen2.5-7B-Instruct, or an RTX 3090. Anyone who runs
this compose file gets the same number. It says nothing about the work done
here. Useful for capacity planning, worthless as evidence of contribution.

**CONTRIBUTION** measures a difference a decision made, against a stated
baseline. "X is fast" is not a contribution number. "X went from A to B because
of this change" is.

Then two rules that keep the table honest:

1. **Every number carries a source**: a `file:line`, a command with its output,
   or a commit hash. No exceptions.
2. **A number with no source is written UNMEASURED**, never estimated, rounded,
   or carried over from memory. Two claims in this repo survived for weeks
   because nobody could tell the difference between "measured once" and
   "sounds about right" — see [Corrections needed](#corrections-needed).

Line numbers drift. Every citation here was re-verified against `8e6d88a` on
2026-09-21; if you are reading this much later, spot-check before quoting.

---

## 1. SPEC: what the hardware and the model do

None of these belong on a résumé. All of them are the right inputs for
"can this box run that model?"

| Metric | Value | Source |
|---|---|---|
| Cold start, container up to serving | 3 min 16 s | `README.md:43` |
| Weight download | 117 s for 14.19 GiB | `README.md:44` |
| Model load into VRAM | 14.29 GiB, 123 s | `README.md:45` |
| `torch.compile` | 20.2 s | `README.md:46` |
| KV cache after weights | 5.79 GiB, 108,384 tokens | `README.md:47` |
| Theoretical concurrency at 8k context | 13.23 sequences | `README.md:48` |
| Serving footprint, default settings | 21,023 MiB | `docs/spikes/s0-2-gpu-coresidency.md:113` |
| Serving footprint, shrunk (1 seq, 2048 ctx, 0.72) | 16,675 MiB | `docs/spikes/s0-2-gpu-coresidency.md:114` |
| QLoRA one step, peak, serving stopped | 13.2 GB (loaded 5.9) | `docs/spikes/s0-2-gpu-coresidency.md:115` |
| Shrunk serving + QLoRA co-resident | OOM, 7.1 GB trainer alloc, 50 MiB free | `docs/spikes/s0-2-gpu-coresidency.md:116` |
| One training step, fwd+bwd, seq 1024 | 1.0 s | `docs/spikes/s0-2-gpu-coresidency.md:133` |
| Stack recovery with cached weights | ~60 s | `docs/spikes/s0-2-gpu-coresidency.md:131` |
| LoRA support (`--enable-lora --max-loras=2`) | costs ~900 MiB of KV cache | `docs/spikes/s0-1-runtime-lora.md:129-130` |
| Adapter load VRAM delta, 40 MB adapter | 0 MiB measurable | `docs/spikes/s0-1-runtime-lora.md:127` |
| Community RTX 3090 rate | $0.10-0.15/hr | `docs/gpu-box-runbook.md:26` (vendor price) |

**What to do with these.** Three of them are load-bearing decisions already
made, and they are the reason to keep the table:

- The OOM row killed co-residency. The training loop is stop-train-restart,
  and the ~60 s recovery is a real cost charged to every cycle (S6-4).
- 13.2 GB QLoRA peak against a 16,675 MiB shrunk server on a 24,576 MiB card is
  the arithmetic. A 48 GB card changes the answer; nothing else does.
- LoRA support costing ~900 MiB of KV cache is the price of Sprint 5's runtime
  adapter swapping, payable before any adapter exists.

---

## 2. CONTRIBUTION: measured, with a baseline

The short list. Each one contrasts a before against an after.

| Metric | Before | After | Source |
|---|---|---|---|
| Trace-store startup against an unreachable DB | 5.06 s | 2.01 s | commit `147e8f4`; `PoolTimeout after 5.06s` / `after 2.01s` |
| Trace metric families on `/metrics` before any failure | 2 of 4 | 4 of 4 (7 series) | commit `c45b0e9`; live container `/metrics` grep |
| Local verification cycle, warm | — | 4.58 s | `bench/results/local-verify-20260921T225423Z.json` |
| Local verification cycle, cold caches | — | 19.14 s | same file, cycle 1 |

**The two pool and metrics numbers are the strongest evidence in the repo**,
and both came from running real infrastructure rather than mocks. Each is a
bug that mocked tests reported as passing.

**The verification-cycle numbers need their caveat attached or they overclaim.**
There is no historical GPU-CI baseline here: all five CI jobs have always run on
`ubuntu-latest` (`.github/workflows/ci.yml`). So no "CI got faster" claim
exists. The defensible reading is the iteration floor, and it must name the
other side: locally, feedback costs 4.58 s and no GPU; on the box, nothing can
be verified until vLLM is serving, observed once at 3 min 16 s. Those are not
the same operation. The script records this framing in its own output under
`comparison.honest_framing`, so it travels with the data.

**Do not quote the mean.** The run reports `cycle_wall_s_mean: 9.77`, which
averages one cold cycle (19.14 s) with two warm ones (5.61 s, 4.58 s) and
describes nothing that happens in practice. Use 4.58 s for iteration speed,
19.14 s for a fresh clone. Re-run with `--warmup 1` to make the summary mean
usable.

---

## 3. Counts: real, but weak

Volume of work, not outcomes. Fine in a project description, thin as a bullet.
All verified by command on 2026-09-21.

| Metric | Value | Command |
|---|---|---|
| Tests, total | **84** | `cd api && uv run pytest -q` → 77; `cd ui && uv run pytest -q` → 7 |
| Test functions before parametrize | 54 | `grep -c "^def test_\|^async def test_" api/tests/*.py` |
| mypy strict, api | clean, 10 files | `uv run mypy app` → `Success: no issues found in 10 source files` |
| ruff lint + format | clean, 29 files | `uvx ruff@0.14.2 check .` / `format --check .` |
| Eval prompts, hand-authored | 52 | `python3 pipeline/eval/validate.py pipeline/eval/worldcup-v1.jsonl` |
| CI jobs | 5 | `.github/workflows/ci.yml` |
| CI jobs needing a GPU | 0 of 5 | all `runs-on: ubuntu-latest` |
| Compose services | 7 | `docker-compose.yml` |
| DB migrations | 2 | `ls pipeline/db/migrations/` |

**What to do with these.** Use 84 as a fact about test discipline, not as an
achievement, and always cite the command rather than a doc — the doc version of
this number has already gone stale once.

---

## 4. Chosen constants: decisions without measured alternatives

Every one of these was picked deliberately, and not one was compared against
the alternative. That is what separates them from section 2.

| Constant | Value | Source |
|---|---|---|
| vLLM batch cap | 4 sequences | `docker-compose.yml:29` |
| Max model length | 8192 (below the 32k default) | `docker-compose.yml:35` |
| GPU memory utilisation | 0.90 (vLLM's own default, so SPEC) | `docker-compose.yml:36` |
| Load-test concurrency | 12 | `scripts/loadtest.py:94` |
| Load-test requests / max-tokens | 60 / 200 | `scripts/loadtest.py:97-98` |
| Upstream connect timeout | 10.0 s | `api/app/config.py:30` |
| Max conversation characters | 48,000 | `api/app/config.py:60` |
| Trace queue size | 500 | `api/app/config.py:49` |
| Trace write timeout | 5.0 s | `api/app/config.py:54` |
| Prometheus scrape interval | 5 s (default is 15 s) | `metrics/prometheus.yml` |

**What to do with these.** "I set the batch cap to 4" is only interesting with
what 8 or 16 did beside it. `bench/bench_stream.py` can produce exactly that
comparison for the cap and the queue size, and doing so would convert the two
most interesting rows here into section 2 entries.

---

## 5. UNMEASURED: claimed or implied, with no artifact

| Claim | Where it appears | What is missing |
|---|---|---|
| Throughput "holds roughly flat" through a burst | `docs/commands.md` burst table | No captured run. Prose only |
| `waiting` climbs to ~8 during the burst | same | Same |
| Gateway latency overhead | nowhere measured | Nothing compares direct vLLM to through-gateway |
| Naive HF transformers baseline | does not exist | No baseline implementation in the repo |
| Dev iterations saved by the GPU-free path | nowhere | No before/after; no GPU CI ever existed |
| Actual GPU session cost | `README.md:393` | No invoice or billing export |
| Base-model eval score, Qwen2.5-7B on `worldcup-v1` | S2-4 acceptance; the reference every adapter is compared to | No run on the box yet. `docs/gpu-box-runbook.md` step 6 |
| Run-to-run eval drift | `pipeline/eval/rubric-v1.md` tolerance table | Same run. Sprint 5's promotion bar cannot be set without it |
| Tokens/sec from this project's own tooling | — | `scripts/loadtest.py` records TTFT and wall-clock only and never counts a token (`Results`, `scripts/loadtest.py:39`) |

---

## 6. Corrections needed

Two numbers currently in the repo are wrong or unsupportable. Neither is fixed
by this document.

**`README.md:301` says "36 tests, 29 for the api layer and 7 for the UI."**
The real count is 84 (77 api, 7 ui). It was true when written (commit
`6840805`, which itself corrected an earlier "28 tests") and Sprint 1 added 48
api tests without updating it. This is the second time this line has gone
stale, which is an argument for citing the command instead of restating the
number.

**`README.md:50` says "~200 tok/s".** Unsourceable. No log, commit body, or
dashboard export contains it, and the project's own load generator has never
counted a token, so the tooling here could not have produced it. It was most
likely read off a Grafana panel by eye. It is also SPEC, so it should be cut
rather than re-measured — unless it is re-measured *as a comparison*, which is
what `bench/bench_hf_baseline.py` exists for.

**Cost claims disagree by 2x.** `README.md:393` says a session costs "well
under two dollars"; `docs/gpu-box-runbook.md:27` says "well under a dollar".
Both derive from the vendor rate at `docs/gpu-box-runbook.md:26`, so both are
SPEC. Pick one, or replace both with an actual receipt.

**`bench/bench_local_verify.py:25` and `:210` cite `README.md:44`** for the
3 min 16 s cold start. That line is now `README.md:43`; the citation drifted
when Sprint 1 edited the README. Worth fixing in the same pass.

---

## 7. What to measure next, and what it unlocks

Ordered by value per GPU-minute. Full commands in
[`bench/README.md`](../bench/README.md).

| # | Measurement | Unlocks | Needs |
|---|---|---|---|
| 1 | Stack vs naive sequential HF serving, same 12-request workload | The one headline CONTRIBUTION number this project lacks: throughput and p95 latency against the obvious alternative | GPU box |
| 2 | Gateway overhead, direct vLLM vs through the gateway | The measured price of auth, validation, logging, and trace capture. Honest either way | GPU box |
| 3 | Local verification cycle | Done: 4.58 s warm, 19.14 s cold | Free, this laptop |
| 4 | Batch cap 4 vs 8 vs 16, same workload | Converts a chosen constant into a measured decision | GPU box, same session as 1 and 2 |

Measurement 1 is the gap worth closing. Everything else in this repo describes
a stack that works; nothing yet says it works *better than what you would have
written first*, which is the claim an interviewer actually probes.

**Batch these into one GPU session.** The box bills by the minute and the cold
start is 3 min 16 s, so the marginal cost of adding measurements 2 and 4 to a
session opened for measurement 1 is small. A box opened for benchmarks should
also carry the still-outstanding live check: Sprint 1's logprob parsing has
never seen real vLLM chunks, only the S0-3 capture and mocked tests.

**Before any `docker compose up` on a fresh box**, set `API_DB_PASSWORD` in
`.env` (`openssl rand -hex 16`, keep it hex — it goes into a connection URL).
Without it `db-migrate` refuses to run and `api` never starts.

---

## 8. Résumé bullets available today

Only these two are sourced. Neither describes the inference stack; both came
from the trace-capture work.

> Cut trace-store startup latency against an unreachable database from 5.06 s to
> 2.01 s, against a connection pool whose own timeout setting was silently
> ignored: `pool.open()` honoured its 0.05 s timeout, then blocked stopping a
> worker stuck in an uncancellable libpq connect. Bounded it at the libpq layer.

> Closed a silent-failure gap in production metrics where 2 of 4 counter
> families were absent from `/metrics` until their first failure, so the loss
> dashboards rendered "No data" — indistinguishable from "nothing has gone
> wrong". Pre-declaring every label combination took exported series from 2
> families to 4 at process start.

A third becomes available after measurement 1. Do not draft it with a
placeholder number.

The verification-cycle bullet is usable but weaker, and only with both halves:

> Kept the full verification cycle — 84 tests, strict typing, lint, and compose
> validation — at 4.58 s on a laptop with no GPU, by mocking the model server in
> tests and substituting an OpenAI-compatible stand-in for local runs. On the
> GPU box nothing can be verified until vLLM is serving, observed once at
> 3 min 16 s.

---

## Adding a number to this document

1. Run it. Do not estimate it.
2. Record the source: `file:line`, a command with output, or a commit hash.
3. Label it SPEC or CONTRIBUTION. If it is CONTRIBUTION, name the baseline in
   the same sentence. If there is no baseline, it is a count or a constant, not
   a contribution.
4. If it came from a `bench/` script, the result JSON already carries the
   commit, GPU, model, and workload. Cite the file, not the number alone.
5. Update the audit date and commit at the top.

A number that cannot survive someone asking "compared to what, and how do you
know?" does not belong here.
