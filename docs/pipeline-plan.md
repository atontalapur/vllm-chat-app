# Closed-loop fine-tuning pipeline: delivery plan

Six one-week sprints plus a de-risking Sprint 0, sized for one person working part-time.
Built on top of the existing four-service stack (`docs/architecture.md`), not beside it.

## Ground rules

**Sprint length:** 1 week. **Capacity:** ~10-13 points, Fibonacci scale, where 1 point is
roughly a focused evening.

**Definition of Ready.** A story enters a sprint only when it has acceptance criteria that
can be checked without judgement calls, and any dependency on a spike is already resolved.

**Definition of Done.** Per story, all of:

- `uvx ruff check .` and `uvx ruff format --check .` clean
- `uv run mypy` clean on any touched service
- tests written and passing, upstream mocked (the CI pattern in `.github/workflows/ci.yml`)
- docs updated in the same commit when behaviour changes
- one story, one commit, one PR off a `feat/` branch. Never a bulk commit.

**Sprint review.** Demo the loop as it stands to yourself and write down what surprised you.
The threshold decisions in Sprint 6 are made out of those notes, not out of a textbook.

---

## Epics

| ID | Epic | Sprints |
|---|---|---|
| E1 | Signal capture in the serving path | 1 |
| E2 | Held-out eval harness | 2 |
| E3 | Curation pipeline | 3 |
| E4 | LoRA training and experiment tracking | 4 |
| E5 | Eval gate, promotion, rollback | 5 |
| E6 | Loop operation and threshold tuning | 6 |

---

## Sprint 0: spikes (3 days, timeboxed, no points)

Three assumptions the whole plan rests on. If any is wrong, the plan changes shape, so
prove them before committing story points to anything downstream. Spikes produce a written
finding in `docs/`, not production code.

**S0-1: runtime LoRA loading.** Confirm the vLLM version pinned in `.env.example`
(`v0.28.0-cu129`) supports `--enable-lora` plus runtime adapter loading, and confirm the
exact flag and endpoint names against current docs rather than memory.
*Done when:* an adapter loads into a running server and is servable under its own model
name, with the commands recorded.

**S0-2: GPU co-residency.** vLLM claims 0.90 of VRAM (`docker-compose.yml:37`). Measure
whether a 7B LoRA train can run alongside a shrunk vLLM, or whether serving must stop.
*Done when:* you know which, with a number. This decides whether the loop is continuous or
batched.

**S0-3: logprob shape.** Capture one streamed completion with `logprobs` enabled and record
the actual delta structure.
*Done when:* a sample chunk is saved and the confidence proxy formula is written down.

**Risk if skipped:** Sprint 1 and Sprint 5 both get rewritten mid-sprint.

---

## Sprint 1: signal capture

**Goal:** every chat request leaves a durable trace that can be joined to the operational
log, carrying a confidence proxy, without adding latency to the token path.

| ID | Story | Pts |
|---|---|---|
| 1.1 | Trace store service and schema | 3 |
| 1.2 | Capture response text and logprobs in the stream | 5 |
| 1.3 | Async trace writer | 3 |
| 1.4 | Trace metrics exported to Prometheus | 2 |

**1.1** As an operator, I want a durable store for request/response traces, so failures can
be analysed after the fact.
*Acceptance:* Postgres service in `docker-compose.yml`, internal only, no `ports:` (matches
the security boundary in `docs/architecture.md`). Schema holds request_id, timestamp,
messages, response, model name, mean logprob, plus nullable judge_score and curation_status.
Schema is a checked-in migration, not hand-applied SQL.

**1.2** As the pipeline, I want the assistant response and its token logprobs recorded, so
low-confidence outputs can be found later.
*Acceptance:* `_build_payload` (`api/app/vllm_client.py:44`) requests logprobs behind a
config flag. `stream_chat` accumulates text and logprobs while still yielding each line at
line 122 unchanged. Test asserts the first chunk is yielded before any trace work happens.

**1.3** As a user, I want tracing to never slow down my chat, so the streaming experience is
unchanged.
*Acceptance:* trace writes happen after the stream closes, off the request path. A store
that is down or slow drops traces and increments a counter rather than failing or delaying
the request. Test covers the store-unavailable path.

**1.4** As an operator, I want trace volume and failures on the dashboard, so silent data
loss is visible.
*Acceptance:* `traces_written_total` and `trace_write_failures_total` counters, scraped by
the existing `api` job in `metrics/prometheus.yml`, with a Grafana panel.

**Sprint demo:** send ten messages through the UI, show ten rows joined to ten log lines by
`X-Request-ID`, show the latency panel is flat.

---

## Sprint 2: eval harness

**Goal:** a held-out set that nothing downstream may ever touch, and a runner that scores
any model name the server exposes.

| ID | Story | Pts |
|---|---|---|
| 2.1 | Author the held-out eval set | 5 |
| 2.2 | Judge rubric and judge client | 3 |
| 2.3 | Eval runner CLI | 3 |
| 2.4 | Record the base-model baseline | 2 |

**2.1** As the gate, I want a hand-built held-out set, so improvement can be measured
against something the training data has never seen.
*Acceptance:* 50 prompts in versioned JSONL under `pipeline/eval/`, in one narrow domain,
each with the qualities a good answer must have. Checked into git as data. This is written
by hand, not generated. Budget a full evening.

**2.2** As the gate, I want a rubric-based judge, so soft quality has a numeric score.
*Acceptance:* rubric is a versioned file. Judge scores are reproducible enough that
re-running the same set twice moves the mean by less than a documented tolerance.

**2.3** As an operator, I want to score any served model by name, so base and candidate use
the identical code path.
*Acceptance:* `pipeline/eval` CLI takes a model name and an eval set, writes per-prompt
scores. Same command works for base and adapter, no branching.

**2.4** As the gate, I want a recorded baseline, so every later comparison has a reference.
*Acceptance:* base model scored on the full set, results committed. This number appears in
the final resume metric.

**Sprint demo:** score the base model twice, show run-to-run variance. That variance is what
Sprint 5's significance bar has to clear.

---

## Sprint 3: curation

**Goal:** raw traces become a clean training set with a contamination check you actually
trust.

| ID | Story | Pts |
|---|---|---|
| 3.1 | Failure selection | 2 |
| 3.2 | Near-duplicate removal | 3 |
| 3.3 | Contamination check against the eval set | 5 |
| 3.4 | Quality filters | 2 |
| 3.5 | Curated export | 2 |

**3.1** *Acceptance:* traces are selected by judge score below threshold or mean logprob
below threshold, both configurable. Selection is a query, and the thresholds live in config
so Sprint 6 can tune them without a code change.

**3.2** *Acceptance:* sentence-transformers embeddings, cosine similarity above a threshold
collapses to one example. Runs on CPU in the worker, never on the GPU.

**3.3** *Acceptance:* no curated example exceeds the similarity threshold against any eval
prompt. The test suite includes a deliberately planted near-duplicate that the check must
catch, phrased differently enough that exact matching would miss it. This story is pointed
at 5 because the bug it prevents is invisible until your eval scores look too good.

**3.4** *Acceptance:* truncated responses, empty responses, and responses that hit
`max_tokens` are dropped, with counts logged per reason.

**3.5** *Acceptance:* curated examples export to JSONL with a content hash recorded, so a
training run can name exactly which dataset it consumed.

**Sprint demo:** run curation over accumulated traces, show the funnel counts at each stage.

---

## Sprint 4: training and tracking

**Goal:** one manual fine-tune cycle end to end, fully recorded. Automation comes later.

| ID | Story | Pts |
|---|---|---|
| 4.1 | MLflow service | 2 |
| 4.2 | LoRA training entrypoint | 5 |
| 4.3 | GPU handoff | 3 |
| 4.4 | Run provenance logged | 2 |

**4.1** *Acceptance:* MLflow in Compose, bound to `127.0.0.1` like Grafana, reachable over
the existing SSH tunnel. `docs/commands.md` gains the tunnel line.

**4.2** *Acceptance:* PEFT/TRL LoRA train against the curated JSONL, producing an adapter in
a volume shared with the `vllm` service. Behind a Compose profile so a normal
`docker compose up` never starts it.

**4.3** *Acceptance:* a documented script implementing whichever answer Sprint 0's S0-2
produced. If serving must stop, the script stops it, trains, restarts, and waits for the
health gate. Failure mid-train leaves the serving stack recoverable.

**4.4** *Acceptance:* every run logs dataset hash, hyperparameters, base model, adapter
path, and wall-clock plus GPU cost to MLflow. A run with no dataset hash is a failed run.

**Sprint demo:** one adapter trained from real traces, its MLflow run open on screen.

---

## Sprint 5: gate, promotion, rollback

**Goal:** the piece that makes this MLOps. Nothing ships on a raw number going up.

| ID | Story | Pts |
|---|---|---|
| 5.1 | Serve base and candidate side by side | 3 |
| 5.2 | Bootstrap comparison and promotion bar | 5 |
| 5.3 | Active adapter as runtime state | 3 |
| 5.4 | Promote and rollback commands | 3 |
| 5.5 | Gate panels in Grafana | 2 |

**5.1** *Acceptance:* both models served concurrently under distinct names, scored by the
same Sprint 2 runner in one pass. No redeploy, no model swap during eval.

**5.2** *Acceptance:* bootstrap confidence interval on the score difference. Promotion
requires the interval to exclude zero at a documented level. The bar is config, and the
reasoning behind the chosen value is written down. A rejection logs why, with the interval.

**5.3** *Acceptance:* the api resolves the active adapter per request from shared state, not
from the import-time `settings` singleton (`api/app/config.py:39`), cached with a short TTL
so a promotion takes effect without a restart and without a DB round trip per message.
Test covers cache expiry and store-unavailable fallback to the last known good adapter.

**5.4** *Acceptance:* promote sets active adapter and records the transition. Rollback
restores the previous adapter. Both are single commands. Rollback is tested by actually
promoting a known-worse adapter and reverting.

**5.5** *Acceptance:* Grafana panels for eval score over time, promotion and rejection
history, and failure-signal rate.

**Sprint demo:** promote a winner, then deliberately promote a loser and watch the gate
refuse it. That refusal is the demo.

---

## Sprint 6: operate and tune

**Goal:** run the loop enough times that the thresholds stop being guesses.

| ID | Story | Pts |
|---|---|---|
| 6.1 | Prompt bank for synthetic traffic | 2 |
| 6.2 | Capability-retention eval set | 5 |
| 6.3 | Three full loop runs | 3 |
| 6.4 | Threshold tuning and metrics write-up | 3 |

**6.1** *Acceptance:* `scripts/loadtest.py` samples from a prompt bank instead of its single
hardcoded `PROMPT`, covering the target domain with a realistic failure rate.

**6.2** *Acceptance:* a second, broader eval set that the training data does not target, run
alongside the main gate. A candidate that wins the target set but regresses here is
rejected. This is the catastrophic-forgetting guard, and it is the classic way this project
fails, so it is pointed accordingly.

**6.3** *Acceptance:* three end-to-end cycles completed, each recorded in MLflow, at least
one of them rejected by the gate.

**6.4** *Acceptance:* thresholds adjusted based on observed results with the reasoning
recorded, and README updated with the captured metrics below.

---

## Metrics captured from day one

Recorded in MLflow per run, summarised in the README at the end:

- fine-tune cycles run end to end
- eval score delta, base model to best promoted adapter
- promotions the gate rejected, and why
- wall-clock from signal accumulation to a promote/reject decision
- GPU cost per cycle

---

## Backlog: explicitly out of v1

Deferred on purpose. Building these now is what kills the project before it ships.

- multiple base models
- full fine-tuning
- canary or percentage-based rollout
- human-in-the-loop review UI
- automatic rollback triggered by live metric regression (v1 rollback is manual)

---

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Trainer and server cannot share the GPU | Loop becomes batched, not continuous | Sprint 0 S0-2 measures it before any code depends on the answer |
| Eval contamination via near-duplicates | Scores look good, model is not | Planted-duplicate test in 3.3, eval set never enters curation input |
| Catastrophic forgetting | Target set improves, general ability drops | Retention set in 6.2 gates alongside the main set |
| Trace writes slow the token path | Regression in the one thing the stack was built to do | Off-path writer in 1.3, latency panel checked at every sprint demo |
| Judge score variance swamps real improvement | Gate promotes noise | Sprint 2 measures run-to-run variance first, bar set above it |
| Too few failures to hit the training threshold | Loop never fires | Prompt bank in 6.1 tuned to produce a realistic failure rate |

---

## Tracking: Jira and GitHub

Jira owns the backlog, sprints, and story state. GitHub owns code, review, and CI. The
Jira issue key is the join between them.

**Source of truth.** `docs/jira/backlog.json` holds every epic, sprint, and story above,
with acceptance criteria. `scripts/jira_seed.py` pushes it into Jira and remembers created
keys in `docs/jira/.keys.json` (gitignored), so re-running after an edit creates only what
is missing. `--dry-run` prints the plan, `--csv` produces an import file if you would rather
not hand the script a token.

**Project setup, done once by hand.** Create a Scrum project (not Kanban, sprints need a
scrum board), key `CFT`, and install the GitHub for Jira app so branches, commits, and PRs
carrying a key show up on the issue. Set sprint length to one week.

**Branch naming.** `<type>/CFT-<n>-<short-slug>`, e.g. `feat/CFT-12-capture-logprobs`.
The key in the branch name is what the Jira integration matches on.

**Commits.** Conventional Commits with the key in the scope line:

    feat(api): capture logprobs in stream [CFT-12]

**Pull requests.** Title `CFT-<n>: <summary>`. The template at
`.github/pull_request_template.md` asks for the story's acceptance criteria copied in with
how each was verified. Never push to `main`; CI must be green before merge.

**Story states.** To Do -> In Progress (branch pushed) -> In Review (PR open) -> Done (PR
merged). The GitHub integration moves the first two automatically once configured; Done is
set on merge.

**Sprint cadence.** Sprint starts Monday. Review and retro Sunday evening: demo the sprint
goal to yourself, write down what surprised you, then plan the next sprint from the
backlog. Points that slip roll forward, they are not re-estimated.
