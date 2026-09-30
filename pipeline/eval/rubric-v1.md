# Judge rubric v1

The scoring contract. Every number the promotion gate (S5-2) compares was
produced by this file, so **it is versioned and never edited in place.** A
change to the wording, the schema, or the arithmetic below changes what a score
means, and silently makes today's numbers incomparable with last week's. Make a
`rubric-v2.md` and re-record the baseline against it.

One amendment, made before any score was recorded against this version
(2026-09-30, S2-4): the verdict arrays gained per-item `minItems`/`maxItems`.
Where the judge already returned the right count this changes no output. Where
it did not, the item used to fail and void the run; now it is scored. With no
baseline yet, there was nothing for the change to make incomparable.

`judge.py` reads the identifiers in this document. Rename a heading here and
the code stops matching; that is deliberate, so the two cannot drift apart
unnoticed.

---

## What the judge is

The **base model**, pinned by name, never "the model currently being served".

This is the one rule that makes self-judging sound. Sprint 5 promotes adapters;
if the judge resolved to whatever is active, the ruler would change with the
thing being measured and no two runs across the loop would be comparable. vLLM
serves the base and an adapter at once (`--enable-lora`, about 900 MiB of KV
cache, measured in `docs/spikes/s0-1-runtime-lora.md:129-130`), so the runner
asks the *adapter* for answers and this *fixed base model* to grade them.

A judge that shares a blind spot with the thing it grades will miss what it is
already wrong about. That limit is real and is why the string checks exist
beside it, computed in code where no model is involved.

## What the judge is asked

Per eval item, one call, containing:

- the original prompt
- `gold`, the reference answer
- the candidate response being graded
- the `judge.must_state` claims, each to be marked stated or not
- the `judge.must_not_claim` claims, each to be marked claimed or not

`notes` is **never** sent. It records the trap an item sets, and showing it to
the judge would hand over the answer key.

## What the judge returns

Strict JSON, enforced by vLLM's `response_format: {"type": "json_schema"}`, so
a malformed reply is impossible rather than merely unlikely. Each array's
`minItems` and `maxItems` are set to that item's claim count, so the judge
cannot return one verdict too many or too few. Per claim, in order:

```json
{
  "must_state":     [{"reason": "...", "stated": true}, ...],
  "must_not_claim": [{"reason": "...", "claimed": false}, ...]
}
```

`reason` comes before the boolean on purpose: the model writes its
justification first and commits to the verdict second, which grades harder
cases better than deciding first and rationalising after.

**The score uses only the booleans.** `reason` is for a human reading a gate
decision, and nothing computes over it.

## How a score is computed

Two layers, deliberately separate, both in [0, 1].

**String score** — no model involved. Case-insensitive substring matching on
whitespace-collapsed text: 1.0 when every `must_include` is present and no
`must_not_include` is, otherwise 0.0. `null` when an item has no string checks.

**Judge score** — from the booleans above:

1. If any `must_not_claim` is marked claimed, the item scores **0.0**.
2. Otherwise the score is the fraction of `must_state` entries marked stated.
3. `null` when an item has no judge claims.

Rule 1 is a hard zero, not a deduction, and it is the sharpest edge in this
file. The set is built around traps (`pipeline/eval/README.md`), so a response
that asserts the specific wrong thing an item was written to catch has failed
that item no matter how much else it got right. A response that says "India won
by 5 wickets, beating Sri Lanka at the Wankhede" has the venue, the teams and
the outcome right and is still exactly the failure the item exists to detect.

**Item score** is the mean of whichever of the two layers is present.
**Set score** is the mean of the item scores.

## Determinism

`temperature: 0`, `top_p: 1`, and a fixed `seed`. The schema removes format
variance; these remove sampling variance.

They do not make it bitwise reproducible. vLLM batches continuously, so the
numerics of a given request depend on what else is in its batch, and identical
inputs can still diverge. **The run-to-run tolerance is therefore measured, not
assumed**, and until it is measured on hardware it is UNMEASURED:

| Quantity | Value |
|---|---|
| Set-mean drift across two identical runs | UNMEASURED |

Measure it with the method in `pipeline/eval/README.md` ("Measuring the
run-to-run tolerance") and record the number here before any gate threshold is
set against it. Nothing blocks the measurement now that the S2-3 runner
exists; it needs a box and about twenty minutes.

Sprint 5's significance bar has to clear this drift; a bar below the noise
floor would promote adapters at random.
