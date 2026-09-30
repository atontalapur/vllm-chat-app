# Judge rubric v1

The scoring contract. Every number the promotion gate (S5-2) compares was
produced by this file, so **it is versioned and never edited in place.** A
change to the wording, the schema, or the arithmetic below changes what a score
means, and silently makes today's numbers incomparable with last week's. Make a
`rubric-v2.md` and re-record the baseline against it.

One amendment, made before any score was recorded against this version
(2026-09-30, S2-4): the verdicts changed from positional arrays to objects
keyed by claim number, each echoing its claim's text. Arrays aligned a verdict
to a claim by position alone, so a judge that skipped a claim either voided the
run on a count mismatch or, with the count forced, silently scored every later
verdict against the wrong claim. With no baseline yet, there was nothing for
the change to make incomparable. This was the last edit in place: the next
change is `rubric-v2`.

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
a malformed reply is impossible rather than merely unlikely. The schema is
built per item. Each claim is answered under its number from the prompt, and
must repeat the claim's text exactly (`const`) before judging it:

```json
{
  "must_state": {
    "1": {"claim": "India won by 6 wickets", "reason": "...", "stated": true}
  },
  "must_not_claim": {
    "1": {"claim": "the margin was in runs", "reason": "...", "claimed": false}
  }
}
```

A verdict can therefore only sit under its own claim. `judge.py` checks the
numbers and the echoed text again, for any server that ignores the schema, and
fails the item rather than guess.

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
| Set-mean drift, largest gap across three identical runs | UNMEASURED |

Three runs, not two: one pair is one sample of the noise, and a lucky pair sets
the floor too low. The measurement has the judge grading its own model's
answers (`judge_is_model_under_test`), with no adapter in the batch. Whether the
floor holds for an adapter graded by the base is a Sprint 5 question, and the
drift report carries the flag so the two are not confused.

Measure it with the method in `pipeline/eval/README.md` ("Measuring the
run-to-run tolerance") and record the number here before any gate threshold is
set against it. Nothing blocks the measurement now that the S2-3 runner
exists; it needs a box and about twenty minutes.

Sprint 5's significance bar has to clear this drift; a bar below the noise
floor would promote adapters at random.
