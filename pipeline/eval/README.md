# Held-out eval set

The exam every model takes: the base model for its baseline (S2-4), and each
candidate adapter at the promotion gate (S5-2). Nothing in the training path may
read this directory. The curation step (S3-3) uses it only to reject training
examples that resemble an eval prompt.

## Domain

ICC cricket World Cups: men's ODI (1975-2023), men's T20 (2007-2024), and
women's ODI (1973-2025). Finals, records, rules that decided matches, and the
well-known moments. Narrow enough that ~50 prompts cover it, and the 7B base
gets a useful fraction wrong at high confidence (see `docs/spikes/s0-3-logprob-shape.md`
for one example: a confident "5 wickets" in the 2011 final).

## Files

| File | What |
|---|---|
| `worldcup-v1.jsonl` | 52 items, one JSON object per line. Versioned in the filename; never edit in place once a baseline has been recorded against it. Make a `v2`. |
| `validate.py` | Schema and self-consistency checks. Runs in CI. |

## Item schema

```json
{
  "id": "wc-001",
  "prompt": "Who won the 2011 Cricket World Cup final, and by what margin?",
  "must_include": ["India", "Sri Lanka", "6 wickets"],
  "must_not_include": [],
  "judge": {
    "must_state": ["India won by 6 wickets"],
    "must_not_claim": ["the margin was in runs"]
  },
  "gold": "India beat Sri Lanka by 6 wickets at the Wankhede Stadium, Mumbai, chasing 275.",
  "notes": "Trap: the question implies a runs margin.",
  "valid_through": "2023"
}
```

Two layers of scoring, deliberately separate:

- **String checks** (`must_include`, `must_not_include`): case-insensitive
  substring matches on whitespace-collapsed text. Cheap, deterministic, and
  only for tokens that cannot collide with something else in a correct
  answer. Bare numerals are banned here because `"5"` matches `"2015"`.
- **Judge checks** (`judge.must_state`, `judge.must_not_claim`): natural-language
  statements the rubric judge (S2-2) scores the response against, with `gold`
  as the reference. This is where numbers, margins, and multi-part facts live.

An item may have empty string checks (judge-only, e.g. `wc-028`) but must have
at least one positive check of either kind.

`valid_through` marks items whose answer is a standing record and can be
overtaken. It is the last World Cup the item was checked against. After each
tournament, grep for it and re-verify.

`notes` records why the item exists: the trap it sets, the item it pairs with,
or the wrong answer it was written to catch. It is for humans and is never
shown to the model or the judge.

## Design

- Roughly 30 result-and-margin items, 15 rules and edge cases, 7 multi-part.
- Traps are the point. Most items are written around a specific wrong answer a
  model is likely to give: the wrong margin type, the more famous neighbour
  (Yuvraj for Gibbs, Tendulkar for Kohli), the assumed home final, the
  invented super over.
- Paired items (`wc-014`/`wc-044`, `wc-020`/`wc-021`, `wc-003`/`wc-048`) test
  whether the model keeps two similar facts apart, which a training set could
  easily blur.
- A few plain-recall controls (`wc-011`, `wc-052`) so a score drop on the traps
  can be told apart from a model that has broken generally.

## Provenance

Drafted with model assistance, then every fact checked by hand against
scorecards before commit (2026-09-16). The plan's "hand-written" rule exists
to prevent factual drift into the gold answers; the hand check is what
satisfies it. Any later addition follows the same rule: verify before commit,
and run `validate.py`.

## Validate

```bash
python3 pipeline/eval/validate.py pipeline/eval/worldcup-v1.jsonl
```

## Scoring

| File | What |
|---|---|
| `rubric-v1.md` | The scoring contract. Versioned, never edited in place: a change to the wording or the arithmetic makes old scores incomparable with new ones |
| `judge.py` | Implements it. Standard library only, like everything in `pipeline/` |

Two layers, both in [0, 1], averaged per item:

- **string score**, computed in code with no model involved
- **judge score**, from a model asked for a boolean per claim, never for a number

The judge is the **base model, pinned by name**, even when grading an adapter's
output. vLLM serves both at once (`--enable-lora`), so the ruler stays fixed
while the thing being measured changes. A judge that followed the active model
would make no two runs across the loop comparable.

A tripped `must_not_claim` is a hard zero for that item, not a deduction. The
set is built around traps, so a response that asserts the specific wrong thing
an item was written to catch has failed it whatever else it got right.

### Measuring the run-to-run tolerance

The rubric records this as UNMEASURED, and Sprint 5's significance bar cannot
be set until it has a number: a bar below the noise floor promotes adapters at
random.

`temperature: 0` plus a fixed `seed` plus the JSON schema removes format and
sampling variance, but not batching variance — vLLM's numerics for a request
depend on what else shares its batch. So the drift has to be measured rather
than assumed:

```bash
# On the box, with the stack serving. Needs the S2-3 runner.
python3 pipeline/eval/run.py --model "$MODEL_ID" --set pipeline/eval/worldcup-v1.jsonl --out run-a.json
python3 pipeline/eval/run.py --model "$MODEL_ID" --set pipeline/eval/worldcup-v1.jsonl --out run-b.json
# The set-mean difference between the two is the tolerance. Record it in rubric-v1.md.
```

Until the runner lands (S2-3) this cannot be run, which is why the rubric says
UNMEASURED rather than quoting a plausible-looking figure.
