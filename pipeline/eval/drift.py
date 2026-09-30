"""Measure run-to-run drift across identical eval runs of the same model.

    python3 pipeline/eval/drift.py pipeline/eval/results/base-v1-{a,b,c}.json

Identical runs should give the same set score. They do not quite, because
vLLM's numerics depend on batching (see `rubric-v1.md`). The spread between
them is the noise floor: a promotion bar set below it promotes adapters at
random. This prints that spread and says where it came from.

**Takes two or more runs and reports the largest gap between any two.** One
pair is one sample of the noise, and a lucky pair sets the floor too low. The
runbook records three.

**Refuses to compare runs that differ in anything but time.** Different model,
judge, rubric, eval-set hash, workload, or commit means the gap measures the
change, not the noise. The commit matters on its own: `rubric-v1` was amended in
place once, so the version string alone cannot tell two judges apart. A field
missing from either run is a refusal, not a match, and so is a run scored from
a dirty tree. So is a different server build: `server.version` must match, and
`server.system_fingerprint` too when both runs recorded one. `base_url` is
exempt: the same server reached from the host and from a container has two
names.

**Splits the drift by cause.** An item whose score moved while its response
stayed byte-identical was moved by the judge. An item whose response changed was
moved by generation. The fix for each is different, so the count of each is
reported rather than one blended number. Judge flips that cancel out, or that
land on items whose score did not move, are counted separately from the
per-claim verdicts, so they are not invisible.

Standard library only: no install step on the GPU box.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Everything that has to match for the gap to be noise rather than a change.
# Dotted paths into the run file written by run.py.
MUST_MATCH = [
    "model",
    "judge_model",
    "rubric_version",
    "eval_set.sha256",
    "eval_set.items",
    "workload.max_tokens",
    "workload.temperature",
    "workload.seed",
    "workload.concurrency",
    "provenance.commit",
    "server.version",
]

# Compared only when both runs recorded one: not every server reports it.
MATCH_WHEN_PRESENT = ["server.system_fingerprint"]


class DriftError(Exception):
    """The two runs cannot be compared."""


def lookup(run: dict[str, Any], dotted: str) -> Any:
    value: Any = run
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def check_comparable(a: dict[str, Any], b: dict[str, Any]) -> None:
    missing = [
        f"{field} missing from the {name} run"
        for field in MUST_MATCH
        for name, run in (("first", a), ("second", b))
        if lookup(run, field) is None
    ]
    if missing:
        raise DriftError("cannot confirm the runs match:\n  " + "\n  ".join(missing))

    mismatched = [
        f"{field}: {lookup(a, field)!r} vs {lookup(b, field)!r}"
        for field in MUST_MATCH
        if lookup(a, field) != lookup(b, field)
    ] + [
        f"{field}: {lookup(a, field)!r} vs {lookup(b, field)!r}"
        for field in MATCH_WHEN_PRESENT
        if None not in (lookup(a, field), lookup(b, field)) and lookup(a, field) != lookup(b, field)
    ]
    if mismatched:
        raise DriftError("runs differ in more than time:\n  " + "\n  ".join(mismatched))

    for name, run in (("first", a), ("second", b)):
        if lookup(run, "provenance.tree_dirty") is True:
            raise DriftError(f"{name} run was scored from a dirty tree; its commit is not its code")
        if not lookup(run, "summary.complete") or lookup(run, "summary.set_score") is None:
            raise DriftError(f"{name} run is incomplete, so it has no set score to compare")

    stamp = lookup(a, "provenance.timestamp_utc")
    if stamp is None or stamp == lookup(b, "provenance.timestamp_utc"):
        raise DriftError("runs share a timestamp; this is one run compared with itself")


def item_ids(run: dict[str, Any], name: str) -> list[str]:
    ids = [item["id"] for item in run["items"]]
    if len(ids) != len(set(ids)):
        raise DriftError(f"{name} run has duplicate item ids")
    return ids


def verdict_signature(item: dict[str, Any]) -> tuple[bool | None, ...]:
    """The judge's booleans for one item, in claim order."""
    return tuple(v.get("stated") for v in item.get("must_state") or []) + tuple(
        v.get("claimed") for v in item.get("must_not_claim") or []
    )


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """One pair of runs."""
    check_comparable(a, b)

    if set(item_ids(a, "first")) != set(item_ids(b, "second")):
        raise DriftError("runs scored different item ids")
    items_b = {item["id"]: item for item in b["items"]}

    moved = []
    responses_changed = 0
    hidden_judge_flips = []
    for item_a in a["items"]:
        item_b = items_b[item_a["id"]]
        same_response = item_a["response"] == item_b["response"]
        if not same_response:
            responses_changed += 1
        if (
            same_response
            and item_a["score"] == item_b["score"]
            and verdict_signature(item_a) != verdict_signature(item_b)
        ):
            # Same answer, same score, different verdicts: judge noise the
            # set score cannot show.
            hidden_judge_flips.append(item_a["id"])
        if item_a["score"] != item_b["score"]:
            moved.append(
                {
                    "id": item_a["id"],
                    "a": item_a["score"],
                    "b": item_b["score"],
                    "cause": "judge" if same_response else "generation",
                }
            )

    score_a = a["summary"]["set_score"]
    score_b = b["summary"]["set_score"]
    return {
        "model": a["model"],
        "judge_model": a["judge_model"],
        # A floor measured with the judge grading itself is not automatically
        # the floor for an adapter graded by the base, so carry which it was.
        "judge_is_model_under_test": a["model"] == a["judge_model"],
        "rubric_version": a["rubric_version"],
        "eval_set_sha256": a["eval_set"]["sha256"],
        "commit": lookup(a, "provenance.commit"),
        "runs": [lookup(a, "provenance.timestamp_utc"), lookup(b, "provenance.timestamp_utc")],
        "set_scores": [score_a, score_b],
        "drift": abs(score_a - score_b),
        "items": len(a["items"]),
        "responses_changed": responses_changed,
        "items_moved": len(moved),
        "moved_by_judge": sum(1 for m in moved if m["cause"] == "judge"),
        "moved_by_generation": sum(1 for m in moved if m["cause"] == "generation"),
        "hidden_judge_flips": hidden_judge_flips,
        "traps_tripped": [a["summary"]["traps_tripped"], b["summary"]["traps_tripped"]],
        "moved": moved,
    }


def compare_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Every pair of two or more runs. The drift is the largest pairwise gap."""
    if len(runs) < 2:
        raise DriftError("need at least two runs to measure drift")
    pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pair = compare(runs[i], runs[j])
            pairs.append({**pair, "pair": [i, j]})

    first = pairs[0]
    scores = [run["summary"]["set_score"] for run in runs]
    return {
        key: first[key]
        for key in (
            "model",
            "judge_model",
            "judge_is_model_under_test",
            "rubric_version",
            "eval_set_sha256",
            "commit",
            "items",
        )
    } | {
        "server": runs[0].get("server"),
        "runs": [lookup(run, "provenance.timestamp_utc") for run in runs],
        "set_scores": scores,
        "drift": max(pair["drift"] for pair in pairs),
        "pairs": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", type=Path, nargs="+", help="two or more run files")
    parser.add_argument("--out", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    try:
        report = compare_runs([json.loads(path.read_text()) for path in args.runs])
    except DriftError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        # Not a run file run.py wrote. Same exit as a refusal, so a script
        # branching on the code cannot mistake it for a crash in the tool.
        print(f"REFUSED: unreadable run file: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"model          {report['model']}")
    print(f"set scores     {'  '.join(f'{s:.4f}' for s in report['set_scores'])}")
    print(f"drift (max)    {report['drift']:.4f}")
    for pair in report["pairs"]:
        i, j = pair["pair"]
        print(
            f"  runs {i}-{j}  drift {pair['drift']:.4f}  "
            f"responses diff {pair['responses_changed']}/{pair['items']}  "
            f"moved {pair['items_moved']} "
            f"(judge {pair['moved_by_judge']}, generation {pair['moved_by_generation']})  "
            f"hidden judge flips {len(pair['hidden_judge_flips'])}"
        )
        for m in pair["moved"]:
            print(f"    {m['id']}  {m['a']} -> {m['b']}  [{m['cause']}]")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
