"""Measure run-to-run drift between two eval runs of the same model.

    python3 pipeline/eval/drift.py pipeline/eval/results/base-v1-a.json \\
                                   pipeline/eval/results/base-v1-b.json

Two identical runs should give the same set score. They do not quite, because
vLLM's numerics depend on batching (see `rubric-v1.md`). The gap between them is
the noise floor: a promotion bar set below it promotes adapters at random. This
prints that gap and says where it came from.

**Refuses to compare runs that differ in anything but time.** Different model,
judge, rubric, eval-set hash, or workload means the gap measures the change, not
the noise. `base_url` is exempt: the same server reached from the host and from
a container has two names.

**Splits the drift by cause.** An item whose score moved while its response
stayed byte-identical was moved by the judge. An item whose response changed was
moved by generation. The fix for each is different, so the count of each is
reported rather than one blended number.

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
]


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
    mismatched = [
        f"{field}: {lookup(a, field)!r} vs {lookup(b, field)!r}"
        for field in MUST_MATCH
        if lookup(a, field) != lookup(b, field)
    ]
    if mismatched:
        raise DriftError("runs differ in more than time:\n  " + "\n  ".join(mismatched))
    for name, run in (("first", a), ("second", b)):
        if not lookup(run, "summary.complete"):
            raise DriftError(f"{name} run is incomplete, so it has no set score to compare")


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    check_comparable(a, b)

    items_b = {item["id"]: item for item in b["items"]}
    if set(items_b) != {item["id"] for item in a["items"]}:
        raise DriftError("runs scored different item ids")

    moved = []
    responses_changed = 0
    for item_a in a["items"]:
        item_b = items_b[item_a["id"]]
        same_response = item_a["response"] == item_b["response"]
        if not same_response:
            responses_changed += 1
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
        "rubric_version": a["rubric_version"],
        "eval_set_sha256": a["eval_set"]["sha256"],
        "runs": [lookup(a, "provenance.timestamp_utc"), lookup(b, "provenance.timestamp_utc")],
        "set_scores": [score_a, score_b],
        "drift": abs(score_a - score_b),
        "items": len(a["items"]),
        "responses_changed": responses_changed,
        "items_moved": len(moved),
        "moved_by_judge": sum(1 for m in moved if m["cause"] == "judge"),
        "moved_by_generation": sum(1 for m in moved if m["cause"] == "generation"),
        "traps_tripped": [a["summary"]["traps_tripped"], b["summary"]["traps_tripped"]],
        "moved": moved,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    try:
        report = compare(json.loads(args.first.read_text()), json.loads(args.second.read_text()))
    except DriftError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    a, b = report["set_scores"]
    print(f"model          {report['model']}")
    print(f"set scores     {a:.4f}  {b:.4f}")
    print(f"drift          {report['drift']:.4f}")
    print(f"responses diff {report['responses_changed']} of {report['items']}")
    print(
        f"items moved    {report['items_moved']} "
        f"(judge {report['moved_by_judge']}, generation {report['moved_by_generation']})"
    )
    for m in report["moved"]:
        print(f"  {m['id']}  {m['a']} -> {m['b']}  [{m['cause']}]")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
