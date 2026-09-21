"""Turn two result files into the delta that a resume bullet can cite.

    python3 bench/compare.py --baseline bench/results/hf-baseline-*.json \\
                             --candidate bench/results/stream-gateway-*.json

Prints a markdown table and writes a comparison file next to the inputs, so the
number quoted later has a single artifact behind it that names both sides, both
commits, and both workloads.

## Refusals

The comparison is void unless both sides ran the same workload, so this refuses
on a mismatch of model, prompt-set version, max_tokens, temperature, or prompt
count. That check exists because the easiest way to produce an impressive and
completely false speedup is to benchmark two different workloads a week apart
and subtract them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import refuse, write_results  # noqa: E402

COMPARABLE = ["prompts_file_version", "prompt_count", "max_tokens", "temperature"]


def ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return candidate / baseline


def pct_change(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline == 0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def fmt(value: float | None, digits: int = 2) -> str:
    return "UNMEASURED" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--allow-workload-mismatch",
        action="store_true",
        help="compare anyway, and record the mismatch in the output",
    )
    args = parser.parse_args()

    base = json.loads(args.baseline.read_text())
    cand = json.loads(args.candidate.read_text())

    mismatches = [
        {"field": f, "baseline": base["workload"].get(f), "candidate": cand["workload"].get(f)}
        for f in COMPARABLE
        if base["workload"].get(f) != cand["workload"].get(f)
    ]
    base_model = base["provenance"].get("model")
    cand_model = cand["provenance"].get("model")
    if base_model != cand_model:
        mismatches.append({"field": "model", "baseline": base_model, "candidate": cand_model})

    if mismatches and not args.allow_workload_mismatch:
        detail = "\n".join(
            f"    {m['field']}: baseline={m['baseline']!r} candidate={m['candidate']!r}"
            for m in mismatches
        )
        refuse(
            "the two runs did not use the same workload, so the difference between them "
            f"is not attributable to the thing being compared:\n{detail}\n"
            "  Re-run both sides with matching parameters, or pass "
            "--allow-workload-mismatch to record the comparison with the mismatch noted."
        )

    bs, cs = base["summary"], cand["summary"]
    b_rate = (bs.get("tokens_per_second_per_wave") or {}).get("p50")
    c_rate = (cs.get("tokens_per_second_per_wave") or {}).get("p50")

    rows = [
        ("throughput tok/s (wave p50)", b_rate, c_rate, "higher is better"),
        ("TTFT p50 (s)", bs["ttft_s"]["p50"], cs["ttft_s"]["p50"], "lower is better"),
        ("TTFT p95 (s)", bs["ttft_s"]["p95"], cs["ttft_s"]["p95"], "lower is better"),
        (
            "end-to-end p50 (s)",
            bs["total_latency_s"]["p50"],
            cs["total_latency_s"]["p50"],
            "lower is better",
        ),
        (
            "end-to-end p95 (s)",
            bs["total_latency_s"]["p95"],
            cs["total_latency_s"]["p95"],
            "lower is better",
        ),
    ]

    print(f"\nbaseline : {base['measurement']}  ({args.baseline.name})")
    print(f"candidate: {cand['measurement']}  ({args.candidate.name})\n")
    print("| Metric | Baseline | Candidate | Change | Ratio |")
    print("|---|---|---|---|---|")
    computed: list[dict[str, Any]] = []
    for name, b, c, direction in rows:
        change = pct_change(c, b)
        r = ratio(c, b)
        print(
            f"| {name} | {fmt(b)} | {fmt(c)} | "
            f"{'UNMEASURED' if change is None else f'{change:+.1f}%'} | "
            f"{'UNMEASURED' if r is None else f'{r:.2f}x'} |"
        )
        computed.append(
            {
                "metric": name,
                "baseline": b,
                "candidate": c,
                "pct_change": change,
                "ratio": r,
                "direction": direction,
            }
        )

    payload = {
        "measurement": "comparison",
        "baseline": {
            "file": str(args.baseline),
            "measurement": base["measurement"],
            "commit": base["provenance"]["git"]["commit"],
            "tree_dirty": base["provenance"]["git"]["tree_dirty"],
            "gpu": base["provenance"]["gpu"]["name"],
            "summary": bs,
        },
        "candidate": {
            "file": str(args.candidate),
            "measurement": cand["measurement"],
            "commit": cand["provenance"]["git"]["commit"],
            "tree_dirty": cand["provenance"]["git"]["tree_dirty"],
            "gpu": cand["provenance"]["gpu"]["name"],
            "summary": cs,
        },
        "workload_mismatches": mismatches,
        "metrics": computed,
    }
    path = write_results("comparison", payload)
    print(f"\nwrote {path}")
    if mismatches:
        print("WARNING: workload mismatch recorded in the output file.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
