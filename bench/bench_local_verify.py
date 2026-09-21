"""Measurement 3: wall-clock of the full local verification cycle, no GPU.

    python3 bench/bench_local_verify.py

Times each gate a change must pass before it can be trusted — both test
suites, ruff, mypy, the eval-set validator, and compose config validation —
and records the total. Every one runs on a laptop with no GPU, which is a
consequence of a design decision: the upstream model server is mocked in tests
(`api/tests/conftest.py`) and replaced by an OpenAI-compatible stand-in for
local running (`docker-compose.local.yml`).

## Read the resulting number carefully

This is the honest version of "the GPU-free dev path saves time", and it is
weaker than it first looks.

There is **no historical GPU-CI baseline in this repo** to diff against. CI has
run on `ubuntu-latest` for all five jobs since it was written
(`.github/workflows/ci.yml`), so there is no before-and-after and nothing was
migrated. Inventing one would be fabrication.

What can be said honestly is a *floor comparison*, and only if it is stated
precisely: verifying a change locally costs the number this script measures,
while the same change verified on the GPU box cannot begin until vLLM is
serving, which was measured once at 3 min 16 s cold start (README.md:44). Those
are not the same operation — the local cycle runs tests the box does not, and
the box exercises a real model the tests mock. The defensible claim is about
the *iteration floor*: the minimum wait before feedback. Any bullet built on
this must say "verification cycle" and "cold-start floor", not "CI got faster",
because CI never got slower to begin with.

The 3 min 16 s side is a single observation from a doc, not something this
script measures. It is labelled UNMEASURED-HERE in the output for that reason.
To measure it properly, time `docker compose up -d` to a healthy vllm on the
box; `scripts/s0-2-take.sh` already prints exactly that clock.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REPO_ROOT, provenance, refuse, write_results  # noqa: E402

# Each gate a change must pass locally. Kept in the order a developer hits
# them, and deliberately identical in content to the CI jobs in
# .github/workflows/ci.yml so the measured cycle is the real one.
STEPS: list[dict[str, Any]] = [
    {
        "name": "ruff check",
        "cmd": ["uvx", "ruff@0.14.2", "check", "."],
        "cwd": ".",
        "ci_job": "lint",
    },
    {
        "name": "ruff format --check",
        "cmd": ["uvx", "ruff@0.14.2", "format", "--check", "."],
        "cwd": ".",
        "ci_job": "lint",
    },
    {
        "name": "eval set validate",
        "cmd": ["python3", "pipeline/eval/validate.py", "pipeline/eval/worldcup-v1.jsonl"],
        "cwd": ".",
        "ci_job": "lint",
    },
    {
        "name": "mypy (api, strict)",
        "cmd": ["uv", "run", "mypy", "app"],
        "cwd": "api",
        "ci_job": "types-and-tests",
    },
    {
        "name": "pytest (api)",
        "cmd": ["uv", "run", "pytest", "-q"],
        "cwd": "api",
        "ci_job": "types-and-tests",
    },
    {
        "name": "pytest (ui)",
        "cmd": ["uv", "run", "pytest", "-q"],
        "cwd": "ui",
        "ci_job": "ui-tests",
    },
    {
        "name": "compose config validate",
        "cmd": ["docker", "compose", "config", "--quiet"],
        "cwd": ".",
        "ci_job": "compose-validate",
        # Needs a populated .env. On a tree without one this fails, and the
        # failure is recorded rather than hidden: a cycle time that skipped a
        # gate is not the cycle time.
        "may_fail_without_env": True,
    },
    {
        "name": "compose config validate (local overlay, GPU-free path)",
        "cmd": [
            "docker",
            "compose",
            "-f",
            "docker-compose.yml",
            "-f",
            "docker-compose.local.yml",
            "config",
            "--quiet",
        ],
        "cwd": ".",
        "ci_job": "compose-validate",
        "may_fail_without_env": True,
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="skip the two compose validation steps (they need docker and a populated .env)",
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    if not (REPO_ROOT / "docker-compose.yml").exists():
        refuse(f"{REPO_ROOT} does not look like the repo root")

    steps = [s for s in STEPS if not (args.skip_docker and s["cmd"][0] == "docker")]

    all_runs: list[dict[str, Any]] = []
    for r in range(args.repeats):
        print(f"cycle {r + 1}/{args.repeats}", file=sys.stderr)
        cycle_started = time.perf_counter()
        step_results: list[dict[str, Any]] = []
        for step in steps:
            started = time.perf_counter()
            try:
                completed = subprocess.run(  # noqa: S603 - fixed argv from STEPS, no shell
                    step["cmd"],
                    cwd=REPO_ROOT / step["cwd"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=900,
                )
                rc: int | None = completed.returncode
                tail = (completed.stdout or completed.stderr or "").strip().splitlines()
            except (OSError, subprocess.SubprocessError) as exc:
                rc = None
                tail = [f"{type(exc).__name__}: {exc}"]
            elapsed = time.perf_counter() - started
            step_results.append(
                {
                    "name": step["name"],
                    "ci_job": step["ci_job"],
                    "cmd": " ".join(step["cmd"]),
                    "cwd": step["cwd"],
                    "returncode": rc,
                    "elapsed_s": elapsed,
                    # Last line only: enough to confirm what ran (e.g. the
                    # pytest count) without pasting a full log into a result
                    # file that is meant to be read.
                    "last_line": tail[-1] if tail else "",
                }
            )
            print(f"  {step['name']}: {elapsed:.2f}s (rc={rc})", file=sys.stderr)
        cycle_wall = time.perf_counter() - cycle_started
        all_runs.append(
            {
                "cycle": r,
                "wall_s": cycle_wall,
                "all_passed": all(s["returncode"] == 0 for s in step_results),
                "steps": step_results,
            }
        )
        print(f"  cycle total {cycle_wall:.2f}s", file=sys.stderr)

    cycle_walls = [run["wall_s"] for run in all_runs]
    payload = {
        "measurement": "local-verification-cycle",
        "label": args.label,
        "provenance": provenance(
            note=(
                "Measures the local, GPU-free verification cycle only. "
                "The GPU-box side of the comparison is NOT measured here."
            ),
        ),
        "workload": {
            "repeats": args.repeats,
            "steps": [s["name"] for s in steps],
            "gpu_required": False,
            "ci_jobs_covered": sorted({s["ci_job"] for s in steps}),
        },
        "summary": {
            "cycle_wall_s_min": min(cycle_walls) if cycle_walls else None,
            "cycle_wall_s_max": max(cycle_walls) if cycle_walls else None,
            "cycle_wall_s_mean": (sum(cycle_walls) / len(cycle_walls)) if cycle_walls else None,
            "all_cycles_passed": all(run["all_passed"] for run in all_runs),
        },
        "comparison": {
            "gpu_box_cold_start_s": 196,
            "gpu_box_cold_start_display": "3 min 16 s",
            "gpu_box_cold_start_source": "README.md:44",
            "gpu_box_cold_start_status": "UNMEASURED-HERE",
            "honest_framing": (
                "Not a like-for-like comparison. The local cycle runs gates the box does "
                "not; the box exercises a real model the local tests mock. The only "
                "defensible reading is the iteration floor: locally, feedback costs the "
                "measured cycle time and no GPU; on the box, nothing can be verified "
                "until vLLM is serving, observed once at 3 min 16 s. There is NO "
                "historical GPU-CI baseline in this repo — all five CI jobs have always "
                "run on ubuntu-latest — so no 'CI got faster' claim is available."
            ),
        },
        "cycles": all_runs,
    }

    path = write_results(f"local-verify{('-' + args.label) if args.label else ''}", payload)
    print(f"\nwrote {path}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
