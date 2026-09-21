"""Shared measurement plumbing: provenance, percentiles, result files.

Standard library only, matching the house rule in `scripts/loadtest.py`: no
install step on the GPU box. The one exception in this directory is
`bench_hf_baseline.py`, which cannot avoid torch.

Two decisions here exist because of what an unsourced number costs.

**Every result file carries its own provenance.** The project already has a
number — "~200 tok/s" in README.md:50 — that survives in three places and
cannot be traced to any run, any log, or any commit. It is therefore unusable.
Nothing produced here should end up in that state, so each result file records
the commit SHA, whether the tree was dirty, the GPU, the model, the vLLM
version, and every workload parameter. A summary without provenance is a
rumour.

**Percentiles are nearest-rank, with no interpolation.** Every percentile this
module reports is therefore an actually-observed sample, not a number computed
between two samples. That matters when the figure ends up on a resume and
someone asks where it came from: it came from request number k of the raw
samples array, which is in the same file.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    """Best-effort capture of a command's stdout. None when it cannot run.

    Provenance is worth collecting but never worth failing a benchmark over, so
    every probe here degrades to None rather than raising.
    """
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_provenance() -> dict[str, Any]:
    sha = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    branch = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "commit": sha,
        "branch": branch,
        # A benchmark from a modified tree cannot be reproduced from the commit
        # alone. Recording it is the difference between a citable number and a
        # number someone has to take on trust.
        "tree_dirty": bool(status),
        "dirty_paths": status.splitlines() if status else [],
    }


def gpu_provenance() -> dict[str, Any]:
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if raw is None:
        return {"present": False, "name": None, "memory_total": None, "driver": None}
    parts = [p.strip() for p in raw.split(",")]
    while len(parts) < 3:
        parts.append("")
    return {
        "present": True,
        "name": parts[0] or None,
        "memory_total": parts[1] or None,
        "driver": parts[2] or None,
    }


def host_provenance() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def provenance(**extra: Any) -> dict[str, Any]:
    """The block every result file opens with."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git": git_provenance(),
        "gpu": gpu_provenance(),
        "host": host_provenance(),
        **extra,
    }


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: the returned value is always an observed sample.

    No interpolation on purpose — see the module docstring. Returns None for an
    empty input rather than raising, so a run where everything failed produces
    a result file saying so instead of a traceback.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil((p / 100.0) * len(ordered))
    index = min(max(rank, 1), len(ordered)) - 1
    return ordered[index]


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "percentile_method": "nearest-rank, no interpolation",
    }


def write_results(name: str, payload: dict[str, Any]) -> Path:
    """Write one result file and return its path.

    Timestamped rather than overwritten: comparing two runs is the entire point
    of this directory, and a benchmark that clobbers its own history cannot be
    compared against itself.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = RESULTS_DIR / f"{name}-{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def load_prompts(path: Path | None = None) -> list[str]:
    """The fixed prompt set. Same text for every target, or nothing compares."""
    prompts_path = path or (Path(__file__).resolve().parent / "prompts.json")
    data = json.loads(prompts_path.read_text())
    prompts = data["prompts"]
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit(f"{prompts_path}: 'prompts' must be a non-empty list")
    return [str(p) for p in prompts]


def refuse(message: str) -> None:
    """Stop the run rather than emit a number that would mislead later.

    Every precondition in this directory exits through here. A benchmark that
    silently measures the wrong thing is worse than one that does not run: the
    number it produces looks exactly like a real one.
    """
    print(f"REFUSING TO RUN: {message}", file=sys.stderr)
    raise SystemExit(2)
