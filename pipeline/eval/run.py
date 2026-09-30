"""Score any served model against an eval set.

    python3 pipeline/eval/run.py --model Qwen/Qwen2.5-7B-Instruct \
        --set pipeline/eval/worldcup-v1.jsonl --out baseline.json

The same command scores the base model (S2-4's baseline) and a candidate
adapter (S5-2's gate). `--model` is the only thing that changes, and nothing in
this file branches on which it is: a gate whose two sides run different code is
measuring the code as much as the model.

**Goes straight to vLLM, not through the api.** The gateway builds its own
upstream payload and sets the model from its own config
(`api/app/vllm_client.py:44`), so a client cannot name an adapter through it.
The eval is infrastructure, not user traffic, and it has to say which model it
is scoring.

**The judge is a separate, pinned model.** `--judge-model` defaults to the base
model and should stay there even when `--model` is an adapter. See
`rubric-v1.md`: a judge that follows the model under test is a ruler that
changes with the thing being measured.

**Reproducibility beats speed here.** Both the answer and the judge call run at
temperature 0, and requests are issued one at a time by default. vLLM's
numerics for a request depend on what else shares its batch, so concurrency is
itself a source of run-to-run drift; `--concurrency` exists but a run using it
is only comparable to another run using the same value, which is why it is
recorded in the output.

Standard library only: no install step on the GPU box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.eval.judge import RUBRIC_VERSION, ItemScore, Judge, JudgeError  # noqa: E402


@dataclass
class ItemResult:
    """One item's outcome, including the failure case."""

    id: str
    prompt: str
    response: str | None
    string_score: float | None
    judge_score: float | None
    score: float | None
    tripped_trap: bool
    error: str | None
    latency_s: float
    # Per-claim verdicts, so a score that moves between runs can be traced to
    # the claim that flipped. Empty for string-only items and failures.
    must_state: list[dict[str, Any]] = field(default_factory=list)
    must_not_claim: list[dict[str, Any]] = field(default_factory=list)


class RunError(Exception):
    """The run cannot produce a comparable number."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_set(path: Path) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not items:
        raise RunError(f"{path} has no items")
    return items


REPO_ROOT = Path(__file__).resolve().parents[2]

_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


def _sha_or_none(value: str) -> str | None:
    """Only a real object name counts. `git rev-parse HEAD` on an unborn branch
    exits 128 and prints the literal `HEAD`, and a corrupt ref file can hold
    anything; either would otherwise be recorded as the commit."""
    value = value.strip()
    return value if _SHA.fullmatch(value) else None


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            # Resolved from PATH on purpose: this runs on whatever box the
            # eval runs on, where git's absolute path is not knowable here.
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            cwd=repo,
        )
    except OSError:
        return None


def git_commit(repo: Path = REPO_ROOT) -> str | None:
    """The commit the run was scored at, which ties the number to the code.

    Falls back to reading `.git` directly: on the box the eval runs in the api
    image, which has no git binary, and a baseline with no commit cannot be
    traced back to the rubric and runner that produced it.
    """
    out = _git(repo, "rev-parse", "HEAD")
    if out is not None and out.returncode == 0:
        sha = _sha_or_none(out.stdout)
        if sha is not None:
            return sha
    return read_head(repo)


def tree_dirty(repo: Path = REPO_ROOT) -> bool | None:
    """Whether tracked files differ from the commit. None when git is missing:
    unknown, which must not read as clean.

    Untracked files are ignored, or the first run's result file would mark the
    second run dirty.
    """
    out = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if out is None or out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def read_head(repo: Path) -> str | None:
    """Resolve HEAD from the files under `.git`, without the git binary."""
    try:
        git_dir = repo / ".git"
        if git_dir.is_file():  # a worktree: `.git` points at the real directory
            git_dir = (repo / git_dir.read_text().split(":", 1)[1].strip()).resolve()
        head = (git_dir / "HEAD").read_text().strip()
    except (OSError, IndexError):
        return None
    if not head.startswith("ref:"):
        return _sha_or_none(head)  # detached HEAD holds the sha itself

    ref = head.split(":", 1)[1].strip()
    try:
        # A worktree keeps its own HEAD but shares refs with the main checkout.
        common = git_dir / "commondir"
        ref_root = (git_dir / common.read_text().strip()).resolve() if common.is_file() else git_dir
    except OSError:
        return None
    try:
        return _sha_or_none((ref_root / ref).read_text())
    except OSError:
        pass
    try:
        packed = (ref_root / "packed-refs").read_text()
    except OSError:
        return None
    for line in packed.splitlines():
        sha, _, name = line.partition(" ")
        if name == ref:
            return _sha_or_none(sha)
    return None


def _get_json(url: str, timeout_s: float, payload: dict[str, Any] | None = None) -> Any:
    request = urllib.request.Request(  # noqa: S310 - http(s) base_url from argv
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        return json.loads(response.read().decode())


def server_identity(base_url: str, model: str, timeout_s: float) -> dict[str, str | None]:
    """Which server build produced the run, so drift.py can refuse two runs
    against different ones. Best effort: provenance never fails a run.

    `version` comes from vLLM's `/version` (Ollama's `/api/version` as a local
    fallback). `system_fingerprint` needs a completion, so one 1-token request
    is spent on it; vLLM's reads like `vllm-0.28.0-b0c62709`.
    """
    root = base_url.rstrip("/").removesuffix("/v1")
    version = None
    for path in ("/version", "/api/version"):
        try:
            found = _get_json(root + path, timeout_s).get("version")
        except (urllib.error.URLError, TimeoutError, ValueError, AttributeError):
            continue
        if isinstance(found, str):
            version = found
            break

    fingerprint = None
    try:
        body = _get_json(
            f"{base_url.rstrip('/')}/chat/completions",
            timeout_s,
            {
                "model": model,
                "messages": [{"role": "user", "content": "ok"}],
                "max_tokens": 1,
                "temperature": 0,
            },
        )
        found = body.get("system_fingerprint")
        fingerprint = found if isinstance(found, str) else None
    except (urllib.error.URLError, TimeoutError, ValueError, AttributeError):
        pass
    return {"version": version, "system_fingerprint": fingerprint}


def ask_model(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
    seed: int,
) -> str:
    """One answer from the model under test. Greedy, so the run measures the
    model rather than the sampler."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": seed,
    }
    request = urllib.request.Request(  # noqa: S310 - http(s) base_url from argv
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RunError(f"model returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RunError(f"cannot reach {base_url}: {exc}") from exc

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RunError("model response had no message content") from exc
    if not isinstance(content, str):
        raise RunError(f"model returned {type(content).__name__} content, expected str")
    return content


def score_one(
    item: dict[str, Any],
    judge: Judge,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_s: float,
    seed: int,
) -> ItemResult:
    """Ask, then judge. Records a failure rather than raising, so one bad item
    does not discard the other 51 answers already paid for in GPU time."""
    started = time.perf_counter()
    try:
        response = ask_model(base_url, model, item["prompt"], max_tokens, timeout_s, seed)
        scored: ItemScore = judge.score_item(item, response)
    except (RunError, JudgeError) as exc:
        return ItemResult(
            id=item["id"],
            prompt=item["prompt"],
            response=None,
            string_score=None,
            judge_score=None,
            score=None,
            tripped_trap=False,
            error=f"{type(exc).__name__}: {exc}",
            latency_s=round(time.perf_counter() - started, 3),
        )
    return ItemResult(
        id=item["id"],
        prompt=item["prompt"],
        response=response,
        string_score=scored.string_score,
        judge_score=scored.judge_score,
        score=scored.score,
        tripped_trap=scored.tripped_trap,
        error=None,
        latency_s=round(time.perf_counter() - started, 3),
        must_state=scored.must_state,
        must_not_claim=scored.must_not_claim,
    )


def summarise(results: list[ItemResult]) -> dict[str, Any]:
    """The set score, and a refusal to compute one from a partial run.

    A mean over 50 of 52 items is not comparable to a mean over 52, and the
    difference is invisible once it is a single number in a table. An
    incomplete run reports `set_score: null` and says how many failed.
    """
    scored = [r.score for r in results if r.score is not None]
    failed = [r for r in results if r.error is not None]
    complete = not failed

    summary: dict[str, Any] = {
        "items": len(results),
        "scored": len(scored),
        "failed": len(failed),
        "complete": complete,
        "set_score": statistics.fmean(scored) if complete and scored else None,
        "traps_tripped": sum(1 for r in results if r.tripped_trap),
        "perfect_items": sum(1 for r in results if r.score == 1.0),
        "zero_items": sum(1 for r in results if r.score == 0.0),
    }
    if not complete:
        summary["incomplete_reason"] = (
            f"{len(failed)} of {len(results)} items failed; a set score over a subset "
            "is not comparable to one over the whole set"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="model to score: base id or adapter name")
    parser.add_argument("--set", dest="eval_set", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--judge-model",
        default=None,
        help="defaults to --model; pin it to the BASE model when scoring an adapter",
    )
    parser.add_argument("--base-url", default="http://vllm:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="1 by default: concurrent requests share a vLLM batch, and batching "
        "changes numerics, so runs at different concurrency are not comparable",
    )
    parser.add_argument("--limit", type=int, default=None, help="score only the first N items")
    args = parser.parse_args(argv)

    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    judge_model = args.judge_model or args.model
    items = load_set(args.eval_set)
    if args.limit is not None:
        items = items[: args.limit]

    judge = Judge(
        base_url=args.base_url,
        judge_model=judge_model,
        timeout_s=args.timeout_s,
        seed=args.seed,
    )

    def work(item: dict[str, Any]) -> ItemResult:
        return score_one(
            item, judge, args.base_url, args.model, args.max_tokens, args.timeout_s, args.seed
        )

    server = server_identity(args.base_url, args.model, args.timeout_s)
    if server["version"] is None:
        print(
            "WARNING: cannot read the server version; drift.py will refuse this run",
            file=sys.stderr,
        )

    commit = git_commit()
    if commit is None:
        # Warn up front, before the GPU time is spent: drift.py refuses a run
        # with no commit, because its number cannot be traced to its code.
        print(
            "WARNING: cannot resolve the git commit; this run will not be comparable",
            file=sys.stderr,
        )

    started = time.perf_counter()
    if args.concurrency == 1:
        results = [work(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(work, items))
    wall_s = time.perf_counter() - started

    payload = {
        "model": args.model,
        "judge_model": judge_model,
        # The two being equal is correct for a baseline and wrong for an
        # adapter, so record it rather than leaving a reader to infer it.
        "judge_is_model_under_test": judge_model == args.model,
        "rubric_version": RUBRIC_VERSION,
        "eval_set": {
            "path": str(args.eval_set),
            # The set is versioned by filename and must never be edited in
            # place once a baseline exists. The hash is how an edit is caught.
            "sha256": sha256_file(args.eval_set),
            "items": len(items),
        },
        "server": server,
        "workload": {
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "seed": args.seed,
            "concurrency": args.concurrency,
            "base_url": args.base_url,
        },
        "provenance": {
            "commit": commit,
            "tree_dirty": tree_dirty(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_s": round(wall_s, 2),
        },
        "summary": summarise(results),
        "items": [asdict(r) for r in results],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    summary = payload["summary"]
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)
    if not summary["complete"]:
        print(
            f"INCOMPLETE: {summary['failed']} item(s) failed, no set score emitted",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
