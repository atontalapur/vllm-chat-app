"""The eval runner.

No model and no judge are called. The runner's job is to produce a number that
two different runs can be compared on, so these tests guard the things that
would make two numbers quietly incomparable:

- a partial run must not emit a set score
- the eval set's hash must travel with the score
- base and adapter must take the same code path
- the judge must be nameable separately from the model under test
"""

import json
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.eval import run as runner  # noqa: E402
from pipeline.eval.judge import Judge, JudgeError  # noqa: E402
from pipeline.tests.verdicts import answer_schema  # noqa: E402

EVAL_SET = Path(__file__).resolve().parents[1] / "eval" / "worldcup-v1.jsonl"


def result(
    item_id: str = "wc-001", score: float | None = 1.0, error: str | None = None, trap: bool = False
) -> runner.ItemResult:
    return runner.ItemResult(
        id=item_id,
        prompt="q",
        response=None if error else "a",
        string_score=score,
        judge_score=score,
        score=score,
        tripped_trap=trap,
        error=error,
        latency_s=0.1,
    )


# --- refusing to average a partial run ---------------------------------------


def test_complete_run_reports_the_mean() -> None:
    summary = runner.summarise([result(score=1.0), result(score=0.0), result(score=0.5)])

    assert summary["complete"] is True
    assert summary["set_score"] == pytest.approx(0.5)
    assert summary["failed"] == 0


def test_partial_run_emits_no_set_score() -> None:
    """A mean over 2 of 3 items is not comparable to a mean over 3.

    The difference is invisible once it is one number in a table, which is
    exactly when a gate decision gets made on it.
    """
    summary = runner.summarise([result(score=1.0), result(score=1.0), result(error="boom")])

    assert summary["complete"] is False
    assert summary["set_score"] is None
    assert summary["failed"] == 1
    assert "not comparable" in summary["incomplete_reason"]


def test_summary_counts_traps_and_extremes() -> None:
    summary = runner.summarise([result(score=0.0, trap=True), result(score=1.0), result(score=0.0)])

    assert summary["traps_tripped"] == 1
    assert summary["perfect_items"] == 1
    assert summary["zero_items"] == 2


# --- provenance that makes two runs comparable -------------------------------


def test_eval_set_hash_changes_when_the_set_changes(tmp_path: Path) -> None:
    """The set is versioned by filename and must not be edited in place once a
    baseline exists. The hash is how an edit gets caught."""
    a = tmp_path / "set.jsonl"
    a.write_text('{"id": "x"}\n')
    before = runner.sha256_file(a)

    a.write_text('{"id": "x", "prompt": "edited"}\n')

    assert runner.sha256_file(a) != before


def test_load_set_rejects_an_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n")

    with pytest.raises(runner.RunError, match="no items"):
        runner.load_set(empty)


def test_load_set_reads_the_committed_set() -> None:
    items = runner.load_set(EVAL_SET)

    assert len(items) == 52
    assert {"id", "prompt", "gold"} <= set(items[0])


# --- asking the model --------------------------------------------------------


def test_request_names_the_model_and_is_greedy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Temperature 0 so the run measures the model, not the sampler."""
    sent: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float = 0) -> Any:
        sent.update(json.loads(request.data))
        return FakeResponse({"choices": [{"message": {"content": "an answer"}}]})

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    out = runner.ask_model("http://vllm:8000/v1", "my-adapter", "q", 512, 10.0, 0)

    assert out == "an answer"
    assert sent["model"] == "my-adapter"
    assert sent["temperature"] == 0
    assert sent["top_p"] == 1
    assert sent["seed"] == 0


def test_unreachable_server_raises_rather_than_scoring_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(request: Any, timeout: float = 0) -> Any:
        raise runner.urllib.error.URLError("connection refused")

    monkeypatch.setattr(runner.urllib.request, "urlopen", boom)

    with pytest.raises(runner.RunError, match="cannot reach"):
        runner.ask_model("http://vllm:8000/v1", "m", "q", 512, 10.0, 0)


class FakeResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def read(self) -> bytes:
        return json.dumps(self._body).encode()

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


# --- one item end to end -----------------------------------------------------


def stub_everything(
    monkeypatch: pytest.MonkeyPatch, answer: str = "India beat Sri Lanka by 6 wickets."
) -> None:
    monkeypatch.setattr(runner, "ask_model", lambda base_url, model, prompt, mt, ts, seed: answer)

    def fake_post(self: Judge, payload: dict[str, Any]) -> dict[str, Any]:
        reply = answer_schema(payload["response_format"]["json_schema"]["schema"])
        return {"choices": [{"message": {"content": json.dumps(reply)}}]}

    monkeypatch.setattr(Judge, "_post", fake_post)


def test_failed_item_is_recorded_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """One bad item must not discard 51 answers already paid for in GPU time."""

    def boom(*args: object, **kwargs: object) -> str:
        raise JudgeError("judge fell over")

    monkeypatch.setattr(runner, "ask_model", lambda *a, **k: "an answer")
    monkeypatch.setattr(Judge, "score_item", boom)
    judge = Judge(base_url="http://x/v1", judge_model="base")

    out = runner.score_one(
        {"id": "wc-001", "prompt": "q", "gold": "g"}, judge, "http://x/v1", "m", 512, 10.0, 0
    )

    assert out.error is not None
    assert "judge fell over" in out.error
    assert out.score is None


# --- the CLI contract --------------------------------------------------------


def run_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str) -> dict[str, Any]:
    stub_everything(monkeypatch)
    out = tmp_path / "result.json"
    rc = runner.main(
        ["--model", "base-model", "--set", str(EVAL_SET), "--out", str(out), "--limit", "2", *extra]
    )
    payload: dict[str, Any] = json.loads(out.read_text())
    payload["_rc"] = rc
    return payload


def test_base_and_adapter_take_the_same_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The acceptance criterion: only --model differs, no branching.

    A gate whose two sides run different code measures the code as much as
    the model.
    """
    base = run_cli(monkeypatch, tmp_path)
    adapter = run_cli(
        monkeypatch, tmp_path, "--judge-model", "base-model", "--model", "candidate-adapter"
    )

    assert base["summary"] == adapter["summary"]
    assert adapter["model"] == "candidate-adapter"
    assert adapter["judge_model"] == "base-model"


def test_judge_defaults_to_the_model_and_flags_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Self-judging is right for a baseline and wrong for an adapter, so the
    output records which happened rather than leaving it to be inferred."""
    payload = run_cli(monkeypatch, tmp_path)

    assert payload["judge_model"] == "base-model"
    assert payload["judge_is_model_under_test"] is True


def test_output_carries_the_provenance_a_comparison_needs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = run_cli(monkeypatch, tmp_path)

    assert payload["rubric_version"] == "rubric-v1"
    assert payload["eval_set"]["sha256"] == runner.sha256_file(EVAL_SET)
    assert payload["workload"]["temperature"] == 0
    assert payload["workload"]["concurrency"] == 1
    assert payload["provenance"]["timestamp_utc"].endswith("Z")


def test_concurrency_is_recorded_because_it_changes_comparability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Concurrent requests share a vLLM batch, and batching changes numerics."""
    payload = run_cli(monkeypatch, tmp_path, "--concurrency", "4")

    assert payload["workload"]["concurrency"] == 4


def test_exit_code_is_nonzero_on_an_incomplete_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CI and a human both need the failure to be loud."""
    monkeypatch.setattr(runner, "ask_model", lambda *a, **k: "an answer")
    monkeypatch.setattr(
        Judge, "score_item", lambda self, item, response: (_ for _ in ()).throw(JudgeError("nope"))
    )
    out = tmp_path / "r.json"

    rc = runner.main(["--model", "m", "--set", str(EVAL_SET), "--out", str(out), "--limit", "1"])

    assert rc == 1
    assert json.loads(out.read_text())["summary"]["set_score"] is None


def test_rejects_zero_concurrency(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.main(
            [
                "--model",
                "m",
                "--set",
                str(EVAL_SET),
                "--out",
                str(tmp_path / "x.json"),
                "--concurrency",
                "0",
            ]
        )


# --- commit provenance without a git binary ----------------------------------

SHA = "a" * 40


def fake_repo(tmp_path: Path, head: str = "ref: refs/heads/main\n") -> Path:
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text(head)
    return tmp_path


def test_commit_is_read_from_the_files_when_git_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The api image on the box has no git. A baseline with no commit cannot be
    traced to the rubric and runner that produced it."""
    repo = fake_repo(tmp_path)
    (repo / ".git" / "refs" / "heads" / "main").write_text(SHA + "\n")

    def no_git(*_: Any, **__: Any) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(runner.subprocess, "run", no_git)

    assert runner.git_commit(repo) == SHA


def test_read_head_finds_a_packed_ref(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    (repo / ".git" / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n{SHA} refs/heads/main\n"
    )

    assert runner.read_head(repo) == SHA


def test_read_head_returns_a_detached_sha(tmp_path: Path) -> None:
    assert runner.read_head(fake_repo(tmp_path, head=SHA + "\n")) == SHA


def test_read_head_follows_a_worktree_to_the_shared_refs(tmp_path: Path) -> None:
    main = fake_repo(tmp_path / "main")
    (main / ".git" / "refs" / "heads" / "feat").write_text(SHA + "\n")
    wt_git = main / ".git" / "worktrees" / "wt"
    wt_git.mkdir(parents=True)
    (wt_git / "HEAD").write_text("ref: refs/heads/feat\n")
    (wt_git / "commondir").write_text("../..\n")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {wt_git}\n")

    assert runner.read_head(worktree) == SHA


def test_read_head_is_none_outside_a_repo(tmp_path: Path) -> None:
    assert runner.read_head(tmp_path) is None


@pytest.mark.skipif(
    shutil.which("git") is None or not (runner.REPO_ROOT / ".git").exists(),
    reason="needs a git checkout and a git binary",
)
def test_read_head_matches_git_in_this_checkout() -> None:
    sha = runner.git_commit()

    assert sha is not None and len(sha) == 40
    assert runner.read_head(runner.REPO_ROOT) == sha


def git_exits(code: int, stdout: str) -> Any:
    def fake(args: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, code, stdout=stdout, stderr="fatal")

    return fake


def test_a_failing_git_falls_back_to_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """e.g. `dubious ownership` on a mounted repo: git present, exit 128, no sha."""
    repo = fake_repo(tmp_path)
    (repo / ".git" / "refs" / "heads" / "main").write_text(SHA + "\n")
    monkeypatch.setattr(runner.subprocess, "run", git_exits(128, ""))

    assert runner.git_commit(repo) == SHA


def test_an_unborn_branch_records_no_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """git rev-parse HEAD exits 128 there and prints the literal `HEAD` to stdout."""
    repo = fake_repo(tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", git_exits(128, "HEAD\n"))

    assert runner.git_commit(repo) is None


def test_a_non_sha_ref_is_not_recorded(tmp_path: Path) -> None:
    repo = fake_repo(tmp_path)
    (repo / ".git" / "refs" / "heads" / "main").write_text("API_KEY=secret\n")

    assert runner.read_head(repo) is None


def test_a_malformed_git_file_returns_none(tmp_path: Path) -> None:
    """Recording provenance must never be what crashes a paid eval run."""
    (tmp_path / ".git").write_text("garbage\n")

    assert runner.read_head(tmp_path) is None


def test_read_head_follows_a_relative_worktree_gitdir(tmp_path: Path) -> None:
    main = fake_repo(tmp_path / "main")
    (main / ".git" / "refs" / "heads" / "feat").write_text(SHA + "\n")
    wt_git = main / ".git" / "worktrees" / "wt"
    wt_git.mkdir(parents=True)
    (wt_git / "HEAD").write_text("ref: refs/heads/feat\n")
    (wt_git / "commondir").write_text("../..\n")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: ../main/.git/worktrees/wt\n")

    assert runner.read_head(worktree) == SHA


@pytest.mark.parametrize(("stdout", "dirty"), [("", False), (" M pipeline/eval/judge.py\n", True)])
def test_tree_dirty_reads_git_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stdout: str, dirty: bool
) -> None:
    monkeypatch.setattr(runner.subprocess, "run", git_exits(0, stdout))

    assert runner.tree_dirty(tmp_path) is dirty


def test_tree_dirty_is_unknown_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown must not read as clean."""

    def no_git(*_: Any, **__: Any) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(runner.subprocess, "run", no_git)

    assert runner.tree_dirty(tmp_path) is None


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_limit_below_one_is_rejected(limit: str) -> None:
    """0 gives a complete run with no score; -1 silently drops the last item."""
    with pytest.raises(SystemExit):
        runner.main(["--model", "m", "--set", str(EVAL_SET), "--out", "x.json", "--limit", limit])


# --- server identity ---------------------------------------------------------


def test_server_identity_reads_version_and_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_get(url: str, timeout_s: float, payload: Any = None) -> Any:
        seen.append(url)
        if url.endswith("/version"):
            return {"version": "0.28.0"}
        return {"system_fingerprint": "vllm-0.28.0-b0c62709", "choices": []}

    monkeypatch.setattr(runner, "_get_json", fake_get)

    identity = runner.server_identity("http://vllm:8000/v1", "base", 5.0)

    assert identity == {"version": "0.28.0", "system_fingerprint": "vllm-0.28.0-b0c62709"}
    assert seen[0] == "http://vllm:8000/version"


def test_server_identity_never_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(*_: Any, **__: Any) -> Any:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(runner, "_get_json", down)

    assert runner.server_identity("http://vllm:8000/v1", "base", 5.0) == {
        "version": None,
        "system_fingerprint": None,
    }


def test_per_claim_verdicts_are_kept_in_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without them a moved score cannot be traced to the claim that flipped."""
    stub_everything(monkeypatch)
    item = {
        "id": "wc-001",
        "prompt": "q",
        "gold": "g",
        "must_include": [],
        "judge": {"must_state": ["India won"], "must_not_claim": ["runs margin"]},
    }
    judge = Judge(base_url="http://x/v1", judge_model="base")

    out = runner.score_one(item, judge, "http://x/v1", "m", 512, 10.0, 0)

    assert [v["claim"] for v in out.must_state] == ["India won"]
    assert out.must_not_claim[0]["claimed"] is False
