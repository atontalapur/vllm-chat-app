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
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.eval import run as runner  # noqa: E402
from pipeline.eval.judge import Judge, JudgeError  # noqa: E402

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
        claims = payload["messages"][1]["content"]
        must_state = claims.count("\n1.") and 1 or 0
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "must_state": [{"reason": "r", "stated": True}] * must_state,
                                "must_not_claim": [{"reason": "r", "claimed": False}],
                            }
                        )
                    }
                }
            ]
        }

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
