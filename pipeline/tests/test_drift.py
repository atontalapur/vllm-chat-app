"""The drift tool.

The drift is the noise floor Sprint 5's promotion bar has to clear, so these
tests guard the ways it could be quietly wrong:

- two runs that differ in anything but time must not be compared
- an incomplete run has no set score to compare
- a moved item is attributed to the judge or to generation correctly
"""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.eval import drift  # noqa: E402


def item(item_id: str, score: float, response: str = "a") -> dict[str, Any]:
    return {"id": item_id, "response": response, "score": score}


def run(items: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    scores = [i["score"] for i in items]
    payload: dict[str, Any] = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "judge_model": "Qwen/Qwen2.5-7B-Instruct",
        "rubric_version": "rubric-v1",
        "eval_set": {"path": "x.jsonl", "sha256": "abc", "items": len(items)},
        "workload": {
            "max_tokens": 512,
            "temperature": 0,
            "seed": 0,
            "concurrency": 1,
            "base_url": "http://vllm:8000/v1",
        },
        "provenance": {"timestamp_utc": "2026-09-30T00:00:00Z"},
        "summary": {
            "complete": True,
            "set_score": sum(scores) / len(scores),
            "traps_tripped": 0,
        },
        "items": items,
    }
    payload.update(overrides)
    return payload


def test_identical_runs_have_zero_drift() -> None:
    a = run([item("wc-001", 1.0), item("wc-002", 0.5)])

    report = drift.compare(a, copy.deepcopy(a))

    assert report["drift"] == 0
    assert report["items_moved"] == 0
    assert report["responses_changed"] == 0


def test_drift_is_the_absolute_gap_between_set_means() -> None:
    a = run([item("wc-001", 1.0), item("wc-002", 0.5)])
    b = run([item("wc-001", 1.0), item("wc-002", 0.0, response="b")])

    report = drift.compare(b, a)

    assert report["drift"] == pytest.approx(0.25)


def test_a_moved_item_with_the_same_response_is_blamed_on_the_judge() -> None:
    a = run([item("wc-001", 1.0, response="same")])
    b = run([item("wc-001", 0.5, response="same")])

    report = drift.compare(a, b)

    assert report["moved_by_judge"] == 1
    assert report["moved_by_generation"] == 0
    assert report["moved"][0]["cause"] == "judge"


def test_a_moved_item_with_a_new_response_is_blamed_on_generation() -> None:
    a = run([item("wc-001", 1.0, response="one")])
    b = run([item("wc-001", 0.5, response="two")])

    report = drift.compare(a, b)

    assert report["moved_by_generation"] == 1
    assert report["moved"][0]["cause"] == "generation"


def test_a_changed_response_with_the_same_score_is_counted_but_not_moved() -> None:
    a = run([item("wc-001", 1.0, response="one")])
    b = run([item("wc-001", 1.0, response="two")])

    report = drift.compare(a, b)

    assert report["responses_changed"] == 1
    assert report["items_moved"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "my-adapter"),
        ("judge_model", "some-other-judge"),
        ("rubric_version", "rubric-v2"),
        ("eval_set", {"path": "x.jsonl", "sha256": "edited", "items": 1}),
        (
            "workload",
            {
                "max_tokens": 512,
                "temperature": 0,
                "seed": 0,
                "concurrency": 8,
                "base_url": "http://vllm:8000/v1",
            },
        ),
    ],
)
def test_runs_that_differ_in_more_than_time_are_refused(field: str, value: Any) -> None:
    """A gap between two different setups measures the difference, not the noise."""
    a = run([item("wc-001", 1.0)])
    b = run([item("wc-001", 1.0)], **{field: value})

    with pytest.raises(drift.DriftError, match="differ in more than time"):
        drift.compare(a, b)


def test_base_url_is_allowed_to_differ() -> None:
    """The same server reached from the host and from a container has two names."""
    a = run([item("wc-001", 1.0)])
    b = copy.deepcopy(a)
    b["workload"]["base_url"] = "http://localhost:8000/v1"

    assert drift.compare(a, b)["drift"] == 0


def test_an_incomplete_run_is_refused() -> None:
    a = run([item("wc-001", 1.0)])
    b = copy.deepcopy(a)
    b["summary"] = {"complete": False, "set_score": None, "traps_tripped": 0}

    with pytest.raises(drift.DriftError, match="incomplete"):
        drift.compare(a, b)


def test_runs_over_different_items_are_refused() -> None:
    a = run([item("wc-001", 1.0)])
    b = run([item("wc-002", 1.0)])

    with pytest.raises(drift.DriftError, match="different item ids"):
        drift.compare(a, b)


def test_cli_refusal_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(run([item("wc-001", 1.0)])))
    second.write_text(json.dumps(run([item("wc-001", 1.0)], model="my-adapter")))

    assert drift.main([str(first), str(second)]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_cli_writes_the_report(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(run([item("wc-001", 1.0)])))
    second.write_text(json.dumps(run([item("wc-001", 0.5, response="b")])))
    out = tmp_path / "drift.json"

    assert drift.main([str(first), str(second), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["drift"] == pytest.approx(0.5)
