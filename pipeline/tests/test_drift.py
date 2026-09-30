"""The drift tool.

The drift is the noise floor Sprint 5's promotion bar has to clear, so these
tests guard the ways it could be quietly wrong:

- two runs that differ in anything but time must not be compared
- a field missing from a run is a refusal, not a match
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

COMMIT = "c" * 40
_stamps = iter(range(1_000_000))


def item(
    item_id: str, score: float, response: str = "a", stated: tuple[bool, ...] = (True,)
) -> dict[str, Any]:
    return {
        "id": item_id,
        "response": response,
        "score": score,
        "must_state": [{"claim": f"c{i}", "stated": v} for i, v in enumerate(stated)],
        "must_not_claim": [],
    }


def run(items: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    scores = [i["score"] for i in items]
    payload: dict[str, Any] = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "judge_model": "Qwen/Qwen2.5-7B-Instruct",
        "rubric_version": "rubric-v1",
        "eval_set": {"path": "x.jsonl", "sha256": "abc", "items": len(items)},
        "server": {"version": "0.28.0", "system_fingerprint": "vllm-0.28.0-b0c62709"},
        "workload": {
            "max_tokens": 512,
            "temperature": 0,
            "seed": 0,
            "concurrency": 1,
            "base_url": "http://vllm:8000/v1",
        },
        # Unique per call: two runs sharing a timestamp are refused as one
        # run compared with itself.
        "provenance": {
            "commit": COMMIT,
            "tree_dirty": False,
            "timestamp_utc": f"2026-09-30T00:00:00Z#{next(_stamps)}",
        },
        "summary": {
            "complete": True,
            "set_score": sum(scores) / len(scores),
            "traps_tripped": 0,
        },
        "items": items,
    }
    payload.update(overrides)
    return payload


def rerun(a: dict[str, Any]) -> dict[str, Any]:
    """An identical second run: same everything, later timestamp."""
    b = copy.deepcopy(a)
    b["provenance"]["timestamp_utc"] = f"{a['provenance']['timestamp_utc']}-rerun"
    return b


def set_path(run_payload: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    target = run_payload
    for key in parents:
        target = target[key]
    assert leaf in target, f"{dotted} missing from the fixture"
    target[leaf] = value


def test_identical_runs_have_zero_drift() -> None:
    a = run([item("wc-001", 1.0), item("wc-002", 0.5)])

    report = drift.compare(a, rerun(a))

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


@pytest.mark.parametrize("path", drift.MUST_MATCH)
def test_runs_that_differ_in_more_than_time_are_refused(path: str) -> None:
    """A gap between two different setups measures the difference, not the noise."""
    a = run([item("wc-001", 1.0)])
    b = rerun(a)
    set_path(b, path, "changed")

    with pytest.raises(drift.DriftError, match="differ in more than time"):
        drift.compare(a, b)


@pytest.mark.parametrize("path", drift.MUST_MATCH)
def test_a_field_missing_from_both_runs_is_refused(path: str) -> None:
    """None equals None, so without this two trimmed files would compare as matching."""
    a = run([item("wc-001", 1.0)])
    set_path(a, path, None)
    b = rerun(a)

    with pytest.raises(drift.DriftError, match="missing"):
        drift.compare(a, b)


def test_a_dirty_tree_is_refused() -> None:
    """The commit of a dirty run is not the code that scored it."""
    a = run([item("wc-001", 1.0)])
    b = rerun(a)
    b["provenance"]["tree_dirty"] = True

    with pytest.raises(drift.DriftError, match="dirty"):
        drift.compare(a, b)


def test_unknown_dirty_state_is_allowed() -> None:
    """The box has no git, so dirty state is unknown there. The commit still pins it."""
    a = run([item("wc-001", 1.0)])
    a["provenance"]["tree_dirty"] = None
    b = rerun(a)

    assert drift.compare(a, b)["drift"] == 0


def test_a_run_compared_with_itself_is_refused() -> None:
    """The same file twice gives a drift of zero, which would set the floor at nothing."""
    a = run([item("wc-001", 1.0)])

    with pytest.raises(drift.DriftError, match="itself"):
        drift.compare(a, copy.deepcopy(a))


def test_base_url_is_allowed_to_differ() -> None:
    """The same server reached from the host and from a container has two names."""
    a = run([item("wc-001", 1.0)])
    b = rerun(a)
    b["workload"]["base_url"] = "http://localhost:8000/v1"

    assert drift.compare(a, b)["drift"] == 0


@pytest.mark.parametrize("which", ["first", "second"])
def test_an_incomplete_run_is_refused(which: str) -> None:
    a = run([item("wc-001", 1.0)])
    b = rerun(a)
    target = a if which == "first" else b
    target["summary"] = {"complete": False, "set_score": None, "traps_tripped": 0}

    with pytest.raises(drift.DriftError, match=f"{which} run is incomplete"):
        drift.compare(a, b)


def test_a_complete_run_with_no_set_score_is_refused() -> None:
    """What a zero-item run reports. Subtracting None would crash instead of refusing."""
    a = run([item("wc-001", 1.0)])
    b = rerun(a)
    b["summary"]["set_score"] = None

    with pytest.raises(drift.DriftError, match="incomplete"):
        drift.compare(a, b)


def test_duplicate_item_ids_are_refused() -> None:
    a = run([item("wc-001", 1.0), item("wc-001", 0.0)])

    with pytest.raises(drift.DriftError, match="duplicate"):
        drift.compare(a, rerun(a))


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


def test_cli_refuses_an_unreadable_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A traceback exits 1 and reads as a crash; bad input is a refusal."""
    good = tmp_path / "a.json"
    good.write_text(json.dumps(run([item("wc-001", 1.0)])))
    bad = tmp_path / "b.json"
    bad.write_text("{not json")

    assert drift.main([str(good), str(bad)]) == 2
    assert drift.main([str(good), str(tmp_path / "missing.json")]) == 2
    assert "unreadable run file" in capsys.readouterr().err


def test_cli_writes_the_report(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps(run([item("wc-001", 1.0)])))
    second.write_text(json.dumps(run([item("wc-001", 0.5, response="b")])))
    out = tmp_path / "drift.json"

    assert drift.main([str(first), str(second), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["drift"] == pytest.approx(0.5)


# --- server identity ---------------------------------------------------------


def test_a_different_fingerprint_is_refused() -> None:
    """Same version string, different build: not the same server."""
    a = run([item("wc-001", 1.0)])
    b = rerun(a)
    b["server"]["system_fingerprint"] = "vllm-0.28.0-deadbeef"

    with pytest.raises(drift.DriftError, match="system_fingerprint"):
        drift.compare(a, b)


def test_a_fingerprint_missing_from_one_run_is_not_a_mismatch() -> None:
    """Not every server reports one. The version still has to match."""
    a = run([item("wc-001", 1.0)])
    b = rerun(a)
    b["server"]["system_fingerprint"] = None

    assert drift.compare(a, b)["drift"] == 0


# --- judge flips the score cannot show ----------------------------------------


def test_cancelling_judge_flips_are_counted() -> None:
    """Two verdicts flip in opposite directions: same response, same score."""
    a = run([item("wc-001", 0.5, stated=(True, False))])
    b = rerun(a)
    b["items"][0]["must_state"] = [
        {"claim": "c0", "stated": False},
        {"claim": "c1", "stated": True},
    ]

    report = drift.compare(a, b)

    assert report["items_moved"] == 0
    assert report["hidden_judge_flips"] == ["wc-001"]


# --- more than two runs ------------------------------------------------------


def test_drift_across_runs_is_the_largest_pairwise_gap() -> None:
    a = run([item("wc-001", 1.0), item("wc-002", 1.0)])
    b = rerun(a)
    b["provenance"]["timestamp_utc"] += "-b"
    b["summary"]["set_score"] = 0.75
    c = rerun(a)
    c["provenance"]["timestamp_utc"] += "-c"
    c["summary"]["set_score"] = 0.5

    report = drift.compare_runs([a, b, c])

    assert report["drift"] == pytest.approx(0.5)
    assert [p["pair"] for p in report["pairs"]] == [[0, 1], [0, 2], [1, 2]]
    assert report["set_scores"] == [1.0, 0.75, 0.5]


def test_one_run_is_refused() -> None:
    with pytest.raises(drift.DriftError, match="at least two"):
        drift.compare_runs([run([item("wc-001", 1.0)])])


def test_any_bad_pair_refuses_the_whole_set() -> None:
    a = run([item("wc-001", 1.0)])
    b = rerun(a)
    c = rerun(a)
    c["provenance"]["timestamp_utc"] += "-c"
    c["model"] = "my-adapter"

    with pytest.raises(drift.DriftError, match="differ in more than time"):
        drift.compare_runs([a, b, c])


def test_cli_takes_three_runs(tmp_path: Path) -> None:
    a = run([item("wc-001", 1.0)])
    paths = []
    for n, payload in enumerate([a, rerun(a), rerun(rerun(a))]):
        path = tmp_path / f"{n}.json"
        path.write_text(json.dumps(payload))
        paths.append(str(path))
    out = tmp_path / "drift.json"

    assert drift.main([*paths, "--out", str(out)]) == 0
    assert len(json.loads(out.read_text())["pairs"]) == 3
