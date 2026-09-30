"""The rubric judge.

No model is called. Every test either exercises the arithmetic directly or
stubs the HTTP call, because what needs guarding here is the scoring contract
in `rubric-v1.md`, not the ability to reach a server.

Three properties matter more than the rest, and each has a test that fails
loudly if the contract drifts:

- a tripped trap is a hard zero, however much else the answer got right
- `notes` never reaches the judge, because it holds the answer key
- misaligned verdicts raise rather than score, because a verdict list that
  cannot be matched to its claims is not data
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.eval.judge import (  # noqa: E402
    RUBRIC_VERSION,
    ItemScore,
    Judge,
    JudgeError,
    align,
    build_messages,
    compute_judge_score,
    normalise,
    schema_for,
    score_strings,
)
from pipeline.tests.verdicts import verdicts  # noqa: E402

ITEM: dict[str, Any] = {
    "id": "wc-001",
    "prompt": "Who won the 2011 Cricket World Cup final, and by what margin?",
    "must_include": ["India", "Sri Lanka", "6 wickets"],
    "must_not_include": [],
    "judge": {
        "must_state": ["India won by 6 wickets"],
        "must_not_claim": ["the margin was in runs"],
    },
    "gold": "India beat Sri Lanka by 6 wickets at the Wankhede Stadium, Mumbai, chasing 275.",
    "notes": "Trap: the question implies a runs margin. Base model answered '5 wickets'.",
}


def stub_judge(monkeypatch: pytest.MonkeyPatch, verdicts: dict[str, Any]) -> list[dict[str, Any]]:
    """Replace the HTTP call; return the list that captures each payload sent."""
    sent: list[dict[str, Any]] = []

    def fake_post(self: Judge, payload: dict[str, Any]) -> dict[str, Any]:
        sent.append(payload)
        return {"choices": [{"message": {"content": json.dumps(verdicts)}}]}

    monkeypatch.setattr(Judge, "_post", fake_post)
    return sent


def judge() -> Judge:
    return Judge(base_url="http://vllm:8000/v1", judge_model="Qwen/Qwen2.5-7B-Instruct")


def all_good() -> dict[str, Any]:
    """Verdicts matching ITEM: its one claim stated, its one trap untripped."""
    return verdicts(ITEM["judge"]["must_state"], ITEM["judge"]["must_not_claim"])


# --- the deterministic layer -------------------------------------------------


def test_string_score_needs_every_required_token() -> None:
    assert score_strings(ITEM, "India beat Sri Lanka by 6 wickets.") == 1.0
    assert score_strings(ITEM, "India beat Sri Lanka by 5 wickets.") == 0.0


def test_string_score_is_case_and_whitespace_insensitive() -> None:
    assert score_strings(ITEM, "india   beat\nsri lanka by 6   WICKETS") == 1.0


def test_string_score_rejects_forbidden_tokens() -> None:
    item = {**ITEM, "must_include": ["India"], "must_not_include": ["Pakistan"]}

    assert score_strings(item, "India won") == 1.0
    assert score_strings(item, "India beat Pakistan") == 0.0


def test_string_score_is_none_when_there_are_no_string_checks() -> None:
    """None and 0.0 differ: 'nothing to pass' is not 'passed nothing'."""
    item = {**ITEM, "must_include": [], "must_not_include": []}

    assert score_strings(item, "anything at all") is None


def test_normalise_matches_the_validator() -> None:
    assert normalise("  India   beat\tSri  Lanka\n") == "india beat sri lanka"


# --- the judge layer ---------------------------------------------------------


def test_all_claims_stated_scores_one() -> None:
    score = compute_judge_score([{"stated": True}, {"stated": True}], [{"claimed": False}])

    assert score == 1.0


def test_partial_credit_is_the_fraction_stated() -> None:
    score = compute_judge_score(
        [{"stated": True}, {"stated": False}, {"stated": True}], [{"claimed": False}]
    )

    assert score == pytest.approx(2 / 3)


def test_a_tripped_trap_is_a_hard_zero() -> None:
    """The sharpest rule in the rubric, and the reason the eval set exists.

    A response can state every required claim and still fail the item by
    asserting the specific wrong thing it was written to catch.
    """
    score = compute_judge_score([{"stated": True}, {"stated": True}], [{"claimed": True}])

    assert score == 0.0, "a claimed must_not_claim must zero the item, not deduct from it"


def test_negative_only_item_scores_none_when_untripped() -> None:
    """Nothing was required, so there is no positive evidence to average."""
    assert compute_judge_score([], [{"claimed": False}]) is None
    assert compute_judge_score([], [{"claimed": True}]) == 0.0


# --- item scoring ------------------------------------------------------------


def test_item_score_is_the_mean_of_present_layers() -> None:
    both = ItemScore(item_id="x", string_score=1.0, judge_score=0.0)
    judge_only = ItemScore(item_id="x", string_score=None, judge_score=0.5)
    string_only = ItemScore(item_id="x", string_score=1.0, judge_score=None)

    assert both.score == 0.5
    assert judge_only.score == 0.5
    assert string_only.score == 1.0


def test_item_with_no_checks_raises_rather_than_scoring_zero() -> None:
    """validate.py rejects these, so reaching one means the set is corrupt."""
    with pytest.raises(JudgeError, match="neither string checks nor judge claims"):
        _ = ItemScore(item_id="wc-999", string_score=None, judge_score=None).score


def test_item_score_records_the_rubric_version() -> None:
    """A score without its rubric version cannot be compared to anything."""
    assert ItemScore(item_id="x", string_score=1.0, judge_score=None).rubric_version == "rubric-v1"
    assert RUBRIC_VERSION == "rubric-v1"


# --- the prompt --------------------------------------------------------------


def test_notes_never_reach_the_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """notes holds the trap each item was written around: it is the answer key."""
    sent = stub_judge(monkeypatch, all_good())

    judge().score_item(ITEM, "India beat Sri Lanka by 6 wickets.")

    blob = json.dumps(sent[0])
    assert "Trap:" not in blob
    assert "Base model answered" not in blob


def test_prompt_carries_question_gold_and_candidate() -> None:
    messages = build_messages(
        "Q?", "GOLD ANSWER", "CANDIDATE ANSWER", {"must_state": ["claim one"], "must_not_claim": []}
    )
    user = messages[1]["content"]

    assert "Q?" in user
    assert "GOLD ANSWER" in user
    assert "CANDIDATE ANSWER" in user
    assert "claim one" in user
    assert messages[0]["role"] == "system"


def test_claims_are_numbered_so_verdicts_can_be_aligned() -> None:
    claims = {"must_state": ["alpha", "beta"], "must_not_claim": []}
    messages = build_messages("Q", "G", "C", claims)

    assert "1. alpha" in messages[1]["content"]
    assert "2. beta" in messages[1]["content"]


# --- the request -------------------------------------------------------------


def test_request_pins_the_judge_model_and_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    """The judge is the base model by name, never whatever is being served."""
    sent = stub_judge(monkeypatch, all_good())

    judge().score_item(ITEM, "India beat Sri Lanka by 6 wickets.")

    payload = sent[0]
    assert payload["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["temperature"] == 0
    assert payload["top_p"] == 1
    assert payload["seed"] == 0


def test_request_demands_a_json_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without this vLLM may return prose, and every score in the run is suspect."""
    sent = stub_judge(monkeypatch, all_good())

    judge().score_item(ITEM, "India beat Sri Lanka by 6 wickets.")

    fmt = sent[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["required"] == ["must_state", "must_not_claim"]


def test_schema_keys_each_verdict_to_its_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positional arrays let a skipped claim shift every later verdict onto the
    wrong one. Each verdict now sits under its claim's number and must echo the
    claim's text. Unequal counts, so swapped sides would fail here."""
    item = {**ITEM, "judge": {"must_state": ["a", "b"], "must_not_claim": ["c"]}}
    sent = stub_judge(monkeypatch, verdicts(["a", "b"], ["c"]))

    judge().score_item(item, "India beat Sri Lanka by 6 wickets.")

    schema = sent[0]["response_format"]["json_schema"]["schema"]["properties"]
    state = schema["must_state"]
    assert state["required"] == ["1", "2"]
    assert state["additionalProperties"] is False
    assert state["properties"]["2"]["properties"]["claim"]["const"] == "b"
    assert state["properties"]["1"]["required"] == ["claim", "reason", "stated"]
    assert schema["must_not_claim"]["required"] == ["1"]
    assert schema["must_not_claim"]["properties"]["1"]["properties"]["claim"]["const"] == "c"


def test_schema_for_uses_no_array_length_keywords() -> None:
    """Keyed objects need only properties/required/const, which every backend
    this runs on supports; no dependence on minItems or prefixItems."""
    text = json.dumps(schema_for(["a", "b"], ["c"]))

    for keyword in ("minItems", "maxItems", "prefixItems", "items"):
        assert f'"{keyword}"' not in text


def test_an_empty_claim_side_requires_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    item = {**ITEM, "judge": {"must_state": ["a"], "must_not_claim": []}}
    sent = stub_judge(monkeypatch, verdicts(["a"], []))

    result = judge().score_item(item, "India beat Sri Lanka by 6 wickets.")

    side = sent[0]["response_format"]["json_schema"]["schema"]["properties"]["must_not_claim"]
    assert side["required"] == []
    assert result.judge_score == 1.0


def test_verdicts_are_returned_in_claim_order_whatever_the_key_order() -> None:
    got = {
        "2": {"claim": "b", "reason": "r", "stated": False},
        "1": {"claim": "a", "reason": "r", "stated": True},
    }

    ordered = align(got, ["a", "b"], "must_state", "wc-x")

    assert [v["claim"] for v in ordered] == ["a", "b"]


def test_no_call_is_made_for_a_string_only_item(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judging an item with no claims would spend GPU time to learn nothing."""
    sent = stub_judge(monkeypatch, {"must_state": [], "must_not_claim": []})
    item = {**ITEM, "judge": {"must_state": [], "must_not_claim": []}}

    result = judge().score_item(item, "India beat Sri Lanka by 6 wickets.")

    assert sent == []
    assert result.judge_score is None
    assert result.string_score == 1.0


# --- failure modes -----------------------------------------------------------


def test_a_missing_claim_number_raises_rather_than_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skipped claim cannot be scored: filling it in would invent data."""
    reply = verdicts(ITEM["judge"]["must_state"], ITEM["judge"]["must_not_claim"])
    reply["must_state"]["2"] = reply["must_state"]["1"]
    stub_judge(monkeypatch, reply)

    with pytest.raises(JudgeError, match="answered must_state"):
        judge().score_item(ITEM, "India beat Sri Lanka by 6 wickets.")


def test_a_verdict_under_the_wrong_claim_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """What positional arrays could not catch: the right count, the wrong claim."""
    item = {**ITEM, "judge": {"must_state": ["a", "b"], "must_not_claim": []}}
    reply = verdicts(["b", "a"], [])  # both answered, under swapped numbers
    stub_judge(monkeypatch, reply)

    with pytest.raises(JudgeError, match="does not echo its claim"):
        judge().score_item(item, "India beat Sri Lanka by 6 wickets.")


def test_a_non_boolean_verdict_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = all_good()
    reply["must_state"]["1"]["stated"] = "yes"
    stub_judge(monkeypatch, reply)

    with pytest.raises(JudgeError, match="no boolean stated"):
        judge().score_item(ITEM, "India beat Sri Lanka by 6 wickets.")


def test_array_shaped_verdicts_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old positional shape, from a server that ignored the schema."""
    stub_judge(
        monkeypatch,
        {
            "must_state": [{"reason": "r", "stated": True}],
            "must_not_claim": [{"reason": "r", "claimed": False}],
        },
    )

    with pytest.raises(JudgeError, match="answered must_state list"):
        judge().score_item(ITEM, "India beat Sri Lanka by 6 wickets.")


def test_non_json_reply_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(self: Judge, payload: dict[str, Any]) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "I think the answer is good!"}}]}

    monkeypatch.setattr(Judge, "_post", fake_post)

    with pytest.raises(JudgeError, match="non-JSON"):
        judge().score_item(ITEM, "India beat Sri Lanka by 6 wickets.")


def test_empty_response_body_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(self: Judge, payload: dict[str, Any]) -> dict[str, Any]:
        return {"choices": []}

    monkeypatch.setattr(Judge, "_post", fake_post)

    with pytest.raises(JudgeError, match="no message content"):
        judge().score_item(ITEM, "India beat Sri Lanka by 6 wickets.")


# --- end to end on the real eval set ----------------------------------------


def test_scores_a_real_item_from_the_committed_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guards against the schema drifting away from the data it scores."""
    path = Path(__file__).resolve().parents[1] / "eval" / "worldcup-v1.jsonl"
    first = json.loads(path.read_text().splitlines()[0])
    claims = first.get("judge") or {}
    stub_judge(
        monkeypatch, verdicts(claims.get("must_state", []), claims.get("must_not_claim", []))
    )

    result = judge().score_item(first, first["gold"])

    # gold passes its own string checks — validate.py enforces that — so a
    # fully-stated gold answer must score a clean 1.0.
    assert result.score == 1.0
    assert not result.tripped_trap
