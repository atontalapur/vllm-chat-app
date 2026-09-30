"""Rubric judge: score one candidate response against one eval item.

    from pipeline.eval.judge import Judge, score_strings

The contract lives in `rubric-v1.md`, not here. This module implements it and
nothing more; the arithmetic comments below quote the rubric so a reader can
check the two agree.

Design notes worth knowing before changing anything:

**The judge never returns a number.** It returns a boolean per claim, and the
score is computed here. Asking a 7B model for "a score out of 10" produces a
number that moves between runs for reasons nobody can inspect. Asking it
"does this response state X?" produces a verdict that can be checked by hand,
disagreed with, and explained in a gate decision. Reproducibility is the
acceptance criterion for this story, and this is the design choice that buys
most of it.

**The model is pinned by name.** `Judge(judge_model=...)` is the base model,
always, even when the response being graded came from an adapter. See the
rubric: a judge that follows the active model is a ruler that changes with the
thing it measures.

**`notes` is never sent.** It holds the trap each item was written around.
`build_messages` takes only the fields it needs, so there is no path for it to
leak into a prompt by accident, and a test asserts that.

Standard library only, like the rest of `pipeline/`: no install step on the GPU
box.
"""

from __future__ import annotations

import copy
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

RUBRIC_VERSION = "rubric-v1"

_WHITESPACE = re.compile(r"\s+")

# Enforced by vLLM's structured output, so a malformed reply is impossible
# rather than merely unlikely. Property order matches the rubric: reason first,
# verdict second, so the model justifies before it commits. The array lengths
# are set per item by `schema_for`: without them the shape is enforced but the
# count is not, and one miscounted item voids a whole run.
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "must_state": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "stated": {"type": "boolean"},
                },
                "required": ["reason", "stated"],
                "additionalProperties": False,
            },
        },
        "must_not_claim": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "claimed": {"type": "boolean"},
                },
                "required": ["reason", "claimed"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["must_state", "must_not_claim"],
    "additionalProperties": False,
}


def schema_for(n_state: int, n_not_claim: int) -> dict[str, Any]:
    """The verdict schema with each array pinned to its claim count.

    The decoder then cannot emit one verdict too many or too few, which the
    length check in `Judge.score_item` would otherwise have to reject.
    """
    schema = copy.deepcopy(_SCHEMA)
    for key, n in (("must_state", n_state), ("must_not_claim", n_not_claim)):
        schema["properties"][key]["minItems"] = n
        schema["properties"][key]["maxItems"] = n
    return schema


_SYSTEM = (
    "You grade answers to cricket World Cup questions against a reference answer. "
    "You are strict and literal. For each claim you are given, decide only whether "
    "the candidate answer makes that claim. Do not reward fluency, length, or "
    "confidence. Do not penalise extra correct detail. Write your reason first, "
    "then the verdict."
)


class JudgeError(Exception):
    """The judge could not produce a usable verdict for an item."""


def normalise(text: str) -> str:
    """Lower-case, whitespace-collapsed. The same normalisation validate.py uses."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def score_strings(item: dict[str, Any], response: str) -> float | None:
    """Deterministic layer: no model involved.

    1.0 only when every `must_include` is present and no `must_not_include` is.
    None when the item has no string checks, so the caller can tell "passed
    nothing" apart from "had nothing to pass".
    """
    must = item.get("must_include") or []
    must_not = item.get("must_not_include") or []
    if not must and not must_not:
        return None

    haystack = normalise(response)
    if any(normalise(s) not in haystack for s in must):
        return 0.0
    if any(normalise(s) in haystack for s in must_not):
        return 0.0
    return 1.0


@dataclass(frozen=True)
class ItemScore:
    """One item's result, carrying enough to explain a gate decision."""

    item_id: str
    string_score: float | None
    judge_score: float | None
    must_state: list[dict[str, Any]] = field(default_factory=list)
    must_not_claim: list[dict[str, Any]] = field(default_factory=list)
    rubric_version: str = RUBRIC_VERSION

    @property
    def score(self) -> float:
        """Mean of whichever layers the item has. Rubric: 'Item score'."""
        present = [s for s in (self.string_score, self.judge_score) if s is not None]
        if not present:
            # validate.py rejects such items, so this is a corrupt set, not a
            # zero-scoring answer. Say so rather than returning a quiet 0.0.
            raise JudgeError(f"{self.item_id} has neither string checks nor judge claims")
        return sum(present) / len(present)

    @property
    def tripped_trap(self) -> bool:
        """True when a must_not_claim was claimed: the hard-zero case."""
        return any(v.get("claimed") for v in self.must_not_claim)


def build_messages(
    prompt: str, gold: str, response: str, claims: dict[str, Any]
) -> list[dict[str, str]]:
    """Assemble the judge prompt.

    Takes the four fields it needs as arguments rather than the whole item, so
    `notes` has no path into the prompt even by accident.
    """
    must_state = claims.get("must_state") or []
    must_not_claim = claims.get("must_not_claim") or []

    lines = [
        f"QUESTION:\n{prompt}",
        f"\nREFERENCE ANSWER:\n{gold}",
        f"\nCANDIDATE ANSWER:\n{response}",
    ]
    if must_state:
        listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(must_state))
        lines.append(
            "\nFor each claim below, does the CANDIDATE ANSWER state it? "
            f"Answer in the same order.\n{listed}"
        )
    if must_not_claim:
        listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(must_not_claim))
        lines.append(
            "\nFor each claim below, does the CANDIDATE ANSWER make it? "
            f"Answer in the same order.\n{listed}"
        )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


class Judge:
    """Calls the pinned base model and turns its verdicts into a score."""

    def __init__(
        self,
        base_url: str,
        judge_model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        seed: int = 0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.judge_model = judge_model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.seed = seed

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        request = urllib.request.Request(  # noqa: S310 - fixed https/http base_url from config
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                body: dict[str, Any] = json.loads(response.read().decode())
                return body
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise JudgeError(f"judge returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise JudgeError(f"cannot reach judge at {self.base_url}: {exc}") from exc

    def score_item(self, item: dict[str, Any], response: str) -> ItemScore:
        """Score one response. Raises JudgeError rather than guessing."""
        claims = item.get("judge") or {}
        must_state = claims.get("must_state") or []
        must_not_claim = claims.get("must_not_claim") or []
        string_score = score_strings(item, response)

        if not must_state and not must_not_claim:
            return ItemScore(item_id=item["id"], string_score=string_score, judge_score=None)

        payload = {
            "model": self.judge_model,
            "messages": build_messages(item["prompt"], item["gold"], response, claims),
            # Determinism, per the rubric. The schema removes format variance;
            # these remove sampling variance. Neither makes vLLM bitwise
            # reproducible under continuous batching — hence the measured
            # tolerance the rubric demands.
            "temperature": 0,
            "top_p": 1,
            "seed": self.seed,
            "max_tokens": 1024,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "verdicts",
                    "strict": True,
                    "schema": schema_for(len(must_state), len(must_not_claim)),
                },
            },
        }
        body = self._post(payload)
        verdicts = self._parse(body, item["id"])

        got_state = verdicts.get("must_state", [])
        got_not_claim = verdicts.get("must_not_claim", [])
        # A verdict list of the wrong length cannot be aligned to the claims it
        # is supposed to answer, so scoring it would be inventing data.
        if len(got_state) != len(must_state) or len(got_not_claim) != len(must_not_claim):
            raise JudgeError(
                f"{item['id']}: judge returned {len(got_state)}/{len(got_not_claim)} verdicts "
                f"for {len(must_state)}/{len(must_not_claim)} claims"
            )

        return ItemScore(
            item_id=item["id"],
            string_score=string_score,
            judge_score=compute_judge_score(got_state, got_not_claim),
            must_state=got_state,
            must_not_claim=got_not_claim,
        )

    @staticmethod
    def _parse(body: dict[str, Any], item_id: str) -> dict[str, Any]:
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeError(f"{item_id}: judge response had no message content") from exc
        try:
            parsed: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            # response_format makes this near-impossible; if it happens the
            # server ignored the schema and every score in the run is suspect.
            raise JudgeError(f"{item_id}: judge returned non-JSON: {content[:200]}") from exc
        return parsed


def compute_judge_score(
    must_state: list[dict[str, Any]], must_not_claim: list[dict[str, Any]]
) -> float | None:
    """Rubric: 'How a score is computed', judge layer.

    Hard zero on any tripped trap, otherwise the fraction of required claims
    stated. The hard zero is the point of the eval set: a response asserting
    the specific wrong thing an item was written to catch has failed it,
    whatever else it got right.
    """
    if any(v.get("claimed") for v in must_not_claim):
        return 0.0
    if not must_state:
        # Only negative claims, none tripped: nothing was required, so there is
        # no positive evidence to average. The string layer carries this item.
        return None
    stated = sum(1 for v in must_state if v.get("stated"))
    return stated / len(must_state)
