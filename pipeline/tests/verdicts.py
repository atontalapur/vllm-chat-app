"""Build judge replies that satisfy the schema a request actually sent.

Shared by the judge and runner tests, so a stubbed judge answers exactly the
claims it was asked about, under their numbers, echoing each claim's text.
"""

from typing import Any


def keyed(claims: list[str], verdict_key: str, values: list[bool]) -> dict[str, Any]:
    return {
        str(i + 1): {"claim": claim, "reason": "r", verdict_key: value}
        for i, (claim, value) in enumerate(zip(claims, values, strict=True))
    }


def verdicts(
    must_state: list[str],
    must_not_claim: list[str],
    stated: list[bool] | None = None,
    claimed: list[bool] | None = None,
) -> dict[str, Any]:
    """Every claim stated and no trap claimed, unless told otherwise."""
    return {
        "must_state": keyed(must_state, "stated", stated or [True] * len(must_state)),
        "must_not_claim": keyed(
            must_not_claim, "claimed", claimed or [False] * len(must_not_claim)
        ),
    }


def answer_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The all-good reply to a request, read off the claims in its schema."""

    def claims(side: str) -> list[str]:
        props = schema["properties"][side]["properties"]
        return [props[k]["properties"]["claim"]["const"] for k in sorted(props, key=int)]

    return verdicts(claims("must_state"), claims("must_not_claim"))
