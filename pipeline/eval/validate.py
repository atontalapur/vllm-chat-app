"""Validate an eval set before anything downstream reads it.

    python pipeline/eval/validate.py pipeline/eval/worldcup-v1.jsonl

Exit code is non-zero on any error, so CI can run it. The checks are the ones
that catch silent scoring bugs, not style:

- every line is one JSON object with the required fields and nothing unknown
- ids are unique
- the gold answer passes its own string checks (contains every `must_include`,
  none of `must_not_include`), otherwise the item can never score full marks
- `must_include` holds no bare numerals: "5" matches "2015", so a number on
  its own is a check that passes by accident. Put numeric facts in `judge`.
- an item has at least one string check or one judge statement
- `valid_through`, when present, is a four-digit year

Matching is case-insensitive on whitespace-collapsed text, the same
normalisation the eval runner (S2-3) must use. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED = {"id", "prompt", "must_include", "must_not_include", "judge", "gold", "notes"}
OPTIONAL = {"valid_through"}
JUDGE_KEYS = {"must_state", "must_not_claim"}


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def check_item(item: object, line_no: int) -> list[str]:
    where = f"line {line_no}"
    if not isinstance(item, dict):
        return [f"{where}: not a JSON object"]
    errors: list[str] = []
    item_id = item.get("id", "<no id>")
    where = f"{where} ({item_id})"

    missing = REQUIRED - item.keys()
    unknown = item.keys() - REQUIRED - OPTIONAL
    if missing:
        errors.append(f"{where}: missing fields {sorted(missing)}")
    if unknown:
        errors.append(f"{where}: unknown fields {sorted(unknown)}")
    if missing:
        return errors

    for key in ("id", "prompt", "gold", "notes"):
        if not isinstance(item[key], str) or not item[key].strip():
            errors.append(f"{where}: {key} must be a non-empty string")
    for key in ("must_include", "must_not_include"):
        value = item[key]
        if not isinstance(value, list) or not all(isinstance(s, str) and s for s in value):
            errors.append(f"{where}: {key} must be a list of non-empty strings")

    judge = item["judge"]
    if not isinstance(judge, dict) or set(judge) != JUDGE_KEYS:
        errors.append(f"{where}: judge must have exactly {sorted(JUDGE_KEYS)}")
    else:
        for key in JUDGE_KEYS:
            value = judge[key]
            if not isinstance(value, list) or not all(isinstance(s, str) and s for s in value):
                errors.append(f"{where}: judge.{key} must be a list of non-empty strings")

    if errors:
        return errors

    gold = normalise(item["gold"])
    for needle in item["must_include"]:
        if re.fullmatch(r"[\d,.]+", needle):
            errors.append(f"{where}: bare numeral {needle!r} in must_include; move it to judge")
        if normalise(needle) not in gold:
            errors.append(f"{where}: gold does not contain must_include {needle!r}")
    for needle in item["must_not_include"]:
        if normalise(needle) in gold:
            errors.append(f"{where}: gold contains must_not_include {needle!r}")

    if not item["must_include"] and not judge["must_state"]:
        errors.append(f"{where}: no positive check at all (must_include and must_state empty)")

    valid_through = item.get("valid_through")
    if valid_through is not None and not (
        isinstance(valid_through, str) and re.fullmatch(r"\d{4}", valid_through)
    ):
        errors.append(f"{where}: valid_through must be a four-digit year string")

    return errors


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    seen: dict[str, int] = {}
    count = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            errors.append(f"line {line_no}: blank line; JSONL must be one object per line")
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg} at col {exc.colno})")
            continue
        count += 1
        errors.extend(check_item(item, line_no))
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            item_id = item["id"]
            if item_id in seen:
                first = seen[item_id]
                errors.append(f"line {line_no}: duplicate id {item_id!r} (first on line {first})")
            seen[item_id] = line_no
    if count == 0:
        errors.append("file has no items")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="eval set, one JSON object per line")
    args = parser.parse_args()

    errors = validate(args.path)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        print(f"{args.path}: {len(errors)} error(s)", file=sys.stderr)
        return 1
    n = sum(1 for line in args.path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"{args.path}: {n} items, ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
