"""Seed the Jira project from docs/jira/backlog.json.

The backlog file is the source of truth for epics, sprints, and stories. This
script pushes it into Jira Cloud so the board and the plan document cannot
drift: edit the JSON, re-run, and only the missing issues are created.

Three modes:

    python scripts/jira_seed.py --dry-run          # print what would be created
    python scripts/jira_seed.py --csv out.csv      # Jira CSV import, no credentials
    python scripts/jira_seed.py                    # create via REST API

The API mode needs, in the environment or .env:

    JIRA_BASE_URL     https://<site>.atlassian.net
    JIRA_EMAIL        the Atlassian account email
    JIRA_API_TOKEN    from https://id.atlassian.com/manage-profile/security/api-tokens
    JIRA_PROJECT_KEY  e.g. CFT

Created issue keys are remembered in docs/jira/.keys.json (gitignored) so a
second run skips anything already created rather than duplicating it.

Standard library only, matching scripts/loadtest.py: no install step.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BACKLOG = ROOT / "docs" / "jira" / "backlog.json"
KEYS = ROOT / "docs" / "jira" / ".keys.json"

# Jira Cloud names the estimate field differently depending on project type.
# Team-managed projects use "Story point estimate"; company-managed use
# "Story Points". Discovered at runtime rather than hardcoded, because the
# customfield_NNNNN id differs per site.
STORY_POINT_FIELD_NAMES = ("Story point estimate", "Story Points")


def load_env_file() -> None:
    """Minimal .env loader so the script works on the GPU box without dotenv."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"'))


def adf(description: str, acceptance: list[str]) -> dict[str, Any]:
    """Atlassian Document Format: v3 of the API requires it for description."""
    content: list[dict[str, Any]] = [
        {"type": "paragraph", "content": [{"type": "text", "text": description}]}
    ]
    if acceptance:
        content.append(
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Acceptance criteria", "marks": [{"type": "strong"}]}
                ],
            }
        )
        content.append(
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {"type": "paragraph", "content": [{"type": "text", "text": item}]}
                        ],
                    }
                    for item in acceptance
                ],
            }
        )
    return {"type": "doc", "version": 1, "content": content}


class Jira:
    def __init__(self, base_url: str, email: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        auth = base64.b64encode(f"{email}:{token}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {auth}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        # noqa S310: the base URL is operator-supplied (JIRA_BASE_URL) and is
        # always the Atlassian site, never a file: or custom scheme.
        req = urllib.request.Request(  # noqa: S310
            f"{self.base_url}{path}", data=data, method=method, headers=self.headers
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:800]
            sys.exit(f"{method} {path} -> {exc.code}\n{detail}")

    def story_points_field(self) -> str | None:
        fields = self.request("GET", "/rest/api/3/field")
        by_name = {f["name"]: f["id"] for f in fields}
        for name in STORY_POINT_FIELD_NAMES:
            if name in by_name:
                return by_name[name]
        return None

    def board_id(self, project_key: str) -> int:
        # Team-managed projects report their board as type "simple", not
        # "scrum", so no type filter: sprint support is decided by the
        # project's Sprints feature, and the sprint create call reports that.
        boards = self.request("GET", f"/rest/agile/1.0/board?projectKeyOrId={project_key}")
        values = boards.get("values", [])
        if not values:
            sys.exit(f"no board found for {project_key}")
        return int(values[0]["id"])

    def create_issue(self, fields: dict[str, Any]) -> str:
        created = self.request("POST", "/rest/api/3/issue", {"fields": fields})
        return str(created["key"])

    def create_sprint(self, board_id: int, name: str, goal: str) -> int:
        sprint = self.request(
            "POST",
            "/rest/agile/1.0/sprint",
            {"name": name, "goal": goal, "originBoardId": board_id},
        )
        return int(sprint["id"])

    def move_to_sprint(self, sprint_id: int, keys: list[str]) -> None:
        # The endpoint accepts at most 50 issues per call.
        for i in range(0, len(keys), 50):
            self.request(
                "POST", f"/rest/agile/1.0/sprint/{sprint_id}/issue", {"issues": keys[i : i + 50]}
            )


def load_state() -> dict[str, Any]:
    if KEYS.exists():
        return json.loads(KEYS.read_text())
    return {"epics": {}, "stories": {}, "sprints": {}}


def save_state(state: dict[str, Any]) -> None:
    KEYS.write_text(json.dumps(state, indent=2) + "\n")


def write_csv(backlog: dict[str, Any], path: Path) -> None:
    """Jira CSV import format. Map 'Parent' to the parent field in the wizard."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["Issue Type", "Summary", "Description", "Story Points", "Sprint", "Parent", "Issue ID"]
        )
        for epic in backlog["epics"]:
            writer.writerow(["Epic", epic["summary"], epic["description"], "", "", "", epic["id"]])
        sprint_names = {s["id"]: s["name"] for s in backlog["sprints"]}
        for story in backlog["stories"]:
            desc = story["description"]
            if story["acceptance"]:
                desc += "\n\nAcceptance criteria:\n" + "\n".join(
                    f"- {a}" for a in story["acceptance"]
                )
            writer.writerow(
                [
                    story["type"],
                    story["summary"],
                    desc,
                    story["points"] or "",
                    sprint_names[story["sprint"]],
                    story["epic"],
                    story["id"],
                ]
            )
    print(f"wrote {path}")


def dry_run(backlog: dict[str, Any]) -> None:
    epics = {e["id"]: e["summary"] for e in backlog["epics"]}
    for sprint in backlog["sprints"]:
        stories = [s for s in backlog["stories"] if s["sprint"] == sprint["id"]]
        points = sum(s["points"] or 0 for s in stories)
        print(f"\n{sprint['name']}  ({points} pts)")
        print(f"  goal: {sprint['goal']}")
        for s in stories:
            pts = f"{s['points']}pt" if s["points"] else "spike"
            print(f"  [{s['type']:<5}] {s['id']:<5} {s['summary']}  ({pts}, {epics[s['epic']]})")


def seed(backlog: dict[str, Any]) -> None:
    load_env_file()
    missing = [
        k
        for k in ("JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECT_KEY")
        if not os.environ.get(k)
    ]
    if missing:
        sys.exit(f"missing env: {', '.join(missing)} (see module docstring)")

    jira = Jira(os.environ["JIRA_BASE_URL"], os.environ["JIRA_EMAIL"], os.environ["JIRA_API_TOKEN"])
    project = os.environ["JIRA_PROJECT_KEY"]
    state = load_state()

    points_field = jira.story_points_field()
    if points_field is None:
        print("warning: no story points field found; points will not be set")

    for epic in backlog["epics"]:
        if epic["id"] in state["epics"]:
            continue
        key = jira.create_issue(
            {
                "project": {"key": project},
                "issuetype": {"name": "Epic"},
                "summary": epic["summary"],
                "description": adf(epic["description"], []),
            }
        )
        state["epics"][epic["id"]] = key
        save_state(state)
        print(f"epic  {epic['id']} -> {key}")

    for story in backlog["stories"]:
        if story["id"] in state["stories"]:
            continue
        fields: dict[str, Any] = {
            "project": {"key": project},
            "issuetype": {"name": story["type"]},
            "summary": story["summary"],
            "description": adf(story["description"], story["acceptance"]),
            "parent": {"key": state["epics"][story["epic"]]},
            "labels": [story["epic"], f"sprint-{story['sprint']}"],
        }
        if points_field and story["points"] is not None:
            fields[points_field] = story["points"]
        key = jira.create_issue(fields)
        state["stories"][story["id"]] = key
        save_state(state)
        print(f"story {story['id']} -> {key}")

    board = jira.board_id(project)
    for sprint in backlog["sprints"]:
        sid = str(sprint["id"])
        if sid not in state["sprints"]:
            state["sprints"][sid] = jira.create_sprint(board, sprint["name"], sprint["goal"])
            save_state(state)
            print(f"sprint {sprint['name']} -> id {state['sprints'][sid]}")
        keys = [
            state["stories"][s["id"]] for s in backlog["stories"] if s["sprint"] == sprint["id"]
        ]
        jira.move_to_sprint(state["sprints"][sid], keys)

    print(
        f"\ndone. {len(state['epics'])} epics, {len(state['stories'])} stories, "
        f"{len(state['sprints'])} sprints. Keys in {KEYS.relative_to(ROOT)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print the plan, touch nothing")
    parser.add_argument("--csv", type=Path, help="write a Jira CSV import file instead")
    args = parser.parse_args()

    backlog = json.loads(BACKLOG.read_text())
    # Jira rejects longer names at sprint creation, after issues already exist.
    too_long = [s["name"] for s in backlog["sprints"] if len(s["name"]) >= 30]
    if too_long:
        sys.exit(f"sprint names must be under 30 characters: {too_long}")
    if args.dry_run:
        dry_run(backlog)
    elif args.csv:
        write_csv(backlog, args.csv)
    else:
        seed(backlog)


if __name__ == "__main__":
    main()
