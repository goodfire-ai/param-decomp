#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic"]
# ///
"""Tinder-style keyboard UI for sifting PRs and commits into paper-worthy / not."""

import json
import subprocess
import sys
from pathlib import Path

import anthropic

OUTPUT_FILE = Path("paper_prs.json")
STATE_FILE = Path(".pr_sift_state.json")

PR_FILE = Path("/tmp/spd_prs.json")
COMMITS_FILE = Path("/tmp/spd_direct_commits.json")
DESC_CACHE_FILE = Path(".pr_sift_descriptions.json")

client = anthropic.Anthropic()


def get_diff(item: dict) -> str:
    item_id = item["id"]
    if item_id.startswith("pr-"):
        pr_num = item_id.removeprefix("pr-")
        result = subprocess.run(["gh", "pr", "diff", pr_num], capture_output=True, text=True)
    else:
        commit_hash = item_id.removeprefix("commit-")
        result = subprocess.run(["git", "show", commit_hash], capture_output=True, text=True)
    return result.stdout[:12000]  # truncate huge diffs


def describe_item(item: dict, cache: dict) -> str:
    if item["id"] in cache:
        return cache[item["id"]]

    diff = get_diff(item)
    if not diff.strip():
        return "(could not fetch diff)"

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    "Describe this code change in 2-3 sentences. "
                    "Focus on what it does and why it matters. Be terse.\n\n"
                    f"Title: {item['title']}\n\n"
                    f"```diff\n{diff}\n```"
                ),
            }
        ],
    )
    desc = resp.content[0].text
    cache[item["id"]] = desc
    save_desc_cache(cache)
    return desc


def load_desc_cache() -> dict:
    if DESC_CACHE_FILE.exists():
        with open(DESC_CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_desc_cache(cache: dict) -> None:
    with open(DESC_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def raw_print(*args: str, end: str = "\r\n") -> None:
    sys.stdout.write(" ".join(str(a) for a in args) + end)
    sys.stdout.flush()


def load_items() -> list[dict]:
    """Load PRs and direct commits into a unified list sorted by date."""
    items: list[dict] = []

    with open(PR_FILE) as f:
        for pr in json.load(f):
            items.append(
                {
                    "id": f"pr-{pr['number']}",
                    "kind": "PR",
                    "title": f"#{pr['number']}  {pr['title']}",
                    "date": pr["mergedAt"],
                    "additions": pr.get("additions", 0),
                    "deletions": pr.get("deletions", 0),
                    "body": pr.get("body") or "",
                }
            )

    if COMMITS_FILE.exists():
        with open(COMMITS_FILE) as f:
            for c in json.load(f):
                items.append(
                    {
                        "id": f"commit-{c['hash']}",
                        "kind": "commit",
                        "title": f"{c['hash'][:8]}  {c['subject']}",
                        "date": c["date"],
                        "additions": c.get("additions", 0),
                        "deletions": c.get("deletions", 0),
                        "body": "",
                    }
                )

    items.sort(key=lambda p: p["date"])
    return items


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"decisions": {}, "index": 0}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def save_results(items: list[dict], decisions: dict) -> None:
    worthy = [
        {"id": it["id"], "title": it["title"], "date": it["date"]}
        for it in items
        if decisions.get(it["id"]) == "yes"
    ]
    worthy.sort(key=lambda p: p["date"])
    with open(OUTPUT_FILE, "w") as f:
        json.dump(worthy, f, indent=2)


def render_item(item: dict, idx: int, total: int, decisions: dict, desc_cache: dict) -> None:
    n_yes = sum(1 for v in decisions.values() if v == "yes")
    n_no = sum(1 for v in decisions.values() if v == "no")
    n_skip = sum(1 for v in decisions.values() if v == "skip")

    decision = decisions.get(item["id"])
    marker = ""
    if decision == "yes":
        marker = "  \033[32m[ACCEPTED]\033[0m"
    elif decision == "no":
        marker = "  \033[31m[REJECTED]\033[0m"
    elif decision == "skip":
        marker = "  \033[33m[SKIPPED]\033[0m"

    kind_color = "\033[36m" if item["kind"] == "PR" else "\033[35m"
    kind_label = f"{kind_color}{item['kind']}\033[0m"

    raw_print("\033[2J\033[H", end="")  # clear screen
    raw_print(
        f"\033[1m {idx + 1}/{total} \033[0m  "
        f"\033[32m{n_yes} yes\033[0m · \033[31m{n_no} no\033[0m · \033[33m{n_skip} skip\033[0m"
    )
    raw_print("─" * 70)
    raw_print(f"{kind_label}  {item['title']}{marker}")
    adds = item["additions"]
    dels = item["deletions"]
    raw_print(f"\033[2m{item['date'][:10]}  \033[32m+{adds}\033[0m\033[2m/\033[31m-{dels}\033[0m")
    raw_print()

    body = item["body"].strip()
    if not body:
        raw_print("  \033[2m(generating description...)\033[0m", end="")
        body = "\033[3m" + describe_item(item, desc_cache) + "\033[0m"
        # Re-render now that we have the description
        raw_print("\033[2J\033[H", end="")
        raw_print(
            f"\033[1m {idx + 1}/{total} \033[0m  "
            f"\033[32m{n_yes} yes\033[0m · \033[31m{n_no} no\033[0m · \033[33m{n_skip} skip\033[0m"
        )
        raw_print("─" * 70)
        raw_print(f"{kind_label}  {item['title']}{marker}")
        raw_print(
            f"\033[2m{item['date'][:10]}  \033[32m+{adds}\033[0m\033[2m/\033[31m-{dels}\033[0m"
        )
        raw_print()

    lines = body.split("\n")[:15]
    for line in lines:
        raw_print(f"  {line}")
    if len(body.split("\n")) > 15:
        raw_print(f"  \033[2m... ({len(body.split(chr(10)))} lines total)\033[0m")

    raw_print()
    raw_print("─" * 70)
    raw_print(
        "  \033[31m←  / n\033[0m  not worthy    "
        "\033[32m→  / y\033[0m  paper-worthy    "
        "\033[33ms\033[0m  skip"
    )
    raw_print(
        "  \033[2mj/↓\033[0m  next (no decision)  "
        "\033[2mk/↑\033[0m  prev              "
        "\033[2mq\033[0m  save & quit"
    )


def main() -> None:
    items = load_items()
    state = load_state()
    decisions: dict = state["decisions"]
    idx: int = state["index"]
    desc_cache = load_desc_cache()

    if not items:
        print("No items found.")
        return

    idx = min(idx, len(items) - 1)

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        while True:
            render_item(items[idx], idx, len(items), decisions, desc_cache)

            ch = sys.stdin.read(1)

            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "C":  # right arrow
                        ch = "y"
                    elif ch3 == "D":  # left arrow
                        ch = "n"
                    elif ch3 == "B":  # down arrow
                        ch = "j"
                    elif ch3 == "A":  # up arrow
                        ch = "k"

            if ch in ("q", "\x03"):
                break
            elif ch == "y":
                decisions[items[idx]["id"]] = "yes"
                if idx < len(items) - 1:
                    idx += 1
            elif ch == "n":
                decisions[items[idx]["id"]] = "no"
                if idx < len(items) - 1:
                    idx += 1
            elif ch == "s":
                decisions[items[idx]["id"]] = "skip"
                if idx < len(items) - 1:
                    idx += 1
            elif ch == "j":
                if idx < len(items) - 1:
                    idx += 1
            elif ch == "k" and idx > 0:
                idx -= 1

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    state["decisions"] = decisions
    state["index"] = idx
    save_state(state)
    save_results(items, decisions)

    n_yes = sum(1 for v in decisions.values() if v == "yes")
    n_no = sum(1 for v in decisions.values() if v == "no")
    n_skip = sum(1 for v in decisions.values() if v == "skip")
    n_unseen = len(items) - len(decisions)

    print(f"\n\033[1mSaved!\033[0m  {n_yes} yes · {n_no} no · {n_skip} skip · {n_unseen} unseen")
    print(f"  Results: {OUTPUT_FILE}")
    print(f"  State:   {STATE_FILE}  (resume with `python pr_sift.py`)")


if __name__ == "__main__":
    main()
