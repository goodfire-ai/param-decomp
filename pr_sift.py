#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["anthropic"]
# ///
"""Tinder-style keyboard UI for sifting PRs and commits into paper-worthy / not.

Usage: uv run pr_sift.py
"""

import json
import re
import subprocess
import sys
import termios
import tty
from pathlib import Path

OUTPUT_FILE = Path("paper_prs.json")
STATE_FILE = Path(".pr_sift_state.json")

PR_FILE = Path("/tmp/spd_prs.json")
DESC_CACHE_FILE = Path(".pr_sift_descriptions.json")

ALWAYS_SELECTED = {"claude-spd1"}


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
    import anthropic

    if item["id"] in cache:
        return cache[item["id"]]

    diff = get_diff(item)
    if not diff.strip():
        return "(could not fetch diff)"

    resp = anthropic.Anthropic().messages.create(
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


def get_all_authors() -> list[str]:
    """Get sorted list of all unique PR authors."""
    with open(PR_FILE) as f:
        prs = json.load(f)
    authors = sorted({pr["author"]["login"] for pr in prs}, key=str.lower)
    return authors


def select_authors(authors: list[str]) -> set[str]:
    """Interactive multi-select for authors. Returns selected set."""
    selected = {a for a in authors if a in ALWAYS_SELECTED}
    cursor = 0

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        while True:
            raw_print("\033[2J\033[H", end="")
            raw_print("\033[1mSelect authors to include\033[0m")
            raw_print("─" * 40)

            for i, author in enumerate(authors):
                prefix = "\033[7m" if i == cursor else ""
                check = "\033[32m[x]\033[0m" if author in selected else "[ ]"
                raw_print(f"  {prefix}{check} {author}\033[0m")

            raw_print()
            raw_print("─" * 40)
            raw_print(
                "  \033[2mspace\033[0m toggle  \033[2ma\033[0m all  \033[2mz\033[0m none  \033[2menter\033[0m confirm"
            )

            ch = sys.stdin.read(1)
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":  # up
                        cursor = max(0, cursor - 1)
                    elif ch3 == "B":  # down
                        cursor = min(len(authors) - 1, cursor + 1)
            elif ch == " ":
                author = authors[cursor]
                if author in selected:
                    selected.discard(author)
                else:
                    selected.add(author)
            elif ch == "a":
                selected = set(authors)
            elif ch == "z":
                selected.clear()
            elif ch in ("\r", "\n"):
                break
            elif ch in ("q", "\x03"):
                sys.exit(0)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return selected


def build_login_to_git_names() -> dict[str, set[str]]:
    """Map GitHub logins to git author names using squash merge commits.

    Squash merges have subjects like 'Title (#123)'. We match the PR number
    to the PR JSON to get the GitHub login, and record the git author name.
    """
    with open(PR_FILE) as f:
        pr_num_to_login = {pr["number"]: pr["author"]["login"] for pr in json.load(f)}

    mapping: dict[str, set[str]] = {}
    result = subprocess.run(
        ["git", "log", "dev", "--no-merges", "--format=%an|%s"],
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().split("\n"):
        if "|" not in line:
            continue
        git_name, subject = line.split("|", 1)
        m = re.search(r"\(#(\d+)\)$", subject)
        if m:
            pr_num = int(m.group(1))
            if pr_num in pr_num_to_login:
                login = pr_num_to_login[pr_num]
                mapping.setdefault(login, set()).add(git_name)

    return mapping


def get_direct_commits(selected_git_names: set[str]) -> list[dict]:
    """Fetch non-PR commits on dev by the given git author names."""
    if not selected_git_names:
        return []

    # Get all non-merge commits on dev
    result = subprocess.run(
        ["git", "log", "dev", "--no-merges", "--format=%H|%an|%aI|%s"],
        capture_output=True,
        text=True,
    )
    commits = []
    for line in result.stdout.strip().split("\n"):
        if "|" not in line:
            continue
        h, git_name, date, subject = line.split("|", 3)
        # Skip squash merge commits (they're already PRs)
        if re.search(r"\(#\d+\)$", subject):
            continue
        if git_name not in selected_git_names:
            continue

        # Get line stats
        stat = subprocess.run(
            ["git", "show", "--format=", "--numstat", h],
            capture_output=True,
            text=True,
        )
        adds = dels = 0
        for sline in stat.stdout.strip().split("\n"):
            parts = sline.split()
            if len(parts) >= 2 and parts[0] != "-":
                adds += int(parts[0])
                dels += int(parts[1])

        commits.append(
            {
                "id": f"commit-{h[:10]}",
                "kind": "commit",
                "title": f"{h[:8]}  {subject}",
                "author": git_name,
                "date": date,
                "additions": adds,
                "deletions": dels,
                "body": "",
            }
        )
    return commits


def load_items(selected_authors: set[str]) -> list[dict]:
    """Load PRs and direct commits into a unified list sorted by date."""
    items: list[dict] = []

    with open(PR_FILE) as f:
        for pr in json.load(f):
            if pr["author"]["login"] not in selected_authors:
                continue
            items.append(
                {
                    "id": f"pr-{pr['number']}",
                    "kind": "PR",
                    "title": f"#{pr['number']}  {pr['title']}",
                    "author": pr["author"]["login"],
                    "date": pr["mergedAt"],
                    "additions": pr.get("additions", 0),
                    "deletions": pr.get("deletions", 0),
                    "body": pr.get("body") or "",
                }
            )

    # Map selected GitHub logins to git author names for direct commits
    login_to_names = build_login_to_git_names()
    selected_git_names: set[str] = set()
    for login in selected_authors:
        selected_git_names.update(login_to_names.get(login, set()))

    items.extend(get_direct_commits(selected_git_names))
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


def render_item(item: dict, idx: int, total: int, decisions: dict, desc_cache: dict | None) -> None:
    n_yes = sum(1 for v in decisions.values() if v == "yes")
    n_no = sum(1 for v in decisions.values() if v == "no")

    decision = decisions.get(item["id"])
    marker = ""
    if decision == "yes":
        marker = "  \033[32m[ACCEPTED]\033[0m"
    elif decision == "no":
        marker = "  \033[31m[REJECTED]\033[0m"

    kind_color = "\033[36m" if item["kind"] == "PR" else "\033[35m"
    kind_label = f"{kind_color}{item['kind']}\033[0m"

    raw_print("\033[2J\033[H", end="")  # clear screen
    raw_print(
        f"\033[1m {idx + 1}/{total} \033[0m  \033[32m{n_yes} yes\033[0m · \033[31m{n_no} no\033[0m"
    )
    raw_print("─" * 70)
    raw_print(f"{kind_label}  {item['title']}{marker}")
    adds = item["additions"]
    dels = item["deletions"]
    raw_print(
        f"\033[2m{item['date'][:10]}  {item['author']}  \033[32m+{adds}\033[0m\033[2m/\033[31m-{dels}\033[0m"
    )
    raw_print()

    body = item["body"].strip()
    if not body and desc_cache is not None and item["id"] in desc_cache:
        body = "\033[3m" + desc_cache[item["id"]] + "\033[0m"

    lines = body.split("\n")[:15] if body else []
    for line in lines:
        raw_print(f"  {line}")
    if len(body.split("\n")) > 15:
        raw_print(f"  \033[2m... ({len(body.split(chr(10)))} lines total)\033[0m")

    if not lines:
        raw_print("  \033[2m(no description — press i to generate)\033[0m")

    raw_print()
    raw_print("─" * 70)
    raw_print(
        "  \033[31m←\033[0m not worthy    "
        "\033[32m→\033[0m paper-worthy    "
        "\033[2m⌫\033[0m back    "
        "\033[2mi\033[0m describe    "
        "\033[2mq\033[0m quit"
    )


def fetch_prs() -> None:
    """Fetch all merged PRs from GitHub if not already cached."""
    if PR_FILE.exists():
        return
    print("Fetching merged PRs from GitHub...")
    with open(PR_FILE, "w") as f:
        subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "merged",
                "--limit",
                "500",
                "--json",
                "number,title,author,mergedAt,body,additions,deletions",
            ],
            stdout=f,
            check=True,
        )


def main() -> None:
    fetch_prs()
    authors = get_all_authors()
    selected_authors = select_authors(authors)
    if not selected_authors:
        print("No authors selected.")
        return

    items = load_items(selected_authors)
    state = load_state()
    decisions: dict = state["decisions"]
    idx: int = state["index"]
    desc_cache: dict | None = None

    if not items:
        print("No items found.")
        return

    idx = min(idx, len(items) - 1)

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
                        decisions[items[idx]["id"]] = "yes"
                        if idx < len(items) - 1:
                            idx += 1
                    elif ch3 == "D":  # left arrow
                        decisions[items[idx]["id"]] = "no"
                        if idx < len(items) - 1:
                            idx += 1
            elif ch == "\x7f" and idx > 0:  # backspace
                idx -= 1
            elif ch == "i":
                if desc_cache is None:
                    desc_cache = load_desc_cache()
                describe_item(items[idx], desc_cache)
                # re-render will pick it up from cache

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    state["decisions"] = decisions
    state["index"] = idx
    save_state(state)
    save_results(items, decisions)

    n_yes = sum(1 for v in decisions.values() if v == "yes")
    n_no = sum(1 for v in decisions.values() if v == "no")
    n_unseen = len(items) - len(decisions)

    print(f"\n\033[1m{n_yes} accepted\033[0m ({n_no} rejected, {n_unseen} unseen)\n")
    for it in items:
        if decisions.get(it["id"]) != "yes":
            continue
        title = it["title"]
        body = it["body"].strip()
        if not body and desc_cache:
            body = desc_cache.get(it["id"], "")
        summary = body.split("\n")[0][:80] if body else ""
        print(f"  \033[32m✓\033[0m {title}")
        if summary:
            print(f"    \033[2m{summary}\033[0m")
    print(f"\n  Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
