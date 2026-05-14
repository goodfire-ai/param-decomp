#!/usr/bin/env bash
# Migrate existing `snapshot/<id>` branches on origin to `refs/runs/snapshot/<id>` refs.
#
# Background: snapshot/* branches were used to checkpoint the code state of each run,
# but because branches live under refs/heads/* every client's `git fetch` pulls them
# down by default — slowing `git pull` over time. Moving them to refs/runs/* (a custom
# namespace ignored by default `git fetch`) keeps the commits accessible to anyone who
# explicitly asks for them, while invisibly shrinking the default sync.
#
# What this does, per snapshot branch on origin:
#   1. Push the same commit to refs/runs/snapshot/<id>
#   2. Delete refs/heads/snapshot/<id> on origin
# The commit object itself is unchanged. Any saved metadata pointing at the old branch
# name will no longer resolve; only the SHA / new ref will.
#
# Run once. Idempotent — safe to re-run; already-migrated branches are skipped.
#
# Usage:
#   ./scripts/migrate_snapshot_branches_to_runs_refs.sh           # dry run
#   ./scripts/migrate_snapshot_branches_to_runs_refs.sh --apply   # actually push/delete

set -euo pipefail

APPLY=0
if [[ "${1-}" == "--apply" ]]; then
    APPLY=1
fi

if [[ $APPLY -eq 0 ]]; then
    echo "DRY RUN. Pass --apply to perform the migration."
fi
echo

# Pull just the refs we care about — avoid fetching all of origin.
entries="$(git ls-remote origin 'refs/heads/snapshot/*')"

if [[ -z "$entries" ]]; then
    echo "No snapshot/* branches found on origin. Nothing to do."
    exit 0
fi

n_entries="$(printf '%s\n' "$entries" | wc -l | tr -d ' ')"
echo "Found ${n_entries} snapshot branches on origin."
echo

migrated=0
skipped=0
failed=0

while IFS=$'\t' read -r sha ref; do
    # ref is like: refs/heads/snapshot/<id>
    id="${ref#refs/heads/snapshot/}"
    new_ref="refs/runs/snapshot/${id}"

    # Skip if new ref already exists and points at the same commit.
    existing="$(git ls-remote origin "$new_ref" | awk '{print $1}')"
    if [[ "$existing" == "$sha" ]]; then
        echo "[skip]  $ref  (already migrated to $new_ref)"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ $APPLY -eq 1 ]]; then
        if git push origin "${sha}:${new_ref}" >/dev/null 2>&1 \
            && git push origin --delete "$ref" >/dev/null 2>&1; then
            echo "[done]  $ref -> $new_ref  ($sha)"
            migrated=$((migrated + 1))
        else
            echo "[FAIL]  $ref -> $new_ref"
            failed=$((failed + 1))
        fi
    else
        echo "[plan]  $ref -> $new_ref  ($sha)"
        migrated=$((migrated + 1))
    fi
done <<< "$entries"

echo
echo "Summary: ${migrated} migrated, ${skipped} already migrated, ${failed} failed."

if [[ $APPLY -eq 0 ]]; then
    echo
    echo "Re-run with --apply to perform the migration."
fi
