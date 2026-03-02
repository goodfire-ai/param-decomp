/**
 * Utilities for displaying run IDs from the registry.
 */

const DEFAULT_ENTITY_PROJECT = "goodfire/spd";

/**
 * Formats a wandb run id for display.
 * Shows just the 8-char run id if it's from "goodfire/spd",
 * otherwise shows the full path.
 */
export function formatRunIdForDisplay(wandbRunId: string): string {
    if (wandbRunId.startsWith(`${DEFAULT_ENTITY_PROJECT}/`)) {
        // Extract just the run id (last segment)
        const parts = wandbRunId.split("/");
        return parts[parts.length - 1];
    }
    return wandbRunId;
}
