/**
 * Registry of canonical SPD runs for quick access in the app.
 */

export type RegistryEntry = {
    /** Full wandb run id (e.g., "goodfire/spd/jyo9duz5") */
    wandbRunId: string;
    /** Optional notes about the run */
    notes?: string;
    /** Optional cluster mappings for the run */
    clusterMappings?: {
        path: string;
        notes: string;
    }[];
};

const DEFAULT_ENTITY_PROJECT = "goodfire/spd";

/**
 * Canonical runs registry - add new entries here.
 * These appear in the dropdown for quick selection.
 */
export const CANONICAL_RUNS: RegistryEntry[] = [
    {
        wandbRunId: "goodfire/spd/s-55ea3f9b",
        notes: "José - Primary canonical run candidate",
    },
    {
        wandbRunId: "goodfire/spd/s-275c8f21",
        notes: "Lucius' pile run Feb 11",
    },
    {
        wandbRunId: "goodfire/spd/s-eab2ace8",
        notes: "Oli's PPGD run, great metrics",
    },
    {
        wandbRunId: "goodfire/spd/s-892f140b",
        notes: "Lucius run, Jan 22",
    },
    {
        wandbRunId: "goodfire/spd/s-7884efcc",
        notes: "Lucius' new run, Jan 8",
    },
    {
        wandbRunId: "goodfire/spd/vjbol27n",
        notes: "Lucius' run, Dec 8",
        clusterMappings: [],
    },
    {
        wandbRunId: "goodfire/spd/278we8gk",
        notes: "Dan's initial run, Dec 6",
    },
    {
        wandbRunId: "goodfire/spd/jyo9duz5",
    },
    {
        wandbRunId: "goodfire/spd/5cr21lbs",
        clusterMappings: [],
    },
    {
        wandbRunId: "goodfire/spd/itmexlj0",
    },
    {
        wandbRunId: "goodfire/spd/33n6xjjt",
    },
];

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
