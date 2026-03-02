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
        wandbRunId: "goodfire/spd/runs/s-82ffb969",
        notes: "Thomas - pile_llama_simple_mlp-4L",
    },
    {
        wandbRunId: "goodfire/spd/s-55ea3f9b",
        notes: "Jose - pile_llama_simple_mlp-4L",
        clusterMappings: [
            {
                path: "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-70b28465/cluster_mapping.json",
                notes: "All layers, 9100 iterations",
            },
        ],
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
