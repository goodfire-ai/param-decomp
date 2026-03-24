/**
 * Canonical SPD runs for the run picker.
 *
 * Static data (name, notes) renders instantly in the UI.
 * Dynamic data (architecture, availability) is hydrated from the backend.
 */

export type ClusterMappingEntry = { path: string; notes: string };

export type RegistryEntry = {
    wandbRunId: string;
    name?: string;
    notes?: string;
    clusterMappings?: ClusterMappingEntry[];
};

const DEFAULT_ENTITY_PROJECT = "goodfire/spd";

export const CANONICAL_RUNS: RegistryEntry[] = [
    {
        name: "Jose",
        wandbRunId: "goodfire/spd/s-55ea3f9b",
        notes: "pile_llama_simple_mlp-4L",
        clusterMappings: [
            {
                path: "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-70b28465/cluster_mapping.json",
                notes: "All layers, iteration 9100",
            },
            {
                path: "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-7e8b960e/cluster_mapping_alpha10_i3000.json",
                notes: "All layers, iteration 3000, α 10"
            },
            {
                path: "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-eae05b96/cluster_mapping_alpha2_i8000.json",
                notes: "All layers, iteration 8000, α 2"
            },
        ],
    },
    {
        name: "Thomas",
        wandbRunId: "goodfire/spd/s-82ffb969",
        notes: "pile_llama_simple_mlp-4L",
        clusterMappings: [
            {
                path: "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-f9cc81c8/cluster_mapping.json",
                notes: "All layers, iteration 9100",
            },
        ],
    },
    {
        name: "Thomas",
        wandbRunId: "goodfire/spd/s-82ffb969",
        notes: "pile_llama_simple_mlp-4L",
        clusterMappings: [
            {
                path: "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-f9cc81c8/cluster_mapping.json",
                notes: "All layers, 9100 iterations",
            },
        ],
    },
    {
        name: "finetune",
        wandbRunId: "goodfire/spd/s-17805b61",
        notes: "finetune",
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
        const parts = wandbRunId.split("/");
        return parts[parts.length - 1];
    }
    return wandbRunId;
}
