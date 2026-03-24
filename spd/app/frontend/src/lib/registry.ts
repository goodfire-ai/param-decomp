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
                path: "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-bd99c7aa/cluster_mapping.json",
                notes: "exp_rank α=5 decay=0.8, iter 6351, 10M toks (MDL-optimal, best quality)",
            },
            {
                path: "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-651d85c4/cluster_mapping.json",
                notes: "exp_rank α=10 decay=0.8, iter 4423, 10M toks (MDL-optimal, tightest clusters)",
            },
            {
                path: "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-e8fb48bb/cluster_mapping.json",
                notes: "exp_rank α=2 decay=0.8, iter 7999, 10M toks (most merges)",
            },
            {
                path: "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-70b28465/cluster_mapping.json",
                notes: "range α=1, iter 9100, 500K toks (old, lower quality)",
            },
            {
                path: "/mnt/polished-lake/artifacts/mechanisms/spd/clustering/runs/c-eae05b96/cluster_mapping_alpha2_i8000.json",
                notes: "range α=2, iter 8000, 10M toks (old, lower quality)",
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
