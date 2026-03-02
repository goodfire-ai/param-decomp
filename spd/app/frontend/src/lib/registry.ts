/**
 * Registry of canonical SPD runs for quick access in the app.
 *
 * Static data renders instantly; availability + architecture are hydrated
 * lazily from the backend via /api/run_registry.
 */

export type RegistryEntry = {
    wandbRunId: string;
    name?: string;
    notes?: string;
};

const DEFAULT_ENTITY_PROJECT = "goodfire/spd";

export const CANONICAL_RUNS: RegistryEntry[] = [
    {
        name: "Thomas",
        wandbRunId: "goodfire/spd/s-82ffb969",
        notes: "pile_llama_simple_mlp-4L",
    },
    {
        name: "Jose",
        wandbRunId: "goodfire/spd/s-55ea3f9b",
        notes: "pile_llama_simple_mlp-4L",
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
