/** Types for the intervention forward pass feature */

/** Default eval PGD settings (distinct from training PGD which is an optimization regularizer) */
export const EVAL_PGD_N_STEPS = 4;
export const EVAL_PGD_STEP_SIZE = 1.0;

export type TokenPrediction = {
    token: string;
    token_id: number;
    prob: number;
    logit: number;
    target_prob: number;
    target_logit: number;
};

export type LabelPredictions = {
    position: number;
    ci: TokenPrediction;
    stochastic: TokenPrediction;
    adversarial: TokenPrediction;
    ablated: TokenPrediction | null;
};

export type InterventionResult = {
    input_tokens: string[];
    ci: TokenPrediction[][];
    stochastic: TokenPrediction[][];
    adversarial: TokenPrediction[][];
    ablated: TokenPrediction[][] | null;
    ci_loss: number;
    stochastic_loss: number;
    adversarial_loss: number;
    ablated_loss: number | null;
    label: LabelPredictions | null;
};

/** Persisted intervention run from the server */
export type InterventionRunSummary = {
    id: number;
    selected_nodes: string[]; // node keys (layer:seq:cIdx)
    result: InterventionResult;
    created_at: string;
};

/** Request to run and save an intervention */
export type RunInterventionRequest = {
    graph_id: number;
    selected_nodes: string[];
    nodes_to_ablate?: string[];
    top_k: number;
    adv_pgd: { n_steps: number; step_size: number };
};

// --- Frontend-only run lifecycle types ---

import { SvelteSet } from "svelte/reactivity";
import { isInterventableNode } from "./promptAttributionsTypes";

/** Draft run: cloned from a parent, editable node selection. No forwarded results yet. */
export type DraftRun = {
    kind: "draft";
    parentId: number;
    selectedNodes: SvelteSet<string>;
};

/** Baked run: forwarded and immutable. Wraps a persisted InterventionRunSummary. */
export type BakedRun = {
    kind: "baked";
    id: number;
    selectedNodes: Set<string>;
    result: InterventionResult;
    createdAt: string;
};

export type InterventionRun = DraftRun | BakedRun;

export type InterventionState = {
    runs: InterventionRun[];
    activeIndex: number;
};

/** Whether a run's selection is editable */
export function isRunEditable(run: InterventionRun): run is DraftRun {
    return run.kind === "draft";
}

/** Build initial InterventionState from persisted runs.
 * The first persisted run is the base run (all CI > 0 nodes), auto-created during graph computation. */
export function buildInterventionState(persistedRuns: InterventionRunSummary[]): InterventionState {
    if (persistedRuns.length === 0) throw new Error("Graph must have at least one intervention run (the base run)");
    const runs: InterventionRun[] = persistedRuns.map(
        (r): BakedRun => ({
            kind: "baked",
            id: r.id,
            selectedNodes: new Set(r.selected_nodes),
            result: r.result,
            createdAt: r.created_at,
        }),
    );
    return { runs, activeIndex: 0 };
}

/** Get all interventable node keys with CI > 0 from a nodeCiVals record */
export function getInterventableNodes(nodeCiVals: Record<string, number>): Set<string> {
    const nodes = new Set<string>();
    for (const [nodeKey, ci] of Object.entries(nodeCiVals)) {
        if (isInterventableNode(nodeKey) && ci > 0) nodes.add(nodeKey);
    }
    return nodes;
}
