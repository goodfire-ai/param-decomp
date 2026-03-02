/** Types for the intervention forward pass feature */

export type InterventionNode = {
    layer: string;
    seq_pos: number;
    component_idx: number;
};

export type TokenPrediction = {
    token: string;
    token_id: number;
    spd_prob: number;
    target_prob: number;
    logit: number;
    target_logit: number;
};

export type InterventionResponse = {
    input_tokens: string[];
    predictions_per_position: TokenPrediction[][];
};

/** A forked intervention run with modified tokens */
export type ForkedInterventionRunSummary = {
    id: number;
    token_replacements: [number, number][]; // [(seq_pos, new_token_id), ...]
    result: InterventionResponse;
    created_at: string;
};

/** Persisted intervention run from the server */
export type InterventionRunSummary = {
    id: number;
    selected_nodes: string[]; // node keys (layer:seq:cIdx)
    result: InterventionResponse;
    masked_predictions: MaskedPredictions;
    created_at: string;
    forked_runs?: ForkedInterventionRunSummary[]; // child runs with modified tokens
};

/** Request to run and save an intervention */
export type RunInterventionRequest = {
    graph_id: number;
    text: string;
    selected_nodes: string[];
    top_k: number;
    adv_pgd_n_steps: number;
    adv_pgd_step_size: number;
};

export type TokenPred = {
    token: string;
    prob: number;
};

export type MaskedPredictions = {
    ci: TokenPred[][];
    stochastic: TokenPred[][];
    adversarial: TokenPred[][];
};

// --- Frontend-only run lifecycle types ---

import { SvelteSet } from "svelte/reactivity";
import { isInterventableNode } from "./promptAttributionsTypes";

/** Base run: synthesized from the graph's own data. All interventable nodes selected. Not editable. */
export type BaseRun = {
    kind: "base";
};

/** Draft run: cloned from a parent, editable node selection. No forwarded results yet. */
export type DraftRun = {
    kind: "draft";
    parentId: "base" | number;
    selectedNodes: SvelteSet<string>;
};

/** Baked run: forwarded and immutable. Wraps a persisted InterventionRunSummary. */
export type BakedRun = {
    kind: "baked";
    id: number;
    selectedNodes: Set<string>;
    result: InterventionResponse;
    maskedPredictions: MaskedPredictions;
    createdAt: string;
};

export type InterventionRun = BaseRun | DraftRun | BakedRun;

export type InterventionState = {
    runs: InterventionRun[];
    activeIndex: number;
};

/** Get the effective node selection for a run */
export function getRunSelection(run: InterventionRun, allInterventableNodes: Set<string>): Set<string> {
    switch (run.kind) {
        case "base":
            return allInterventableNodes;
        case "draft":
            return run.selectedNodes;
        case "baked":
            return run.selectedNodes;
    }
}

/** Whether a run's selection is editable */
export function isRunEditable(run: InterventionRun): run is DraftRun {
    return run.kind === "draft";
}

/** Build initial InterventionState from persisted runs */
export function buildInterventionState(persistedRuns: InterventionRunSummary[]): InterventionState {
    const runs: InterventionRun[] = [
        { kind: "base" },
        ...persistedRuns.map(
            (r): BakedRun => ({
                kind: "baked",
                id: r.id,
                selectedNodes: new Set(r.selected_nodes),
                result: r.result,
                maskedPredictions: r.masked_predictions,
                createdAt: r.created_at,
            }),
        ),
    ];
    return { runs, activeIndex: 0 };
}

/** Get all interventable node keys from a nodeCiVals record */
export function getInterventableNodes(nodeCiVals: Record<string, number>): Set<string> {
    const nodes = new Set<string>();
    for (const nodeKey of Object.keys(nodeCiVals)) {
        if (isInterventableNode(nodeKey)) nodes.add(nodeKey);
    }
    return nodes;
}
