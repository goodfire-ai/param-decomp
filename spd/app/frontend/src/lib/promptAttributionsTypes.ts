/** Types for the prompt attributions visualizer */

// Server API types

export type PromptPreview = {
    id: number;
    token_ids: number[];
    tokens: string[];
    preview: string;
    next_token_probs: (number | null)[]; // Probability of next token (last is null)
};

export type EdgeData = {
    src: string; // "layer:seq:cIdx"
    tgt: string; // "layer:seq:cIdx"
    val: number;
};

export type EdgeAttribution = {
    key: string; // "layer:seq:cIdx" for prompt or "layer:cIdx" for dataset
    value: number; // raw attribution value (positive or negative)
    normalizedMagnitude: number; // |value| / maxAbsValue, for color intensity (0-1)
    tokenStr: string | null; // resolved token string for embed/output layers
};

export type OutputProbability = {
    prob: number; // CI-masked (SPD model) probability
    logit: number; // CI-masked (SPD model) raw logit
    target_prob: number; // Target model probability
    target_logit: number; // Target model raw logit
    token: string;
};

export type CISnapshot = {
    step: number;
    total_steps: number;
    layers: string[];
    seq_len: number;
    initial_alive: number[][];
    current_alive: number[][];
    l0_total: number;
    loss: number;
};

export type GraphType = "standard" | "optimized" | "manual";

export type GraphData = {
    id: number;
    graphType: GraphType;
    tokens: string[];
    edges: EdgeData[];
    edgesBySource: Map<string, EdgeData[]>; // nodeKey -> edges where this node is source
    edgesByTarget: Map<string, EdgeData[]>; // nodeKey -> edges where this node is target
    // Absolute-target variant (∂|y|/∂x · x), null for old graphs
    edgesAbs: EdgeData[] | null;
    edgesAbsBySource: Map<string, EdgeData[]> | null;
    edgesAbsByTarget: Map<string, EdgeData[]> | null;
    outputProbs: Record<string, OutputProbability>; // key is "seq:cIdx"
    nodeCiVals: Record<string, number>; // node key -> CI value (or output prob for output nodes or 1 for wte node)
    nodeSubcompActs: Record<string, number>; // node key -> subcomponent activation (v_i^T @ a)
    maxAbsAttr: number; // max absolute edge value
    maxAbsAttrAbs: number | null; // max absolute edge value for abs-target variant
    maxAbsSubcompAct: number; // max absolute subcomponent activation for normalization
    l0_total: number; // total active components at current CI threshold
    optimization?: OptimizationResult;
};

/** Build edge indexes from flat edge array (single pass) */
export function buildEdgeIndexes(edges: EdgeData[]): {
    edgesBySource: Map<string, EdgeData[]>;
    edgesByTarget: Map<string, EdgeData[]>;
} {
    const edgesBySource = new Map<string, EdgeData[]>();
    const edgesByTarget = new Map<string, EdgeData[]>();

    for (const edge of edges) {
        const bySrc = edgesBySource.get(edge.src);
        if (bySrc) {
            bySrc.push(edge);
        } else {
            edgesBySource.set(edge.src, [edge]);
        }

        const byTgt = edgesByTarget.get(edge.tgt);
        if (byTgt) {
            byTgt.push(edge);
        } else {
            edgesByTarget.set(edge.tgt, [edge]);
        }
    }

    return { edgesBySource, edgesByTarget };
}

export type MaskType = "stochastic" | "ci";

export type CELossResult = {
    type: "ce";
    coeff: number;
    position: number;
    label_token: number;
    label_str: string;
};

export type KLLossResult = {
    type: "kl";
    coeff: number;
    position: number;
};

export type LogitLossResult = {
    type: "logit";
    coeff: number;
    position: number;
    label_token: number;
    label_str: string;
};

export type LossResult = CELossResult | KLLossResult | LogitLossResult;

export type OptimizationMetrics = {
    ci_masked_label_prob: number | null; // Probability of label under CI mask (CE loss only)
    stoch_masked_label_prob: number | null; // Probability of label under stochastic mask (CE loss only)
    adv_pgd_label_prob: number | null; // Probability of label under adversarial mask (CE loss only)
    l0_total: number; // Total L0 (active components)
};

export type PgdConfig = {
    n_steps: number;
    step_size: number;
};

export type OptimizationResult = {
    imp_min_coeff: number;
    steps: number;
    pnorm: number;
    beta: number;
    mask_type: MaskType;
    loss: LossResult;
    metrics: OptimizationMetrics;
    pgd: PgdConfig | null;
};

export type SubcomponentMetadata = {
    subcomponent_idx: number;
    mean_ci: number;
};

export type ActivationContextsSummary = Record<string, SubcomponentMetadata[]>;

// Note: Token P/R/lift stats come from /token_stats endpoint (batch job), not here
export type SubcomponentActivationContexts = {
    subcomponent_idx: number;
    mean_ci: number;
    example_tokens: string[][];
    example_ci: number[][];
    example_component_acts: number[][];
};

export type CorrelatedSubcomponent = {
    component_key: string;
    score: number;
    count_i: number; // Subject (query component) firing count
    count_j: number; // Object (this component) firing count
    count_ij: number; // Co-occurrence count
    n_tokens: number; // Total tokens
};

export type SubcomponentCorrelationsResponse = {
    precision: CorrelatedSubcomponent[];
    recall: CorrelatedSubcomponent[];
    jaccard: CorrelatedSubcomponent[];
    pmi: CorrelatedSubcomponent[];
    bottom_pmi: CorrelatedSubcomponent[];
};

// Token P/R/lift/PMI for a single category (input or output)
export type TokenPRLiftPMI = {
    top_recall: [string, number][]; // [(token, value), ...] sorted desc
    top_precision: [string, number][]; // [(token, value), ...] sorted desc
    top_lift: [string, number][]; // [(token, lift), ...] sorted desc
    top_pmi: [string, number][]; // [(token, pmi), ...] highest positive association
    bottom_pmi: [string, number][]; // [(token, pmi), ...] highest negative association
};

// Token stats from batch job - includes both input and output stats
export type TokenStatsResponse = {
    input: TokenPRLiftPMI; // What tokens activate this component
    output: TokenPRLiftPMI; // What tokens this component predicts
};

export type TokenizeResponse = {
    token_ids: number[];
    tokens: string[];
    text: string;
    next_token_probs: (number | null)[]; // Probability of next token (last is null)
};

export type TokenSearchResult = {
    id: number;
    string: string;
    prob: number;
};

/** Select active edge set based on variant preference. Falls back to signed if abs unavailable. */
export function getActiveEdges(
    data: GraphData,
    variant: "signed" | "abs_target",
): { edges: EdgeData[]; bySource: Map<string, EdgeData[]>; byTarget: Map<string, EdgeData[]>; maxAbsAttr: number } {
    if (variant === "abs_target" && data.edgesAbs) {
        return {
            edges: data.edgesAbs,
            bySource: data.edgesAbsBySource!,
            byTarget: data.edgesAbsByTarget!,
            maxAbsAttr: data.maxAbsAttrAbs || 1,
        };
    }
    return {
        edges: data.edges,
        bySource: data.edgesBySource,
        byTarget: data.edgesByTarget,
        maxAbsAttr: data.maxAbsAttr || 1,
    };
}

// Client-side computed types

export type NodePosition = {
    x: number;
    y: number;
};

export type PinnedNode = {
    layer: string;
    seqIdx: number;
    cIdx: number;
};

export type HoveredNode = {
    layer: string;
    seqIdx: number;
    cIdx: number;
};

export type HoveredEdge = {
    src: string;
    tgt: string;
    val: number;
};

// Graph layout result
export type LayoutResult = {
    nodePositions: Record<string, NodePosition>;
    layerYPositions: Record<string, number>;
    seqWidths: number[];
    seqXStarts: number[];
    width: number;
    height: number;
};

// Component probe result
export type SubcomponentProbeResult = {
    tokens: string[];
    ci_values: number[];
    subcomp_acts: number[];
    next_token_probs: (number | null)[]; // Probability of next token (last is null)
};

/** Get display name for a layer (e.g., "lm_head" -> "W_U") using model-provided names */
export function getLayerDisplayName(layer: string, displayNames: Record<string, string>): string {
    return displayNames[layer] ?? layer;
}

/** Format a node key for display, replacing layer names with display names */
export function formatNodeKeyForDisplay(nodeKey: string, displayNames: Record<string, string>): string {
    const [layer, ...rest] = nodeKey.split(":");
    const displayName = getLayerDisplayName(layer, displayNames);
    return [displayName, ...rest].join(":");
}

// Node intervention helpers
// "embed" and "output" are pseudo-layers used for visualization but are not part of the
// decomposed model. They cannot be intervened on - only the internal layers (attn/mlp)
// can have their components selectively activated.
const NON_INTERVENTABLE_LAYERS = new Set(["embed", "wte", "output"]);

export function isInterventableNode(nodeKey: string): boolean {
    const layer = nodeKey.split(":")[0];
    return !NON_INTERVENTABLE_LAYERS.has(layer);
}

export function filterInterventableNodes(nodeKeys: Iterable<string>): Set<string> {
    return new Set([...nodeKeys].filter(isInterventableNode));
}

/**
 * Convert a node key (layer:seq:cIdx) to a component key (layer:cIdx).
 * Component keys are used for caching/fetching component data.
 */
export function nodeKeyToComponentKey(nodeKey: string): string {
    const [layer, , cIdx] = nodeKey.split(":");
    return `${layer}:${cIdx}`;
}

/**
 * Extract unique component keys from a graph.
 * Filters out non-interventable nodes (wte, output) and returns unique layer:cIdx keys.
 */
export function extractComponentKeys(graph: GraphData): string[] {
    const componentKeys = new Set<string>();

    for (const nodeKey of Object.keys(graph.nodeCiVals)) {
        if (isInterventableNode(nodeKey)) {
            componentKeys.add(nodeKeyToComponentKey(nodeKey));
        }
    }

    return Array.from(componentKeys);
}
