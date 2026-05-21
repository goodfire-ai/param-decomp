import type { Loadable } from "../../lib";
import type { GraphData, CISnapshot } from "../../lib/promptAttributionsTypes";
import type { InterventionRunSummary } from "../../lib/interventionTypes";
import type { NormalizeType } from "../../lib/api";

export type MaskType = "stochastic" | "ci";
export type LossType = "ce" | "kl" | "logit";

export type ViewSettings = {
    topK: number;
    componentGap: number;
    layerGap: number;
    normalizeEdges: NormalizeType;
    ciThreshold: number;
};

/** Persisted graph data from the database */
export type StoredGraph = {
    id: number; // database ID
    label: string;
    data: GraphData;
    viewSettings: ViewSettings;
    interventionRuns: InterventionRunSummary[];
};

export type PromptCard = {
    id: number; // database prompt ID
    tokens: string[];
    tokenIds: number[];
    nextTokenProbs: (number | null)[]; // probability of each token given previous
    isCustom: boolean;
    graphs: StoredGraph[];
    activeGraphId: number | null; // null means "new graph" mode when graphs exist, or initial state
    activeView: "graph" | "interventions";
    // Config for creating new graphs (per-card, not shared globally)
    newGraphConfig: OptimizeConfigDraft;
    useOptimized: boolean; // whether to compute optimized graph
};

// Draft types for UI state (may be incomplete)
export type CELossConfigDraft = {
    type: "ce";
    coeff: number;
    position: number;
    labelTokenId: number | null; // null = not set yet
    labelTokenText: string; // user input text (may not match a token yet)
};

export type KLLossConfig = {
    type: "kl";
    coeff: number;
    position: number;
};

export type LogitLossConfigDraft = {
    type: "logit";
    coeff: number;
    position: number;
    labelTokenId: number | null;
    labelTokenText: string;
};

export type LossConfigDraft = CELossConfigDraft | KLLossConfig | LogitLossConfigDraft;

export type OptimizeConfigDraft = {
    loss: LossConfigDraft;
    impMinCoeff: number;
    steps: number;
    pnorm: number;
    beta: number;
    maskType: MaskType;
    advPgdNSteps: number | null;
    advPgdStepSize: number | null;
};

// Validated types for API calls (all required fields present)
export type CELossConfigValid = {
    type: "ce";
    coeff: number;
    position: number;
    labelTokenId: number;
    labelTokenText: string;
};

export type LogitLossConfigValid = {
    type: "logit";
    coeff: number;
    position: number;
    labelTokenId: number;
    labelTokenText: string;
};

export type LossConfigValid = CELossConfigValid | KLLossConfig | LogitLossConfigValid;

export type OptimizeConfigValid = {
    loss: LossConfigValid;
    impMinCoeff: number;
    steps: number;
    pnorm: number;
    beta: number;
    maskType: MaskType;
    advPgdNSteps: number | null;
    advPgdStepSize: number | null;
};

/** Validate draft config, returning valid config or null if incomplete */
export function validateOptimizeConfig(draft: OptimizeConfigDraft): OptimizeConfigValid | null {
    if ((draft.loss.type === "ce" || draft.loss.type === "logit") && draft.loss.labelTokenId === null) {
        return null;
    }
    return draft as OptimizeConfigValid;
}

/** Check if config is ready for submission */
export function isOptimizeConfigValid(draft: OptimizeConfigDraft): draft is OptimizeConfigValid {
    return validateOptimizeConfig(draft) !== null;
}

export type ComputeOptions = {
    ciThreshold: number;
    useOptimized: boolean;
    optimizeConfig: OptimizeConfigValid;
};

export type LoadingStage = {
    name: string;
    progress: number | null; // 0-1, or null for indeterminate
};

export type LoadingState = {
    stages: LoadingStage[];
    currentStage: number; // 0-indexed
};

/** Generic state for async actions without a meaningful result */
export type ActionState = { status: "idle" } | { status: "loading" } | { status: "error"; error: string };

/** Result from tokenize endpoint */
export type TokenizeResult = {
    tokens: string[];
    next_token_probs: (number | null)[];
};

/** State for the draft prompt input */
export type DraftState = {
    text: string;
    tokenPreview: Loadable<TokenizeResult>;
    isAdding: boolean;
};

export function defaultDraftState(): DraftState {
    return {
        text: "",
        tokenPreview: { status: "uninitialized" },
        isAdding: false,
    };
}

/** Discriminated union for the tab view - makes invalid states unrepresentable */
export type TabViewState =
    | { view: "draft"; draft: DraftState }
    | { view: "loading" }
    | { view: "card"; cardId: number }
    | { view: "error"; error: string };

/** State for graph computation - tracks which card is computing, progress, and errors */
export type GraphComputeState =
    | { status: "idle" }
    | { status: "computing"; cardId: number; progress: LoadingState; ciSnapshot: CISnapshot | null }
    | { status: "error"; error: string };

/** State for prompt generation - tracks progress and count */
export type PromptGenerateState =
    | { status: "idle" }
    | { status: "generating"; progress: number; count: number }
    | { status: "error"; error: string };

export function defaultOptimizeConfig(numTokens: number): OptimizeConfigDraft {
    return {
        loss: {
            type: "ce",
            coeff: 1,
            position: numTokens - 1,
            labelTokenId: null,
            labelTokenText: "",
        },
        impMinCoeff: 0.001,
        steps: 2000,
        pnorm: 0.3,
        beta: 0,
        maskType: "stochastic",
        advPgdNSteps: null,
        advPgdStepSize: null,
    };
}
