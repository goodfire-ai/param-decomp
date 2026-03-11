/**
 * API client for /api/graphs endpoints.
 */

import type { GraphData, EdgeData, TokenizeResponse, TokenSearchResult, CISnapshot } from "../promptAttributionsTypes";
import { buildEdgeIndexes } from "../promptAttributionsTypes";
import { apiUrl, ApiError, fetchJson } from "./index";

/** Hydrate a raw API graph response into a full GraphData with edge indexes. */
function hydrateGraph(raw: Record<string, unknown>): GraphData {
    const g = raw as Omit<GraphData, "edgesBySource" | "edgesByTarget" | "edgesAbsBySource" | "edgesAbsByTarget">;
    const { edgesBySource, edgesByTarget } = buildEdgeIndexes(g.edges);
    const edgesAbs = (g.edgesAbs satisfies EdgeData[] | null) ?? null;
    let edgesAbsBySource: Map<string, EdgeData[]> | null = null;
    let edgesAbsByTarget: Map<string, EdgeData[]> | null = null;
    if (edgesAbs) {
        const absIndexes = buildEdgeIndexes(edgesAbs);
        edgesAbsBySource = absIndexes.edgesBySource;
        edgesAbsByTarget = absIndexes.edgesByTarget;
    }
    return { ...g, edgesBySource, edgesByTarget, edgesAbs, edgesAbsBySource, edgesAbsByTarget };
}

export type NormalizeType = "none" | "target" | "layer";

export type GraphProgress = {
    current: number;
    total: number;
    stage: string;
};

export type ComputeGraphParams = {
    promptId: number;
    normalize: NormalizeType;
    ciThreshold: number;
    /** If provided, only include these nodes in the graph (creates manual graph) */
    includedNodes?: string[];
};

/** Generic SSE stream parser. Delegates result extraction to the caller via extractResult. */
async function parseSSEStream<T>(
    response: Response,
    extractResult: (data: Record<string, unknown>) => T,
    onProgress?: (progress: GraphProgress) => void,
    onCISnapshot?: (snapshot: CISnapshot) => void,
): Promise<T> {
    const reader = response.body?.getReader();
    if (!reader) {
        throw new Error("Response body is not readable");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let result: T | null = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
            if (!line.trim() || !line.startsWith("data: ")) continue;

            const data = JSON.parse(line.substring(6));

            if (data.type === "progress" && onProgress) {
                onProgress({ current: data.current, total: data.total, stage: data.stage });
            } else if (data.type === "ci_snapshot" && onCISnapshot) {
                onCISnapshot(data as CISnapshot);
            } else if (data.type === "error") {
                throw new ApiError(data.error, 500);
            } else if (data.type === "complete") {
                result = extractResult(data);
                await reader.cancel();
                break;
            }
        }

        if (result) break;
    }

    if (!result) {
        throw new Error("No result received from stream");
    }

    return result;
}

export async function computeGraphStream(
    params: ComputeGraphParams,
    onProgress?: (progress: GraphProgress) => void,
): Promise<GraphData> {
    const url = apiUrl("/api/graphs");
    url.searchParams.set("prompt_id", String(params.promptId));
    url.searchParams.set("normalize", String(params.normalize));
    url.searchParams.set("ci_threshold", String(params.ciThreshold));
    if (params.includedNodes !== undefined) {
        url.searchParams.set("included_nodes", JSON.stringify(params.includedNodes));
    }

    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new ApiError(error.detail || `HTTP ${response.status}`, response.status);
    }

    return parseSSEStream(response, (data) => hydrateGraph(data.data as Record<string, unknown>), onProgress);
}

export type MaskType = "stochastic" | "ci";
export type LossType = "ce" | "kl" | "logit";

export type ComputeGraphOptimizedParams = {
    promptId: number;
    impMinCoeff: number;
    steps: number;
    pnorm: number;
    beta: number;
    normalize: NormalizeType;
    ciThreshold: number;
    maskType: MaskType;
    lossType: LossType;
    lossCoeff: number;
    lossPosition: number;
    labelToken?: number; // Required for CE loss
    advPgdNSteps?: number;
    advPgdStepSize?: number;
};

export async function computeGraphOptimizedStream(
    params: ComputeGraphOptimizedParams,
    onProgress?: (progress: GraphProgress) => void,
    onCISnapshot?: (snapshot: CISnapshot) => void,
): Promise<GraphData> {
    const url = apiUrl("/api/graphs/optimized/stream");
    url.searchParams.set("prompt_id", String(params.promptId));
    url.searchParams.set("imp_min_coeff", String(params.impMinCoeff));
    url.searchParams.set("steps", String(params.steps));
    url.searchParams.set("pnorm", String(params.pnorm));
    url.searchParams.set("beta", String(params.beta));
    url.searchParams.set("normalize", String(params.normalize));
    url.searchParams.set("ci_threshold", String(params.ciThreshold));
    url.searchParams.set("mask_type", params.maskType);
    url.searchParams.set("loss_type", params.lossType);
    url.searchParams.set("loss_coeff", String(params.lossCoeff));
    url.searchParams.set("loss_position", String(params.lossPosition));
    if (params.labelToken !== undefined) {
        url.searchParams.set("label_token", String(params.labelToken));
    }
    if (params.advPgdNSteps !== undefined) {
        url.searchParams.set("adv_pgd_n_steps", String(params.advPgdNSteps));
    }
    if (params.advPgdStepSize !== undefined) {
        url.searchParams.set("adv_pgd_step_size", String(params.advPgdStepSize));
    }

    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new ApiError(error.detail || `HTTP ${response.status}`, response.status);
    }

    return parseSSEStream(
        response,
        (data) => hydrateGraph(data.data as Record<string, unknown>),
        onProgress,
        onCISnapshot,
    );
}

export type ComputeGraphOptimizedBatchParams = {
    promptId: number;
    impMinCoeffs: number[];
    steps: number;
    pnorm: number;
    beta: number;
    normalize: NormalizeType;
    ciThreshold: number;
    maskType: MaskType;
    lossType: LossType;
    lossCoeff: number;
    lossPosition: number;
    labelToken?: number;
    advPgdNSteps?: number;
    advPgdStepSize?: number;
};

export async function computeGraphOptimizedBatchStream(
    params: ComputeGraphOptimizedBatchParams,
    onProgress?: (progress: GraphProgress) => void,
    onCISnapshot?: (snapshot: CISnapshot) => void,
): Promise<GraphData[]> {
    const url = apiUrl("/api/graphs/optimized/batch/stream");

    const body: Record<string, unknown> = {
        prompt_id: params.promptId,
        imp_min_coeffs: params.impMinCoeffs,
        steps: params.steps,
        pnorm: params.pnorm,
        beta: params.beta,
        normalize: params.normalize,
        ci_threshold: params.ciThreshold,
        mask_type: params.maskType,
        loss_type: params.lossType,
        loss_coeff: params.lossCoeff,
        loss_position: params.lossPosition,
    };
    if (params.labelToken !== undefined) body.label_token = params.labelToken;
    if (params.advPgdNSteps !== undefined) body.adv_pgd_n_steps = params.advPgdNSteps;
    if (params.advPgdStepSize !== undefined) body.adv_pgd_step_size = params.advPgdStepSize;

    const response = await fetch(url.toString(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new ApiError(error.detail || `HTTP ${response.status}`, response.status);
    }

    return parseSSEStream(
        response,
        (data) => (data.data as { graphs: Record<string, unknown>[] }).graphs.map((g) => hydrateGraph(g)),
        onProgress,
        onCISnapshot,
    );
}

export async function getGraphs(promptId: number, normalize: NormalizeType, ciThreshold: number): Promise<GraphData[]> {
    const url = apiUrl(`/api/graphs/${promptId}`);
    url.searchParams.set("normalize", normalize);
    url.searchParams.set("ci_threshold", String(ciThreshold));
    const graphs = await fetchJson<Record<string, unknown>[]>(url.toString());
    return graphs.map((g) => hydrateGraph(g));
}

export async function tokenizeText(text: string): Promise<TokenizeResponse> {
    const url = apiUrl("/api/graphs/tokenize");
    url.searchParams.set("text", text);
    return fetchJson<TokenizeResponse>(url.toString(), { method: "POST" });
}

export async function searchTokens(
    query: string,
    promptId: number,
    position: number,
    limit: number = 20,
): Promise<TokenSearchResult[]> {
    const url = apiUrl("/api/graphs/tokens/search");
    url.searchParams.set("q", query);
    url.searchParams.set("limit", String(limit));
    url.searchParams.set("prompt_id", String(promptId));
    url.searchParams.set("position", String(position));
    const response = await fetchJson<{ tokens: TokenSearchResult[] }>(url.toString());
    return response.tokens;
}
