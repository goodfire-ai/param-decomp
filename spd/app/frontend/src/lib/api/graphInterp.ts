/**
 * API client for /api/graph_interp endpoints.
 */

import { fetchJson } from "./index";

export type GraphInterpHeadline = {
    label: string;
    confidence: string;
    output_label: string | null;
    input_label: string | null;
};

export type LabelDetail = {
    label: string;
    confidence: string;
    reasoning: string;
    prompt: string;
};

export type GraphInterpDetail = {
    output: LabelDetail | null;
    input: LabelDetail | null;
    unified: LabelDetail | null;
};

export type PromptEdgeResponse = {
    related_key: string;
    pass_name: string;
    attribution: number;
    related_label: string | null;
    related_confidence: string | null;
    token_str: string | null;
};

export type GraphInterpComponentDetail = {
    output: LabelDetail | null;
    input: LabelDetail | null;
    unified: LabelDetail | null;
    edges: PromptEdgeResponse[];
};

export type GraphNode = {
    component_key: string;
    label: string;
    confidence: string;
};

export type GraphEdge = {
    source: string;
    target: string;
    attribution: number;
    pass_name: string;
};

export type ModelGraphResponse = {
    nodes: GraphNode[];
    edges: GraphEdge[];
};

export type GraphInterpInfo = {
    subrun_id: string;
    config: Record<string, unknown> | null;
    label_counts: Record<string, number>;
};

export async function getAllGraphInterpLabels(): Promise<Record<string, GraphInterpHeadline>> {
    return fetchJson<Record<string, GraphInterpHeadline>>("/api/graph_interp/labels");
}

export async function getGraphInterpDetail(layer: string, cIdx: number): Promise<GraphInterpDetail> {
    return fetchJson<GraphInterpDetail>(`/api/graph_interp/labels/${layer}/${cIdx}`);
}

export async function getGraphInterpComponentDetail(layer: string, cIdx: number): Promise<GraphInterpComponentDetail> {
    return fetchJson<GraphInterpComponentDetail>(`/api/graph_interp/detail/${layer}/${cIdx}`);
}

export async function getModelGraph(): Promise<ModelGraphResponse> {
    return fetchJson<ModelGraphResponse>("/api/graph_interp/graph");
}
