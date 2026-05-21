/**
 * API client for /api/pretrain_info endpoint.
 */

import { fetchJson } from "./index";

export type BlockStructure = {
    index: number;
    attn_type: "separate" | "fused";
    attn_projections: string[];
    ffn_type: "glu" | "mlp";
    ffn_projections: string[];
};

export type TopologyInfo = {
    n_blocks: number;
    block_structure: BlockStructure[];
};

export type PretrainInfoResponse = {
    model_type: string;
    summary: string;
    dataset_short: string | null;
    target_model_config: Record<string, unknown> | null;
    pretrain_config: Record<string, unknown> | null;
    pretrain_wandb_path: string | null;
    topology: TopologyInfo | null;
};

export async function fetchPretrainInfo(wandbPath: string): Promise<PretrainInfoResponse> {
    const params = new URLSearchParams({ wandb_path: wandbPath });
    return fetchJson<PretrainInfoResponse>(`/api/pretrain_info?${params}`);
}

export async function fetchPretrainInfoForLoadedRun(): Promise<PretrainInfoResponse> {
    return fetchJson<PretrainInfoResponse>("/api/pretrain_info/loaded");
}
