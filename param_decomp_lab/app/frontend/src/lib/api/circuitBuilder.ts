/**
 * API client for /api/circuit_builder endpoints (circuit-builder tab).
 */

export interface SiteInfo {
    site: string;
    C: number;
    d_in: number;
    d_out: number;
    rank: number;
}

export interface SubcomponentInfo {
    site: string;
    idx: number;
    label: string | null;
    u_norm_absorbed: number;
    examples: { tokens: string[]; active_position: number; activation: number }[];
}

export interface WriteTerm {
    site: string;
    idx: number;
    weight: number | null; // null -> default: raw ||j||
}

export interface LoraSpec {
    name: string;
    read_site: string;
    read_idx: number;
    writes: WriteTerm[];
    scale: number;
    n_prompts: number;
    enabled: boolean;
}

export interface JVectorInfo {
    site: string;
    idx: number;
    raw_norm: number;
}

export interface TokenLogit {
    token: string;
    token_id: number;
    logit: number;
    prob: number;
}

export interface PositionComparison {
    position: number;
    token: string;
    kl_base_to_edited: number;
    top_base: TokenLogit[];
    top_edited: TokenLogit[];
}

export interface GeneratedToken {
    token: string;
    token_id: number;
    top: TokenLogit[];
}

export interface GenerationResult {
    greedy: string;
    sampled: string;
    greedy_tokens: GeneratedToken[];
    sampled_tokens: GeneratedToken[];
}

export interface CompareResult {
    prompt_tokens: string[];
    positions: PositionComparison[];
    base: GenerationResult;
    edited: GenerationResult;
    mean_kl: number;
}

async function unwrap<T>(response: Response, what: string): Promise<T> {
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Failed to ${what}`);
    }
    return (await response.json()) as T;
}

export async function loadCircuitBuilder(
    source: string = "mock",
    runRef: string | null = null,
    seed: number = 0,
): Promise<{ run_id: string }> {
    const response = await fetch("/api/circuit_builder/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, run_ref: runRef, seed }),
    });
    return unwrap(response, "load circuit builder");
}

export async function getSites(): Promise<SiteInfo[]> {
    return unwrap(await fetch("/api/circuit_builder/sites"), "get sites");
}

export async function getDownstream(readSite: string): Promise<string[]> {
    return unwrap(await fetch(`/api/circuit_builder/downstream/${readSite}`), "get downstream sites");
}

export async function getSubcomponents(
    site: string,
    offset: number,
    limit: number,
    examples: number,
): Promise<SubcomponentInfo[]> {
    const params = new URLSearchParams({
        offset: String(offset),
        limit: String(limit),
        examples: String(examples),
    });
    return unwrap(await fetch(`/api/circuit_builder/subcomponents/${site}?${params}`), "get subcomponents");
}

export async function computeJVectors(
    readSite: string,
    targets: { site: string; idx: number }[],
    nPrompts: number,
): Promise<JVectorInfo[]> {
    const response = await fetch("/api/circuit_builder/j_vectors", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ read_site: readSite, targets, n_prompts: nPrompts }),
    });
    return unwrap(response, "compute j-vectors");
}

export async function listLoras(): Promise<LoraSpec[]> {
    return unwrap(await fetch("/api/circuit_builder/loras"), "list LoRAs");
}

export async function putLora(spec: LoraSpec): Promise<LoraSpec> {
    const response = await fetch(`/api/circuit_builder/loras/${encodeURIComponent(spec.name)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(spec),
    });
    return unwrap(response, "save LoRA");
}

export async function deleteLora(name: string): Promise<void> {
    const response = await fetch(`/api/circuit_builder/loras/${encodeURIComponent(name)}`, {
        method: "DELETE",
    });
    await unwrap(response, "delete LoRA");
}

export async function runCompare(request: {
    prompt: string;
    top_k: number;
    max_new_tokens: number;
    temperature: number;
    seed: number;
}): Promise<CompareResult> {
    const response = await fetch("/api/circuit_builder/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
    });
    return unwrap(response, "run comparison");
}
