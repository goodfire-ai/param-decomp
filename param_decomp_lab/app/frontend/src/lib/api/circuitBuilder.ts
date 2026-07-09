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

export interface ActivationExample {
    tokens: string[];
    active_position: number;
    activation: number;
}

export interface SubcomponentInfo {
    site: string;
    idx: number;
    label: string | null;
    label_source: string | null;
    u_norm_absorbed: number;
    examples: ActivationExample[];
}

export interface SearchHit {
    site: string;
    idx: number;
    label: string;
    label_source: string;
}

export interface ComponentDetail {
    site: string;
    idx: number;
    label: string | null;
    label_source: string | null;
    reasoning: string | null;
    u_norm_absorbed: number;
    examples: ActivationExample[];
}

export interface WriteTerm {
    site: string;
    idx: number; // component idx, or token id for kind "logit"
    weight: number | null; // null -> kind-specific default (||j|| or ||U||*||V||)
    kind: "j" | "u" | "logit";
    label: string | null; // display only (token string for logit terms)
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

export interface TokenInfo {
    token_id: number;
    token: string;
}

export async function tokenizeText(text: string): Promise<TokenInfo[]> {
    const params = new URLSearchParams({ text });
    return unwrap(await fetch(`/api/circuit_builder/tokens?${params}`), "tokenize text");
}

export async function searchLabels(
    q: string,
    limit: number,
    downstreamOf: string | null = null,
): Promise<SearchHit[]> {
    const params = new URLSearchParams({ q, limit: String(limit) });
    if (downstreamOf) params.set("downstream_of", downstreamOf);
    return unwrap(await fetch(`/api/circuit_builder/search?${params}`), "search labels");
}

export async function getComponentDetail(
    site: string,
    idx: number,
    examples: number = 8,
): Promise<ComponentDetail> {
    return unwrap(
        await fetch(`/api/circuit_builder/component/${site}/${idx}?examples=${examples}`),
        "get component detail",
    );
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
