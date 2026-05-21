/**
 * API client for /api/dataset endpoints.
 */

import { apiUrl } from "./index";

export type DatasetSearchResult = {
    text: string;
    occurrence_count: number;
    metadata: Record<string, string>;
};

export type DatasetSearchMetadata = {
    query: string;
    split: string;
    dataset_name: string;
    total_results: number;
    search_time_seconds: number;
};

export type DatasetSearchPage = {
    results: DatasetSearchResult[];
    page: number;
    page_size: number;
    total_results: number;
    total_pages: number;
};

export async function searchDataset(query: string, split: string): Promise<DatasetSearchMetadata> {
    const url = apiUrl("/api/dataset/search");
    url.searchParams.set("query", query);
    url.searchParams.set("split", split);

    const response = await fetch(url.toString(), { method: "POST" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to search dataset");
    }

    return (await response.json()) as DatasetSearchMetadata;
}

export async function getDatasetResults(page: number, pageSize: number): Promise<DatasetSearchPage> {
    const url = apiUrl("/api/dataset/results");
    url.searchParams.set("page", String(page));
    url.searchParams.set("page_size", String(pageSize));

    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get search results");
    }

    return (await response.json()) as DatasetSearchPage;
}

export type TokenizedSearchResult = {
    tokens: string[];
    next_token_probs: (number | null)[];
    occurrence_count: number;
    metadata: Record<string, string>;
};

export type TokenizedSearchPage = {
    results: TokenizedSearchResult[];
    query: string;
    page: number;
    page_size: number;
    total_results: number;
    total_pages: number;
};

export async function getTokenizedResults(
    page: number,
    pageSize: number = 10,
    maxTokens: number = 256,
): Promise<TokenizedSearchPage> {
    const url = apiUrl("/api/dataset/results_tokenized");
    url.searchParams.set("page", String(page));
    url.searchParams.set("page_size", String(pageSize));
    url.searchParams.set("max_tokens", String(maxTokens));

    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get tokenized results");
    }

    return (await response.json()) as TokenizedSearchPage;
}

export type RandomSamplesResult = {
    results: DatasetSearchResult[];
    total_available: number;
    seed: number;
};

export async function getRandomSamples(
    nSamples: number = 100,
    seed: number = 42,
    split: "train" | "test" = "train",
): Promise<RandomSamplesResult> {
    const url = apiUrl("/api/dataset/random");
    url.searchParams.set("n_samples", String(nSamples));
    url.searchParams.set("seed", String(seed));
    url.searchParams.set("split", split);

    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get random samples");
    }

    return (await response.json()) as RandomSamplesResult;
}

export type TokenizedSample = {
    tokens: string[];
    next_token_probs: (number | null)[]; // Probability of next token; null for last position
    metadata: Record<string, string>;
};

export type RandomSamplesWithLossResult = {
    results: TokenizedSample[];
    total_available: number;
    seed: number;
};

export async function getRandomSamplesWithLoss(
    nSamples: number = 20,
    seed: number = 42,
    split: "train" | "test" = "train",
    maxTokens: number = 256,
): Promise<RandomSamplesWithLossResult> {
    const url = apiUrl("/api/dataset/random_with_loss");
    url.searchParams.set("n_samples", String(nSamples));
    url.searchParams.set("seed", String(seed));
    url.searchParams.set("split", split);
    url.searchParams.set("max_tokens", String(maxTokens));

    const response = await fetch(url.toString(), { method: "GET" });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get random samples with loss");
    }

    return (await response.json()) as RandomSamplesWithLossResult;
}
