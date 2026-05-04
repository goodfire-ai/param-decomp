/**
 * Shared API utilities and exports.
 *
 * In development, Vite proxies /api requests to the backend.
 * This allows the frontend to work regardless of which port the backend is on.
 */

/**
 * Build a URL for an API endpoint.
 * Uses relative paths which Vite's proxy forwards to the backend.
 */
export function apiUrl(path: string): URL {
    return new URL(path, window.location.origin);
}

export class ApiError extends Error {
    constructor(
        message: string,
        public status: number,
    ) {
        super(message);
        this.name = "ApiError";
    }
}

export async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(url, options);
    const text = await response.text();

    if (!response.ok) {
        let message = `HTTP ${response.status}`;
        try {
            const data = JSON.parse(text);
            message = data.detail || data.error || message;
        } catch {
            message = text.slice(0, 200) || message;
        }
        throw new ApiError(message, response.status);
    }

    return JSON.parse(text) as T;
}

// Re-export all API modules
export * from "./autointerpCompare";
export * from "./runs";
export * from "./graphs";
export * from "./prompts";
export * from "./activationContexts";
export * from "./correlations";
export * from "./datasetAttributions";
export * from "./intervention";
export * from "./dataset";
export * from "./clusters";
export * from "./investigations";
export * from "./dataSources";
export * from "./graphInterp";
export * from "./pretrainInfo";
export * from "./runRegistry";
