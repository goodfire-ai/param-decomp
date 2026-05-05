/**
 * API client for /api/prompts endpoints.
 */

import type { PromptPreview } from "../promptAttributionsTypes";
import { apiUrl, fetchJson } from "./index";

export async function listPrompts(): Promise<PromptPreview[]> {
    return fetchJson<PromptPreview[]>("/api/prompts");
}

export async function createCustomPrompt(text: string): Promise<PromptPreview> {
    const url = apiUrl("/api/prompts/custom");
    url.searchParams.set("text", text);
    return fetchJson<PromptPreview>(url.toString(), { method: "POST" });
}

export async function deletePrompt(promptId: number): Promise<void> {
    await fetchJson<void>(`/api/prompts/${promptId}`, { method: "DELETE" });
}
