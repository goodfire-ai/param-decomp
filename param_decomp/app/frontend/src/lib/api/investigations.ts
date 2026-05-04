/**
 * API client for investigation results.
 */

export interface InvestigationSummary {
    id: string; // inv_id (e.g., "inv-abc12345")
    wandb_path: string | null;
    prompt: string | null;
    created_at: string;
    has_research_log: boolean;
    has_explanations: boolean;
    event_count: number;
    last_event_time: string | null;
    last_event_message: string | null;
    // Agent-provided summary
    title: string | null;
    summary: string | null;
    status: string | null; // in_progress, completed, inconclusive
}

export interface EventEntry {
    event_type: string;
    timestamp: string;
    message: string;
    details: Record<string, unknown> | null;
}

export interface InvestigationDetail {
    id: string;
    wandb_path: string | null;
    prompt: string | null;
    created_at: string;
    research_log: string | null;
    events: EventEntry[];
    explanations: Record<string, unknown>[];
    artifact_ids: string[]; // List of artifact IDs available for this investigation
    // Agent-provided summary
    title: string | null;
    summary: string | null;
    status: string | null;
}

import type { EdgeData, OutputProbability } from "../promptAttributionsTypes";

/** Data for a graph artifact (subset of GraphData, self-contained for offline viewing) */
export interface ArtifactGraphData {
    tokens: string[];
    edges: EdgeData[];
    outputProbs: Record<string, OutputProbability>;
    nodeCiVals: Record<string, number>;
    nodeSubcompActs: Record<string, number>;
    maxAbsAttr: number;
    l0_total: number;
}

export interface GraphArtifact {
    type: "graph";
    id: string;
    caption: string | null;
    graph_id: number;
    data: ArtifactGraphData;
}

export interface LaunchResponse {
    inv_id: string;
    job_id: string;
}

export async function launchInvestigation(prompt: string): Promise<LaunchResponse> {
    const res = await fetch("/api/investigations/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
    });
    if (!res.ok) throw new Error(`Failed to launch investigation: ${res.statusText}`);
    return res.json();
}

export async function listInvestigations(): Promise<InvestigationSummary[]> {
    const res = await fetch("/api/investigations");
    if (!res.ok) throw new Error(`Failed to list investigations: ${res.statusText}`);
    return res.json();
}

export async function getInvestigation(invId: string): Promise<InvestigationDetail> {
    const res = await fetch(`/api/investigations/${invId}`);
    if (!res.ok) throw new Error(`Failed to get investigation: ${res.statusText}`);
    return res.json();
}

export async function listArtifacts(invId: string): Promise<string[]> {
    const res = await fetch(`/api/investigations/${invId}/artifacts`);
    if (!res.ok) throw new Error(`Failed to list artifacts: ${res.statusText}`);
    return res.json();
}

export async function getArtifact(invId: string, artifactId: string): Promise<GraphArtifact> {
    const res = await fetch(`/api/investigations/${invId}/artifacts/${artifactId}`);
    if (!res.ok) throw new Error(`Failed to get artifact: ${res.statusText}`);
    return res.json();
}
