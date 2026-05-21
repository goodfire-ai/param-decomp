/**
 * API client for /api/intervention endpoints.
 */

import type { InterventionRunSummary, RunInterventionRequest } from "../interventionTypes";

export async function runAndSaveIntervention(request: RunInterventionRequest): Promise<InterventionRunSummary> {
    const response = await fetch("/api/intervention/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to run intervention");
    }
    return (await response.json()) as InterventionRunSummary;
}

export async function getInterventionRuns(graphId: number): Promise<InterventionRunSummary[]> {
    const response = await fetch(`/api/intervention/runs/${graphId}`);
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to get intervention runs");
    }
    return (await response.json()) as InterventionRunSummary[];
}

export async function deleteInterventionRun(runId: number): Promise<void> {
    const response = await fetch(`/api/intervention/runs/${runId}`, {
        method: "DELETE",
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to delete intervention run");
    }
}
