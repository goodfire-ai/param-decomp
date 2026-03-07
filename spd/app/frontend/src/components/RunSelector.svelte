<script lang="ts">
    import { onMount } from "svelte";
    import { CANONICAL_RUNS, formatRunIdForDisplay } from "../lib/registry";
    import { fetchRunInfo, type RunInfoResponse, type DataAvailability } from "../lib/api/runRegistry";

    const AVAILABILITY_COLUMNS: { key: keyof DataAvailability; abbrev: string; tooltip: string }[] = [
        { key: "harvest", abbrev: "H", tooltip: "Harvest: activation stats, correlations, token associations" },
        { key: "autointerp", abbrev: "AI", tooltip: "Autointerp: LLM-generated component labels" },
        {
            key: "attributions",
            abbrev: "DA",
            tooltip: "Dataset Attributions: component-to-component attribution strengths",
        },
        {
            key: "graph_interp",
            abbrev: "GI",
            tooltip: "Graph Interp: context-aware labels using attribution graph structure",
        },
    ];

    type Props = {
        onSelect: (wandbPath: string, contextLength: number) => void;
        isLoading: boolean;
        username: string | null;
    };

    let { onSelect, isLoading, username }: Props = $props();

    let customPath = $state("");
    let contextLength = $state(512);

    let backendData = $state<Record<string, RunInfoResponse>>({});
    let backendLoaded = $state(false);

    onMount(() => {
        fetchRunInfo(CANONICAL_RUNS.map((r) => r.wandbRunId)).then(
            (runs) => {
                const map: Record<string, RunInfoResponse> = {};
                for (const run of runs) {
                    map[run.wandb_run_id] = run;
                }
                backendData = map;
                backendLoaded = true;
            },
            () => {
                backendLoaded = true;
            },
        );
    });

    function handleRowClick(wandbRunId: string) {
        onSelect(wandbRunId, contextLength);
    }

    function handleCustomSubmit(event: Event) {
        event.preventDefault();
        const path = customPath.trim();
        if (!path) return;
        onSelect(path, contextLength);
    }
</script>

<div class="selector-container">
    {#if isLoading}
        <div class="loading-overlay">
            <div class="spinner"></div>
            <p class="loading-text">Loading run...</p>
        </div>
    {/if}
    <div class="selector-content" class:dimmed={isLoading}>
        <h1 class="title">
            {#if username}
                Hello, {username}
            {:else}
                SPD Explorer
            {/if}
        </h1>

        <div class="table-wrapper">
            <table class="runs-table">
                <thead>
                    <tr>
                        <th>Run</th>
                        <th>Architecture</th>
                        <th>Notes</th>
                        {#each AVAILABILITY_COLUMNS as col (col.key)}
                            <th class="avail-col tooltip-wrap"
                                >{col.abbrev}<span class="tooltip">{col.tooltip}</span></th
                            >
                        {/each}
                    </tr>
                </thead>
                <tbody>
                    {#each CANONICAL_RUNS as entry (entry.wandbRunId)}
                        {@const info = backendData[entry.wandbRunId]}
                        <tr
                            class="run-row"
                            onclick={() => handleRowClick(entry.wandbRunId)}
                            role="button"
                            tabindex="0"
                            onkeydown={(e) => {
                                if (e.key === "Enter") handleRowClick(entry.wandbRunId);
                            }}
                        >
                            <td class="cell-run">
                                {#if entry.name}
                                    <span class="run-name">{entry.name}</span>
                                {/if}
                                <span class="run-id">{formatRunIdForDisplay(entry.wandbRunId)}</span>
                            </td>
                            <td class="cell-arch">
                                {#if info?.architecture}
                                    {info.architecture}
                                {:else if !backendLoaded}
                                    <span class="skeleton"></span>
                                {:else}
                                    <span class="muted">-</span>
                                {/if}
                            </td>
                            <td class="cell-notes">
                                {#if entry.notes}
                                    {entry.notes}
                                {/if}
                            </td>
                            {#each AVAILABILITY_COLUMNS as col (col.key)}
                                <td class="cell-avail">
                                    {#if info}
                                        <span
                                            class="dot"
                                            class:available={info.availability[col.key]}
                                            title={col.tooltip}
                                        ></span>
                                    {:else if !backendLoaded}
                                        <span class="skeleton skeleton-dot"></span>
                                    {:else}
                                        <span class="dot"></span>
                                    {/if}
                                </td>
                            {/each}
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>

        <div class="divider">
            <span>or enter a custom path</span>
        </div>

        <form class="custom-form" onsubmit={handleCustomSubmit}>
            <input
                type="text"
                placeholder="e.g. s-17805b61 or goodfire/spd/runs/33n6xjjt"
                bind:value={customPath}
                disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !customPath.trim()}>
                {isLoading ? "Loading..." : "Load"}
            </button>
        </form>
    </div>
</div>

<style>
    .selector-container {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        background: var(--bg-base);
        padding: var(--space-4);
    }

    .loading-overlay {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: var(--space-3);
        background: rgba(0, 0, 0, 0.5);
        z-index: 100;
    }

    .spinner {
        width: 40px;
        height: 40px;
        border: 3px solid var(--border-default);
        border-top-color: var(--accent-primary);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }

    .loading-text {
        color: var(--text-primary);
        font-family: var(--font-sans);
        font-size: var(--text-sm);
        margin: 0;
    }

    .selector-content {
        max-width: 860px;
        width: 100%;
        transition: opacity var(--transition-slow);
    }

    .selector-content.dimmed {
        opacity: 0.3;
        pointer-events: none;
    }

    .title {
        font-size: var(--text-3xl);
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 var(--space-4) 0;
        text-align: center;
        font-family: var(--font-sans);
    }

    .table-wrapper {
        margin-bottom: var(--space-6);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    .runs-table {
        width: 100%;
        border-collapse: collapse;
        font-family: var(--font-sans);
        font-size: var(--text-sm);
    }

    .runs-table thead {
        background: var(--bg-surface);
    }

    .runs-table th {
        padding: var(--space-2) var(--space-3);
        text-align: left;
        font-weight: 500;
        color: var(--text-muted);
        font-size: var(--text-xs);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid var(--border-default);
    }

    .avail-col {
        width: 36px;
        text-align: center !important;
    }

    .tooltip-wrap {
        position: relative;
        cursor: help;
    }

    .tooltip {
        display: none;
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        margin-top: 4px;
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        font-size: var(--text-xs);
        font-weight: 400;
        text-transform: none;
        letter-spacing: normal;
        color: var(--text-secondary);
        white-space: nowrap;
        z-index: 10;
        pointer-events: none;
    }

    .tooltip-wrap:hover .tooltip {
        display: block;
    }

    .runs-table td {
        padding: var(--space-2) var(--space-3);
        border-bottom: 1px solid var(--border-default);
    }

    .runs-table tbody tr:last-child td {
        border-bottom: none;
    }

    .run-row {
        cursor: pointer;
        transition: background var(--transition-normal);
    }

    .run-row:hover {
        background: var(--bg-elevated);
    }

    .cell-run {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .run-name {
        font-weight: 600;
        color: var(--text-primary);
    }

    .run-id {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--accent-primary);
    }

    .cell-arch {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-secondary);
    }

    .cell-notes {
        color: var(--text-muted);
        font-size: var(--text-xs);
    }

    .cell-avail {
        text-align: center;
    }

    .muted {
        color: var(--text-muted);
    }

    .dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--border-default);
    }

    .dot.available {
        background: var(--status-success, #22c55e);
    }

    .skeleton {
        display: inline-block;
        height: 12px;
        width: 80px;
        border-radius: var(--radius-sm);
        background: var(--border-default);
        opacity: 0.4;
        animation: pulse 1.2s ease-in-out infinite;
    }

    .skeleton-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }

    @keyframes pulse {
        0%,
        100% {
            opacity: 0.4;
        }
        50% {
            opacity: 0.15;
        }
    }

    .divider {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        margin-bottom: var(--space-4);
    }

    .divider::before,
    .divider::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border-default);
    }

    .divider span {
        font-size: var(--text-sm);
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .custom-form {
        display: flex;
        gap: var(--space-2);
    }

    .custom-form input[type="text"] {
        flex: 1;
        padding: var(--space-2) var(--space-3);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .custom-form input[type="text"]::placeholder {
        color: var(--text-muted);
    }

    .custom-form input[type="text"]:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .custom-form button {
        padding: var(--space-2) var(--space-4);
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: var(--radius-sm);
        font-weight: 500;
        cursor: pointer;
        font-family: var(--font-sans);
    }

    .custom-form button:hover:not(:disabled) {
        opacity: 0.9;
    }

    .custom-form button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
        cursor: not-allowed;
    }
</style>
