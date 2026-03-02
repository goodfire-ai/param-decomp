<script lang="ts">
    import { onMount } from "svelte";
    import { formatRunIdForDisplay } from "../lib/registry";
    import { fetchRunRegistry, type RegistryRunInfo } from "../lib/api/runRegistry";

    type Props = {
        onSelect: (wandbPath: string, contextLength: number) => void;
        isLoading: boolean;
        username: string | null;
    };

    let { onSelect, isLoading, username }: Props = $props();

    let customPath = $state("");
    let contextLength = $state(512);
    let registryRuns = $state<RegistryRunInfo[] | null>(null);
    let registryError = $state<string | null>(null);

    onMount(() => {
        fetchRunRegistry().then(
            (runs) => {
                registryRuns = runs;
            },
            (err) => {
                registryError = String(err);
            },
        );
    });

    function handleRowClick(entry: RegistryRunInfo) {
        onSelect(entry.wandb_run_id, contextLength);
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

        {#if registryError}
            <p class="error-text">Failed to load registry: {registryError}</p>
        {:else if registryRuns === null}
            <p class="loading-registry">Loading runs...</p>
        {:else}
            <div class="table-wrapper">
                <table class="runs-table">
                    <thead>
                        <tr>
                            <th>Run</th>
                            <th>Architecture</th>
                            <th>Notes</th>
                            <th class="avail-col" title="Harvest">H</th>
                            <th class="avail-col" title="Autointerp">AI</th>
                            <th class="avail-col" title="Dataset Attributions">DA</th>
                            <th class="avail-col" title="Graph Interp">GI</th>
                        </tr>
                    </thead>
                    <tbody>
                        {#each registryRuns as entry (entry.wandb_run_id)}
                            <tr
                                class="run-row"
                                onclick={() => handleRowClick(entry)}
                                role="button"
                                tabindex="0"
                                onkeydown={(e) => {
                                    if (e.key === "Enter") handleRowClick(entry);
                                }}
                            >
                                <td class="cell-run">
                                    {#if entry.name}
                                        <span class="run-name">{entry.name}</span>
                                    {/if}
                                    <span class="run-id">{formatRunIdForDisplay(entry.wandb_run_id)}</span>
                                </td>
                                <td class="cell-arch">
                                    {#if entry.architecture}
                                        {entry.architecture}
                                    {:else}
                                        <span class="muted">-</span>
                                    {/if}
                                </td>
                                <td class="cell-notes">
                                    {#if entry.notes}
                                        {entry.notes}
                                    {/if}
                                </td>
                                <td class="cell-avail">
                                    <span class="dot" class:available={entry.availability.harvest} title="Harvest"
                                    ></span>
                                </td>
                                <td class="cell-avail">
                                    <span class="dot" class:available={entry.availability.autointerp} title="Autointerp"
                                    ></span>
                                </td>
                                <td class="cell-avail">
                                    <span
                                        class="dot"
                                        class:available={entry.availability.attributions}
                                        title="Dataset Attributions"
                                    ></span>
                                </td>
                                <td class="cell-avail">
                                    <span
                                        class="dot"
                                        class:available={entry.availability.graph_interp}
                                        title="Graph Interp"
                                    ></span>
                                </td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
            </div>
        {/if}

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

    .error-text {
        color: var(--status-error, #ef4444);
        font-family: var(--font-sans);
        font-size: var(--text-sm);
        text-align: center;
    }

    .loading-registry {
        color: var(--text-muted);
        font-family: var(--font-sans);
        font-size: var(--text-sm);
        text-align: center;
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
