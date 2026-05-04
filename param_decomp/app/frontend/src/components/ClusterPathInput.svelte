<script lang="ts">
    import { getContext } from "svelte";
    import { loadClusterMapping } from "../lib/api";
    import { CANONICAL_RUNS } from "../lib/registry";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    const loadedRun = $derived.by(() => {
        const run = runState.run;
        if (run.status !== "loaded") {
            throw new Error("Run is not loaded");
        }
        return run.data;
    });

    let inputPath = $state("");
    let loading = $state(false);
    let error = $state<string | null>(null);
    let showDropdown = $state(false);
    let showLoadedTooltip = $state(false);

    const availableClusterMappings = $derived(
        CANONICAL_RUNS.find((r) => r.wandbRunId === loadedRun.wandb_path)?.clusterMappings ?? [],
    );

    const loadedClusterNotes = $derived(
        availableClusterMappings.find((m) => m.path === runState.clusterMapping?.filePath)?.notes ?? null,
    );

    async function handleLoad() {
        const path = inputPath.trim();
        if (!path) return;

        loading = true;
        error = null;
        try {
            const result = await loadClusterMapping(path);
            runState.setClusterMapping(result.mapping, path, loadedRun.wandb_path);
            inputPath = "";
        } catch (e) {
            error = e instanceof Error ? e.message : "Failed to load";
        } finally {
            loading = false;
        }
    }

    function handleClear() {
        runState.clearClusterMapping();
        error = null;
    }

    function handleKeydown(e: KeyboardEvent) {
        if (e.key === "Enter") {
            handleLoad();
        }
    }

    function clusterRunId(path: string): string {
        return path.split("/").at(-2) ?? path;
    }

    async function selectClusterMapping(path: string) {
        showDropdown = false;
        inputPath = path;
        await handleLoad();
    }
</script>

<div class="cluster-input-wrapper">
    {#if runState.clusterMapping}
        <div
            class="cluster-loaded"
            role="group"
            onmouseenter={() => (showLoadedTooltip = true)}
            onmouseleave={() => (showLoadedTooltip = false)}
        >
            <span class="cluster-label">Clusters:</span>
            <span class="cluster-path">
                {clusterRunId(runState.clusterMapping.filePath)}
            </span>
            <button type="button" class="clear-button" onclick={handleClear} title="Clear cluster mapping"> x </button>
            {#if showLoadedTooltip}
                <div class="cluster-tooltip">
                    {#if loadedClusterNotes}
                        <div class="tooltip-notes">{loadedClusterNotes}</div>
                    {/if}
                    <div class="tooltip-path">{runState.clusterMapping.filePath}</div>
                </div>
            {/if}
        </div>
    {:else}
        <div class="cluster-form">
            <div class="input-with-dropdown">
                <input
                    type="text"
                    placeholder="path/to/clustering_run/cluster_mapping.json"
                    bind:value={inputPath}
                    onkeydown={handleKeydown}
                    disabled={loading}
                    class:has-dropdown={availableClusterMappings.length > 0}
                />
                {#if availableClusterMappings.length > 0}
                    <div
                        class="dropdown-wrapper"
                        role="group"
                        onmouseenter={() => (showDropdown = true)}
                        onmouseleave={() => (showDropdown = false)}
                    >
                        <button type="button" class="dropdown-button" title="Select from predefined mappings">
                            ▼
                        </button>
                        {#if showDropdown}
                            <div class="cluster-dropdown">
                                {#each availableClusterMappings as mapping (mapping.path)}
                                    <button
                                        type="button"
                                        class="cluster-entry"
                                        onclick={() => selectClusterMapping(mapping.path)}
                                        title={mapping.path}
                                    >
                                        <span class="entry-id">
                                            {clusterRunId(mapping.path)}
                                        </span>
                                        <span class="entry-notes">{mapping.notes}</span>
                                    </button>
                                {/each}
                            </div>
                        {/if}
                    </div>
                {/if}
            </div>
            <button type="button" onclick={handleLoad} disabled={loading || !inputPath.trim()}>
                {loading ? "..." : "Load"}
            </button>
        </div>
        {#if error}
            <span class="error-text" title={error}>Error</span>
        {/if}
    {/if}
</div>

<style>
    .cluster-input-wrapper {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .cluster-form {
        display: flex;
        align-items: center;
        gap: var(--space-1);
    }

    .input-with-dropdown {
        display: flex;
        align-items: stretch;
    }

    .cluster-form input {
        width: 260px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
    }

    .cluster-form input.has-dropdown {
        border-top-right-radius: 0;
        border-bottom-right-radius: 0;
        border-right: none;
    }

    .cluster-form input::placeholder {
        color: var(--text-muted);
    }

    .cluster-form input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .dropdown-wrapper {
        position: relative;
    }

    .dropdown-button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-top-right-radius: var(--radius-sm);
        border-bottom-right-radius: var(--radius-sm);
        border-top-left-radius: 0;
        border-bottom-left-radius: 0;
        cursor: pointer;
        font-size: var(--text-xs);
        color: var(--text-secondary);
        height: 100%;
        display: flex;
        align-items: center;
    }

    .dropdown-button:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
    }

    .cluster-dropdown {
        position: absolute;
        top: 100%;
        right: 0;
        padding-top: var(--space-1);
        z-index: 1000;
        min-width: 280px;
    }

    .cluster-dropdown > :first-child {
        border-top-left-radius: var(--radius-md);
        border-top-right-radius: var(--radius-md);
    }

    .cluster-dropdown > :last-child {
        border-bottom-left-radius: var(--radius-md);
        border-bottom-right-radius: var(--radius-md);
    }

    .cluster-entry {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: var(--space-1);
        width: 100%;
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-top: none;
        cursor: pointer;
        text-align: left;
        font-family: var(--font-sans);
        border-radius: 0;
    }

    .cluster-entry:first-child {
        border-top: 1px solid var(--border-default);
    }

    .cluster-entry:hover {
        background: var(--bg-inset);
    }

    .entry-id {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        font-weight: 600;
        color: var(--accent-primary);
    }

    .entry-notes {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .cluster-form button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        cursor: pointer;
        font-size: var(--text-xs);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
    }

    .cluster-form button:hover:not(:disabled) {
        background: var(--bg-inset);
        color: var(--text-primary);
    }

    .cluster-form button:disabled {
        cursor: not-allowed;
        opacity: 0.5;
    }

    .cluster-loaded {
        position: relative;
        display: flex;
        align-items: center;
        gap: var(--space-1);
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--accent-primary-dim);
        border-radius: var(--radius-sm);
    }

    .cluster-tooltip {
        position: absolute;
        top: 100%;
        left: 0;
        margin-top: var(--space-1);
        padding: var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
        z-index: 1000;
        min-width: 200px;
        max-width: 400px;
    }

    .tooltip-notes {
        font-size: var(--text-sm);
        color: var(--text-primary);
        margin-bottom: var(--space-1);
    }

    .tooltip-path {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        word-break: break-all;
    }

    .cluster-label {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .cluster-path {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--accent-primary);
        max-width: 150px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .clear-button {
        padding: 0 var(--space-1);
        background: transparent;
        border: none;
        cursor: pointer;
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-mono);
        line-height: 1;
    }

    .clear-button:hover {
        color: var(--text-primary);
    }

    .error-text {
        font-size: var(--text-xs);
        color: var(--accent-negative);
        cursor: help;
    }
</style>
