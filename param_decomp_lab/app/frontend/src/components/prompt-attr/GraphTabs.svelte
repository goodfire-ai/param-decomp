<script lang="ts">
    import type { StoredGraph } from "./types";

    type Props = {
        graphs: StoredGraph[];
        activeGraphId: number | null;
        onSelectGraph: (graphId: number) => void;
        onCloseGraph: (graphId: number) => void;
        onNewGraph: () => void;
    };

    let { graphs, activeGraphId, onSelectGraph, onCloseGraph, onNewGraph }: Props = $props();

    const isNewGraphMode = $derived(activeGraphId === null);
</script>

<div class="graph-tabs">
    {#each graphs as graph (graph.id)}
        <div class="graph-tab" class:active={graph.id === activeGraphId}>
            <button class="tab-label" onclick={() => onSelectGraph(graph.id)}>
                {graph.label} <span class="graph-id">#{graph.id}</span>
            </button>
            <button class="tab-close" onclick={() => onCloseGraph(graph.id)}>×</button>
        </div>
    {/each}
    {#if isNewGraphMode}
        <div class="graph-tab draft-tab active">
            <span class="tab-label draft-label">New Graph</span>
        </div>
    {/if}
    {#if graphs.length > 0 && !isNewGraphMode}
        <button class="btn-add" onclick={onNewGraph}>+</button>
    {/if}
</div>

<style>
    .graph-tabs {
        display: flex;
        gap: var(--space-1);
    }

    .graph-tab {
        display: flex;
        align-items: center;
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .graph-tab:hover {
        background: var(--bg-inset);
        border-color: var(--border-strong);
    }

    .graph-tab.active {
        background: var(--accent-primary);
        color: var(--bg-base);
    }

    .graph-tab.draft-tab {
        border-style: dashed;
        border-color: var(--text-muted);
        background: var(--bg-elevated);
    }

    .graph-tab.draft-tab.active {
        background: var(--bg-elevated);
        border-color: var(--border-strong);
        color: var(--text-primary);
    }

    .tab-label {
        padding: var(--space-1) var(--space-2);
        background: transparent;
        border: none;
        font-size: inherit;
        font-family: inherit;
        color: inherit;
        cursor: pointer;
    }

    .draft-label {
        color: var(--text-secondary);
        font-style: italic;
    }

    .graph-id {
        font-size: var(--text-xs);
        color: var(--text-muted);
        opacity: 0.7;
    }

    .tab-close {
        padding: var(--space-1);
        background: transparent;
        border: none;
        border-left: 1px solid var(--border-subtle);
        font-size: var(--text-sm);
        line-height: 1;
        opacity: 0.6;
        cursor: pointer;
        color: inherit;
    }

    .graph-tab.active .tab-close {
        border-left-color: var(--accent-primary);
    }

    .tab-close:hover {
        opacity: 1;
        color: var(--status-negative-bright);
    }

    .btn-add {
        padding: var(--space-1) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-muted);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        cursor: pointer;
    }

    .btn-add:hover {
        background: var(--bg-inset);
        border-color: var(--border-strong);
        color: var(--text-primary);
    }
</style>
