<script lang="ts">
    type ViewType = "graph" | "interventions";

    type Props = {
        activeView: ViewType;
        versionCount: number;
        onViewChange: (view: ViewType) => void;
    };

    let { activeView, versionCount, onViewChange }: Props = $props();
</script>

<div class="view-tabs">
    <button class="view-tab" class:active={activeView === "graph"} onclick={() => onViewChange("graph")}>
        Attributions
    </button>
    <button
        class="view-tab"
        class:active={activeView === "interventions"}
        onclick={() => onViewChange("interventions")}
    >
        Interventions
        {#if versionCount > 1}
            <span class="badge">{versionCount}</span>
        {/if}
    </button>
</div>

<style>
    .view-tabs {
        display: flex;
        margin-bottom: -1px; /* overlap with content border */
        gap: var(--space-1);
    }

    .view-tab {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
        /* border-bottom: none; */
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: var(--space-1);
        margin-right: -1px; /* connect tabs */
        position: relative;
    }

    .view-tab:hover:not(.active) {
        color: var(--text-secondary);
        background: var(--bg-surface);
    }

    .view-tab.active {
        background: var(--bg-base);
        border-color: var(--border-default);
        border-bottom: 1px solid var(--bg-base); /* hide bottom border */
        color: var(--text-primary);
        font-weight: 500;
        z-index: 1; /* above the border-bottom of container */
    }

    .badge {
        padding: 0 var(--space-1);
        background: var(--accent-primary-dim);
        border-radius: var(--radius-sm);
        font-size: var(--text-xs);
        color: var(--accent-primary-bright);
    }
</style>
