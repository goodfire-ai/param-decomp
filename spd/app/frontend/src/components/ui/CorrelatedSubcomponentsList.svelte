<script lang="ts">
    import { getContext } from "svelte";
    import type { CorrelatedSubcomponent } from "../../lib/promptAttributionsTypes";
    import { displaySettings } from "../../lib/displaySettings.svelte";
    import { RUN_KEY, type RunContext } from "../../lib/useRun.svelte";
    import SetOverlapVis from "./SetOverlapVis.svelte";
    import { lerp } from "../prompt-attr/graphUtils";

    const runState = getContext<RunContext>(RUN_KEY);

    type Props = {
        items: CorrelatedSubcomponent[];
        onComponentClick?: (componentKey: string) => void;
        pageSize: number;
    };

    let { items, onComponentClick, pageSize = 40 }: Props = $props();

    function getInterpretationLabel(componentKey: string): string | null {
        const interp = runState.getInterpretation(componentKey);
        if (interp.status === "loaded" && interp.data.status === "generated") return interp.data.data.label;
        return null;
    }

    let currentPage = $state(0);
    let hoveredKey = $state<string | null>(null);
    const totalPages = $derived(Math.ceil(items.length / pageSize));
    const paginatedItems = $derived(items.slice(currentPage * pageSize, (currentPage + 1) * pageSize));

    function getBorderColor(score: number): string {
        const intensity = lerp(0.3, 1, score);
        return `rgba(22, 163, 74, ${intensity})`;
    }
</script>

<div class="component-pill-list">
    {#if totalPages > 1}
        <div class="pagination">
            <button onclick={() => currentPage--} disabled={currentPage === 0}>&lt;</button>
            <span>{currentPage + 1} / {totalPages}</span>
            <button onclick={() => currentPage++} disabled={currentPage >= totalPages - 1}>&gt;</button>
        </div>
    {/if}
    <div class="components">
        {#each paginatedItems as { component_key, score, count_i, count_j, count_ij, n_tokens } (component_key)}
            {@const borderColor = getBorderColor(score)}
            {@const label = getInterpretationLabel(component_key)}
            <button
                class="component-pill"
                class:clickable={!!onComponentClick}
                style="border-left: 8px solid {borderColor};"
                onclick={() => onComponentClick?.(component_key)}
                onmouseenter={() => (hoveredKey = component_key)}
                onmouseleave={() => (hoveredKey = null)}
                title={component_key}
            >
                <div class="pill-content">
                    {#if label}
                        <span class="interp-label">{label}</span>
                    {:else}
                        <span class="component-text">{component_key}</span>
                    {/if}
                    <span class="component-text">({score.toFixed(2)})</span>
                </div>
                {#if displaySettings.showSetOverlapVis && n_tokens > 0}
                    <SetOverlapVis
                        countA={count_i}
                        countB={count_j}
                        countIntersection={count_ij}
                        totalCount={n_tokens}
                        relativeTo={hoveredKey === component_key ? "union" : "population"}
                    />
                {/if}
            </button>
        {/each}
    </div>
</div>

<style>
    .component-pill-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .components {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-2);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        background: var(--bg-elevated);
        padding: var(--space-2);
        border: 1px solid var(--border-default);
    }

    .pagination {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .pagination button {
        padding: 2px 6px;
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-secondary);
        cursor: pointer;
        font-size: var(--text-xs);
    }

    .pagination button:hover:not(:disabled) {
        background: var(--bg-surface);
        border-color: var(--border-strong);
    }

    .pagination button:disabled {
        opacity: 0.4;
        cursor: default;
    }

    .component-pill {
        display: inline-flex;
        flex-direction: column;
        gap: var(--space-1);
        padding: var(--space-1) var(--space-2);
        white-space: nowrap;
        cursor: default;
        position: relative;
        border: none;
        background: var(--bg-surface);
        color: var(--text-primary);
        font-family: inherit;
        font-size: inherit;
        min-width: 80px;
    }

    .pill-content {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--space-1);
    }

    .component-pill.clickable {
        cursor: pointer;
    }

    .interp-label {
        font-family: var(--font-sans);
        font-weight: 500;
        max-width: 120px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
</style>
