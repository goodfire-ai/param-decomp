<script lang="ts">
    import { colors, rgbaToCss } from "../../lib/colors";

    type TokenValue = {
        token: string;
        value: number;
    };

    type Props = {
        items: TokenValue[];
        /** Scale: value at which color reaches full intensity (1 for precision/recall, max abs observed for PMI) */
        maxScale: number;
        pageSize?: number;
    };

    let { items, maxScale, pageSize = 30 }: Props = $props();

    let page = $state(0);
    const totalPages = $derived(Math.ceil(items.length / pageSize));
    const visibleItems = $derived(items.slice(page * pageSize, (page + 1) * pageSize));
    const hasPagination = $derived(items.length > pageSize);

    function getPmiBg(value: number): string {
        if (value > 0) {
            const intensity = Math.min(1, value / maxScale);
            return rgbaToCss(colors.positiveRgb, intensity);
        } else {
            const intensity = Math.min(1, -value / maxScale);
            return rgbaToCss(colors.negativeRgb, intensity);
        }
    }

    function getTextColor(value: number): string {
        const intensity = Math.min(1, Math.abs(value) / maxScale);
        return intensity > 0.5 ? "white" : "var(--text-primary)";
    }
</script>

<div class="token-pill-list">
    <div class="tokens">
        {#each visibleItems as { token, value }, i (i)}
            <span
                class="token-pill"
                style="background: {getPmiBg(value)}; color: {getTextColor(value)}"
                title={value.toFixed(3)}>{token}</span
            >
        {/each}
    </div>
    {#if hasPagination}
        <div class="pagination">
            <button disabled={page === 0} onclick={() => (page = page - 1)}>&lsaquo;</button>
            <span class="page-info">{page + 1}/{totalPages} ({items.length})</span>
            <button disabled={page >= totalPages - 1} onclick={() => (page = page + 1)}>&rsaquo;</button>
        </div>
    {/if}
</div>

<style>
    .token-pill-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .tokens {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-2);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        background: var(--bg-elevated);
        padding: var(--space-2);
        border: 1px solid var(--border-default);
    }

    .token-pill {
        padding: var(--space-1) var(--space-1);
        border-radius: var(--radius-sm);
        white-space: pre;
        cursor: default;
        position: relative;
        box-shadow: inset 0 0 0 1px transparent;
        transition: box-shadow var(--transition-fast);
    }

    .token-pill:hover {
        box-shadow: inset 0 0 0 1px var(--border-strong);
    }

    .token-pill::after {
        content: attr(title);
        position: absolute;
        bottom: calc(100% + 4px);
        left: 50%;
        transform: translateX(-50%);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        color: var(--text-primary);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-xs);
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        z-index: 100;
        border-radius: var(--radius-sm);
    }

    .token-pill:hover::after {
        opacity: 1;
    }

    .pagination {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        align-self: flex-end;
    }

    .pagination button {
        background: none;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        color: var(--text-primary);
        cursor: pointer;
        padding: 0 6px;
        font-size: var(--text-sm);
        line-height: 1.6;
    }

    .pagination button:hover:not(:disabled) {
        border-color: var(--accent-primary);
    }

    .pagination button:disabled {
        opacity: 0.3;
        cursor: default;
    }

    .page-info {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
    }
</style>
