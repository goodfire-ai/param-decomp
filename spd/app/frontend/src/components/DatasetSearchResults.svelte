<script lang="ts">
    import type { TokenizedSearchResult } from "../lib/api/dataset";
    import TokenizedSearchResultCard from "./TokenizedSearchResultCard.svelte";

    type Props = {
        results: TokenizedSearchResult[];
        page: number;
        pageSize: number;
        totalPages: number;
        onPageChange: (page: number) => void;
        query: string;
    };

    let { results, page, pageSize, totalPages, onPageChange, query }: Props = $props();

    function prevPage() {
        if (page > 1) onPageChange(page - 1);
    }

    function nextPage() {
        if (page < totalPages) onPageChange(page + 1);
    }
</script>

<div class="results-container">
    <div class="results-list">
        {#each results as result, idx (idx)}
            <TokenizedSearchResultCard {result} index={(page - 1) * pageSize + idx} {query} />
        {/each}
    </div>

    {#if totalPages > 1}
        <div class="pagination">
            <button class="page-button" onclick={prevPage} disabled={page === 1}> Previous </button>
            <span class="page-info"> Page {page} of {totalPages} </span>
            <button class="page-button" onclick={nextPage} disabled={page === totalPages}> Next </button>
        </div>
    {/if}
</div>

<style>
    .results-container {
        display: flex;
        flex-direction: column;
        gap: var(--space-4);
        height: 100%;
    }

    .results-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        flex: 1;
        overflow-y: auto;
    }

    .pagination {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: var(--space-3);
        padding: var(--space-3) 0;
        border-top: 1px solid var(--border-default);
    }

    .page-button {
        padding: var(--space-1) var(--space-3);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-weight: 500;
        font-size: var(--text-sm);
    }

    .page-button:hover:not(:disabled) {
        background: var(--accent-primary);
        color: white;
        border-color: var(--accent-primary);
    }

    .page-button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
        cursor: not-allowed;
    }

    .page-info {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }
</style>
