<script lang="ts">
    import * as api from "../lib/api";
    import type { Loadable } from "../lib";
    import DatasetSearchResults from "./DatasetSearchResults.svelte";

    let searchQuery = $state("");
    let searchSplit = $state<"train" | "test">("train");
    let searchMetadata = $state<api.DatasetSearchMetadata | null>(null);
    let searchPage = $state(1);
    let searchPageSize = $state(10);
    let searchResults = $state<Loadable<api.TokenizedSearchPage>>({ status: "uninitialized" });

    async function performSearch() {
        if (!searchQuery.trim()) return;

        searchResults = { status: "loading" };
        searchMetadata = null;
        searchPage = 1;

        try {
            const result = await api.searchDataset(searchQuery.trim(), searchSplit);
            searchMetadata = result;
            await loadSearchPage(1);
        } catch (e) {
            searchResults = { status: "error", error: e };
        }
    }

    async function loadSearchPage(page: number) {
        searchResults = { status: "loading" };
        try {
            const data = await api.getTokenizedResults(page, searchPageSize);
            searchResults = { status: "loaded", data };
            searchPage = page;
        } catch (e) {
            searchResults = { status: "error", error: e };
        }
    }

    function handleSearchKeydown(event: KeyboardEvent) {
        if (event.key === "Enter" && searchResults.status !== "loading") {
            performSearch();
        }
    }
</script>

<div class="panel">
    <div class="config-box">
        <div class="config-header">
            <span class="config-title">Search Dataset{searchMetadata ? `: ${searchMetadata.dataset_name}` : ""}</span>
            <button
                class="action-button"
                onclick={performSearch}
                disabled={searchResults.status === "loading" || !searchQuery.trim()}
            >
                {searchResults.status === "loading" ? "Searching..." : "Search"}
            </button>
        </div>
        <div class="form-grid">
            <div class="form-row">
                <label for="search-query">Query:</label>
                <input
                    id="search-query"
                    type="text"
                    placeholder="e.g. 'dragon' or 'went to the'"
                    bind:value={searchQuery}
                    onkeydown={handleSearchKeydown}
                    disabled={searchResults.status === "loading"}
                />
            </div>
            <div class="form-row">
                <label for="search-split">Split:</label>
                <select id="search-split" bind:value={searchSplit} disabled={searchResults.status === "loading"}>
                    <option value="train">Train</option>
                    <option value="test">Test</option>
                </select>
            </div>
        </div>
        {#if searchMetadata}
            <div class="metadata">
                Found {searchMetadata.total_results} results in {searchMetadata.search_time_seconds.toFixed(2)}s
            </div>
        {/if}
    </div>

    <div class="results-box">
        {#if searchResults.status === "loaded"}
            <DatasetSearchResults
                results={searchResults.data.results}
                page={searchPage}
                pageSize={searchPageSize}
                totalPages={searchResults.data.total_pages}
                onPageChange={loadSearchPage}
                query={searchQuery}
            />
        {:else if searchResults.status === "loading"}
            <div class="empty-state">Searching dataset...</div>
        {:else if searchResults.status === "error"}
            <div class="empty-state error">Error: {searchResults.error}</div>
        {:else}
            <div class="empty-state">
                <p>No search performed yet</p>
                <p class="hint">Enter a query above to search the dataset</p>
            </div>
        {/if}
    </div>
</div>

<style>
    .panel {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
        gap: var(--space-4);
    }

    .config-box {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        padding: var(--space-4);
        border: 1px solid var(--border-default);
        background: var(--bg-inset);
    }

    .config-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .config-title {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .action-button {
        padding: var(--space-1) var(--space-3);
        border: none;
        background: var(--accent-primary);
        color: white;
        font-weight: 500;
        font-size: var(--text-sm);
        cursor: pointer;
    }

    .action-button:hover:not(:disabled) {
        background: var(--accent-primary-dim);
    }

    .action-button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
        cursor: not-allowed;
    }

    .form-grid {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
    }

    .form-row {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .form-row label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
        min-width: 60px;
    }

    .form-row input,
    .form-row select {
        flex: 1;
        max-width: 400px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
    }

    .form-row input:focus,
    .form-row select:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .metadata {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .results-box {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
        padding: var(--space-4);
        border: 1px solid var(--border-default);
        background: var(--bg-inset);
        overflow-y: auto;
    }

    .empty-state {
        display: flex;
        flex: 1;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: var(--text-muted);
        text-align: center;
        font-family: var(--font-sans);
    }

    .empty-state.error {
        color: var(--status-negative);
    }

    .empty-state p {
        margin: var(--space-1) 0;
        font-size: var(--text-base);
    }
</style>
