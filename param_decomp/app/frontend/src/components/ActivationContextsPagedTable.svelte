<script lang="ts">
    import type { Loadable } from "../lib/index";
    import { displaySettings } from "../lib/displaySettings.svelte";
    import TokenHighlights from "./TokenHighlights.svelte";

    export type ActivationExamplesData = {
        tokens: string[][]; // [n_examples, window_size]
        ci: number[][]; // [n_examples, window_size]
        componentActs: number[][]; // [n_examples, window_size]
        maxAbsComponentAct: number;
    };

    interface Props {
        data: Loadable<ActivationExamplesData>;
    }

    let { data }: Props = $props();

    const loading = $derived(data.status !== "loaded");
    const loaded = $derived(data.status === "loaded" ? data.data : null);

    let examplesEl = $state<HTMLDivElement | undefined>(undefined);
    let currentPage = $state(0);
    let pageSize = $state(10);

    let nExamples = $derived(loaded?.tokens.length ?? 0);

    function argmax(arr: number[]): number {
        let maxIdx = 0;
        for (let i = 1; i < arr.length; i++) {
            if (arr[i] > arr[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    let firingPositions = $derived(loaded?.ci.map(argmax) ?? []);

    // Minimum container width (in ch) so that per-row flex centering works without clipping.
    // Each row needs: 2 * max(leftWidth, rightWidth) + centerWidth.
    // Each token adds ~0.3ch overhead for border + margin beyond its character width.
    const TOKEN_OVERHEAD_CH = 0.3;

    let minWidthCh = $derived.by(() => {
        if (!displaySettings.centerOnPeak || !loaded) return 0;
        let max = 0;
        for (let i = 0; i < loaded.tokens.length; i++) {
            const fp = firingPositions[i];
            const tokens = loaded.tokens[i];

            let leftWidth = 0;
            for (let j = 0; j < fp; j++) leftWidth += tokens[j].length + TOKEN_OVERHEAD_CH;

            let rightWidth = 0;
            for (let j = fp + 1; j < tokens.length; j++) rightWidth += tokens[j].length + TOKEN_OVERHEAD_CH;

            const centerWidth = tokens[fp].length + TOKEN_OVERHEAD_CH;
            const required = 2 * Math.max(leftWidth, rightWidth) + centerWidth;
            if (required > max) max = required;
        }
        return Math.ceil(max + 1);
    });

    // Update currentPage when page input changes
    function handlePageInput(event: Event) {
        const { value } = event.target as HTMLInputElement;
        if (value === "") return;
        const valueNum = parseInt(value);
        if (!isNaN(valueNum) && valueNum >= 1 && valueNum <= totalPages) {
            currentPage = valueNum - 1;
        } else {
            throw new Error(`Invalid page number: ${value} (must be 1-${totalPages})`);
        }
    }

    let allIndices = $derived(Array.from({ length: nExamples }, (_, i) => i));

    let paginatedIndices = $derived.by(() => {
        const start = currentPage * pageSize;
        const end = start + pageSize;
        return allIndices.slice(start, end);
    });

    let totalPages = $derived(Math.ceil(allIndices.length / pageSize));

    function previousPage() {
        if (currentPage > 0) currentPage--;
    }

    function nextPage() {
        if (currentPage < totalPages - 1) currentPage++;
    }

    // Reset to page 0 when data or page size changes
    $effect(() => {
        void loaded;
        void pageSize;
        currentPage = 0;
    });

    function centerScroll() {
        if (!examplesEl) return;
        examplesEl.scrollLeft = (examplesEl.scrollWidth - examplesEl.clientWidth) / 2;
    }

    $effect(() => {
        if (!displaySettings.centerOnPeak) return;
        void paginatedIndices;
        requestAnimationFrame(centerScroll);
    });
</script>

<div class="container">
    <div class="controls">
        <div class="pagination">
            <button disabled={loading || currentPage === 0} onclick={previousPage}>&lt;</button>
            <input
                type="number"
                min="1"
                max={totalPages}
                value={loading ? "" : currentPage + 1}
                oninput={handlePageInput}
                class="page-input"
                disabled={loading}
            />
            <span>of {loading ? "-" : totalPages}</span>
            <button disabled={loading || currentPage === totalPages - 1} onclick={nextPage}>&gt;</button>
        </div>
        <div class="page-size-control">
            <label for="page-size">Per page:</label>
            <select id="page-size" bind:value={pageSize} disabled={loading}>
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
            </select>
        </div>
        <label class="center-toggle">
            <input type="checkbox" bind:checked={displaySettings.centerOnPeak} disabled={loading} />
            Center on peak
        </label>
    </div>
    {#if loading}
        <div class="examples">
            <div class="examples-inner">
                {#each Array(pageSize) as _, i (i)}
                    <div class="skeleton-row"></div>
                {/each}
            </div>
        </div>
    {:else}
        {@const d = loaded!}
        <div class="examples" bind:this={examplesEl}>
            {#if displaySettings.centerOnPeak}
                <div class="examples-inner" style="min-width: {minWidthCh}ch">
                    {#each paginatedIndices as idx (idx)}
                        {@const fp = firingPositions[idx]}
                        <div class="example-row">
                            <div class="left-tokens">
                                <TokenHighlights
                                    tokenStrings={d.tokens[idx].slice(0, fp)}
                                    tokenCi={d.ci[idx].slice(0, fp)}
                                    tokenComponentActs={d.componentActs[idx].slice(0, fp)}
                                    maxAbsComponentAct={d.maxAbsComponentAct}
                                />
                            </div>
                            <div class="center-token">
                                <TokenHighlights
                                    tokenStrings={[d.tokens[idx][fp]]}
                                    tokenCi={[d.ci[idx][fp]]}
                                    tokenComponentActs={[d.componentActs[idx][fp]]}
                                    maxAbsComponentAct={d.maxAbsComponentAct}
                                />
                            </div>
                            <div class="right-tokens">
                                <TokenHighlights
                                    tokenStrings={d.tokens[idx].slice(fp + 1)}
                                    tokenCi={d.ci[idx].slice(fp + 1)}
                                    tokenComponentActs={d.componentActs[idx].slice(fp + 1)}
                                    maxAbsComponentAct={d.maxAbsComponentAct}
                                />
                            </div>
                        </div>
                    {/each}
                </div>
            {:else}
                <div class="examples-inner">
                    {#each paginatedIndices as idx (idx)}
                        <div class="example-item">
                            <TokenHighlights
                                tokenStrings={d.tokens[idx]}
                                tokenCi={d.ci[idx]}
                                tokenComponentActs={d.componentActs[idx]}
                                maxAbsComponentAct={d.maxAbsComponentAct}
                            />
                        </div>
                    {/each}
                </div>
            {/if}
        </div>
    {/if}
</div>

<style>
    .container {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
    }

    .examples {
        padding: var(--space-2);
        overflow-x: auto;
        overflow-y: clip;
    }

    .examples-inner {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
        min-width: 100%;
    }

    .example-row {
        display: flex;
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        line-height: 1.8;
        color: var(--text-primary);
        white-space: nowrap;
    }

    .example-item {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        line-height: 1.8;
        color: var(--text-primary);
        white-space: nowrap;
    }

    .left-tokens {
        flex: 1 1 0;
        min-width: 0;
        text-align: right;
    }

    .center-token {
        flex: 0 0 auto;
    }

    .right-tokens {
        flex: 1 1 0;
        min-width: 0;
        text-align: left;
    }

    .controls {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        padding: var(--space-2);
        border-bottom: 1px solid var(--border-default);
        flex-wrap: wrap;
    }

    .center-toggle {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
        cursor: pointer;
        margin-left: auto;
    }

    .center-toggle input {
        cursor: pointer;
    }

    .page-size-control {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .page-size-control label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        white-space: nowrap;
        font-weight: 500;
    }

    .page-size-control select {
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        cursor: pointer;
        min-width: 100px;
    }

    .page-size-control select:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .pagination {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .pagination button {
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-secondary);
    }

    .pagination button:hover:not(:disabled) {
        background: var(--bg-inset);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    .pagination button:disabled {
        opacity: 0.5;
    }

    .pagination span {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
        white-space: nowrap;
    }

    .page-input {
        width: 50px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        appearance: textfield;
    }

    .page-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .page-input::-webkit-inner-spin-button,
    .page-input::-webkit-outer-spin-button {
        appearance: none;
        margin: 0;
    }

    .skeleton-row {
        height: calc(var(--text-sm) * 1.8);
        border-radius: var(--radius-sm);
        background: var(--border-default);
        opacity: 0.3;
        animation: skeleton-pulse 1.2s ease-in-out infinite;
    }

    @keyframes skeleton-pulse {
        0%,
        100% {
            opacity: 0.3;
        }
        50% {
            opacity: 0.1;
        }
    }
</style>
