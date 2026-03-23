<script lang="ts">
    import { getContext } from "svelte";
    import { SvelteMap } from "svelte/reactivity";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import type { PairCorrelation, ClusterPairwiseResponse } from "../lib/api/clusters";
    import { fetchClusterPairwiseCorrelations, fetchClusteringCoactivation } from "../lib/api/clusters";

    type ComponentMember = { layer: string; cIdx: number };

    type Props = {
        members: ComponentMember[];
        clusteringRunId: string;
    };

    let { members, clusteringRunId }: Props = $props();

    const runState = getContext<RunContext>(RUN_KEY);

    type DataSource = "harvest" | "clustering";
    type Metric = "jaccard" | "precision_ab" | "pmi";

    let selectedSource = $state<DataSource>("harvest");
    let selectedMetric = $state<Metric>("jaccard");

    type PairwiseData =
        | { status: "loading" }
        | { status: "loaded"; response: ClusterPairwiseResponse }
        | { status: "error"; error: string };

    let harvestData = $state<PairwiseData>({ status: "loading" });
    let clusteringData = $state<PairwiseData>({ status: "loading" });

    const keys = $derived(members.map((m) => `${m.layer}:${m.cIdx}`));

    function getLabel(key: string): string {
        const loadable = runState.getInterpretation(key);
        if (loadable.status === "loaded" && loadable.data.status === "generated") {
            return loadable.data.data.label;
        }
        return key;
    }

    $effect(() => {
        const currentKeys = keys;
        if (currentKeys.length < 2) {
            harvestData = { status: "loaded", response: { pairs: [], n_tokens: 0 } };
            clusteringData = { status: "loaded", response: { pairs: [], n_tokens: 0 } };
            return;
        }

        harvestData = { status: "loading" };
        fetchClusterPairwiseCorrelations(currentKeys)
            .then((response) => (harvestData = { status: "loaded", response }))
            .catch((e) => (harvestData = { status: "error", error: String(e) }));

        clusteringData = { status: "loading" };
        fetchClusteringCoactivation(currentKeys, clusteringRunId)
            .then((response) => (clusteringData = { status: "loaded", response }))
            .catch((e) => (clusteringData = { status: "error", error: String(e) }));
    });

    // Build lookups from pair data
    function buildPairLookup(pairs: PairCorrelation[]): SvelteMap<string, PairCorrelation> {
        const map = new SvelteMap<string, PairCorrelation>();
        for (const pair of pairs) {
            map.set(`${pair.key_a}|${pair.key_b}`, pair);
            map.set(`${pair.key_b}|${pair.key_a}`, pair);
        }
        return map;
    }

    const activePairData = $derived.by((): PairwiseData => {
        switch (selectedSource) {
            case "harvest":
                return harvestData;
            case "clustering":
                return clusteringData;
        }
    });

    const pairLookup = $derived.by(() => {
        const data = activePairData;
        if (data.status !== "loaded") return new SvelteMap<string, PairCorrelation>();
        return buildPairLookup(data.response.pairs);
    });

    function getCellValue(rowKey: string, colKey: string): number | null {
        if (rowKey === colKey) return null;
        const pair = pairLookup.get(`${rowKey}|${colKey}`);
        if (!pair) return null;
        switch (selectedMetric) {
            case "jaccard":
                return pair.jaccard;
            case "pmi":
                return pair.pmi;
            case "precision_ab":
                if (pair.key_a === rowKey) return pair.precision_ab;
                return pair.precision_ba;
        }
    }

    function cellColor(val: number | null): string {
        if (val === null) return "transparent";

        if (selectedMetric === "pmi") {
            if (val > 0) {
                const norm = Math.min(val / 6, 1);
                return `rgba(59, 130, 246, ${norm * 0.7})`;
            } else {
                const norm = Math.min(Math.abs(val) / 6, 1);
                return `rgba(239, 68, 68, ${norm * 0.7})`;
            }
        }

        const norm = Math.min(Math.max(val, 0), 1);
        return `rgba(59, 130, 246, ${norm * 0.7})`;
    }

    function cellTitle(rowKey: string, colKey: string): string {
        const pair = pairLookup.get(`${rowKey}|${colKey}`);
        if (!pair) return "";
        const precAB = pair.key_a === rowKey ? pair.precision_ab : pair.precision_ba;
        const precBA = pair.key_a === rowKey ? pair.precision_ba : pair.precision_ab;
        return `${rowKey} × ${colKey}\nco-fire: ${pair.count_ab} / A: ${pair.count_a} / B: ${pair.count_b}\nJaccard: ${pair.jaccard.toFixed(4)}\nP(row|col): ${precAB.toFixed(4)}\nP(col|row): ${precBA.toFixed(4)}\nPMI: ${pair.pmi !== null ? pair.pmi.toFixed(4) : "−∞"}`;
    }

    const currentStatus = $derived(activePairData.status);

    const currentError = $derived.by((): string => {
        const data = activePairData;
        if (data.status === "error") return data.error;
        return "";
    });

    const sourceDescription = $derived.by((): string => {
        const metricExplanation =
            selectedMetric === "jaccard"
                ? "Each cell shows the Jaccard index: the proportion of tokens where at least one of the pair fires that both fire."
                : selectedMetric === "precision_ab"
                  ? "Each cell (row, col) shows the proportion of row's activations where col is also active."
                  : "Each cell shows the pointwise mutual information: log-ratio of observed co-firing vs expected under independence. Blue = positive, red = negative.";

        switch (selectedSource) {
            case "harvest":
                return `Co-firing statistics from the harvest pipeline (ci_threshold from harvest config). ${metricExplanation}`;
            case "clustering":
                return `Co-firing statistics from the clustering's own membership snapshot (activation_threshold from merge config) — the exact data the algorithm used to decide merges. ${metricExplanation}`;
        }
    });

    const footerText = $derived.by((): string => {
        const data = activePairData;
        if (data.status === "loaded") {
            return `${data.response.n_tokens.toLocaleString()} tokens · ${data.response.pairs.length} pairs`;
        }
        return "";
    });
</script>

<div class="correlation-matrix">
    <div class="matrix-controls">
        <div class="source-toggle">
            <span class="control-label">Source:</span>
            <button class="source-btn" class:active={selectedSource === "harvest"} onclick={() => (selectedSource = "harvest")}>Harvest</button>
            <button class="source-btn" class:active={selectedSource === "clustering"} onclick={() => (selectedSource = "clustering")}>Clustering</button>
        </div>
        <div class="metric-toggle">
            <span class="control-label">Metric:</span>
            <button class="metric-btn" class:active={selectedMetric === "jaccard"} onclick={() => (selectedMetric = "jaccard")}>Jaccard</button>
            <button class="metric-btn" class:active={selectedMetric === "precision_ab"} onclick={() => (selectedMetric = "precision_ab")}>Precision</button>
            <button class="metric-btn" class:active={selectedMetric === "pmi"} onclick={() => (selectedMetric = "pmi")}>PMI</button>
        </div>
    </div>
    <div class="source-description">{sourceDescription}</div>

    <div class="color-legend">
        {#if selectedMetric === "pmi"}
            <span class="legend-label">-6</span>
            <div class="legend-bar legend-bar-diverging"></div>
            <span class="legend-label">0</span>
            <div class="legend-bar legend-bar-diverging-pos"></div>
            <span class="legend-label">+6</span>
        {:else}
            <span class="legend-label">0</span>
            <div class="legend-bar legend-bar-sequential"></div>
            <span class="legend-label">1</span>
        {/if}
    </div>

    {#if currentStatus === "loading"}
        <div class="loading">Loading...</div>
    {:else if currentStatus === "error"}
        <div class="error">{currentError}</div>
    {:else if keys.length < 2}
        <div class="empty">Need at least 2 components for pairwise correlations.</div>
    {:else}
        <div class="matrix-scroll">
            <table class="matrix-table">
                <thead>
                    <tr>
                        <th class="corner"></th>
                        {#each keys as colKey (colKey)}
                            <th class="col-header" title={colKey}>
                                <div class="header-text">{getLabel(colKey)}</div>
                            </th>
                        {/each}
                    </tr>
                </thead>
                <tbody>
                    {#each keys as rowKey (rowKey)}
                        <tr>
                            <th class="row-header" title={rowKey}>{getLabel(rowKey)}</th>
                            {#each keys as colKey (colKey)}
                                {@const val = getCellValue(rowKey, colKey)}
                                <td
                                    class="cell"
                                    class:diagonal={rowKey === colKey}
                                    style="background: {cellColor(val)}"
                                    title={rowKey === colKey ? rowKey : cellTitle(rowKey, colKey)}
                                ></td>
                            {/each}
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
        <div class="matrix-footer">{footerText}</div>
    {/if}
</div>

<style>
    .correlation-matrix {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
    }

    .matrix-controls {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        flex-wrap: wrap;
    }

    .source-toggle,
    .metric-toggle {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .source-description {
        font-size: var(--text-xs);
        color: var(--text-muted);
        line-height: 1.4;
        max-width: 600px;
    }

    .control-label {
        font-size: var(--text-sm);
        color: var(--text-muted);
    }

    .source-btn,
    .metric-btn {
        padding: var(--space-1) var(--space-3);
        font: inherit;
        font-size: var(--text-sm);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        cursor: pointer;
        color: var(--text-secondary);
        transition:
            background var(--transition-normal),
            border-color var(--transition-normal);
    }

    .source-btn:hover,
    .metric-btn:hover {
        background: var(--bg-elevated);
        border-color: var(--border-strong);
    }

    .source-btn.active {
        background: var(--bg-elevated);
        border-color: var(--accent-positive);
        color: var(--text-primary);
    }

    .metric-btn.active {
        background: var(--bg-elevated);
        border-color: var(--accent-primary);
        color: var(--text-primary);
    }

    .loading,
    .error,
    .empty {
        font-size: var(--text-sm);
        color: var(--text-muted);
        padding: var(--space-4);
    }

    .error {
        color: var(--status-error);
    }

    .matrix-scroll {
        overflow: auto;
    }

    .matrix-table {
        border-collapse: collapse;
        font-size: var(--text-xs);
        font-family: var(--font-mono);
    }

    .corner {
        position: sticky;
        top: 0;
        left: 0;
        z-index: 2;
        background: var(--bg-base);
    }

    .col-header {
        position: sticky;
        top: 0;
        z-index: 1;
        background: var(--bg-base);
        padding: var(--space-1);
        max-width: 100px;
        vertical-align: bottom;
    }

    .header-text {
        writing-mode: vertical-rl;
        transform: rotate(180deg);
        max-height: 120px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-weight: 500;
        color: var(--text-secondary);
    }

    .row-header {
        position: sticky;
        left: 0;
        z-index: 1;
        background: var(--bg-base);
        text-align: right;
        padding: var(--space-1) var(--space-2);
        max-width: 160px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-weight: 500;
        color: var(--text-secondary);
    }

    .cell {
        padding: 0;
        min-width: 20px;
        height: 20px;
        border: 1px solid var(--border-subtle);
        cursor: default;
    }

    .cell.diagonal {
        background: var(--bg-inset) !important;
    }

    .color-legend {
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .legend-label {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-mono);
    }

    .legend-bar {
        width: 100px;
        height: 12px;
        border-radius: 2px;
        border: 1px solid var(--border-subtle);
    }

    .legend-bar-sequential {
        background: linear-gradient(to right, transparent, rgba(59, 130, 246, 0.7));
    }

    .legend-bar-diverging {
        background: linear-gradient(to right, rgba(239, 68, 68, 0.7), transparent);
    }

    .legend-bar-diverging-pos {
        background: linear-gradient(to right, transparent, rgba(59, 130, 246, 0.7));
    }

    .matrix-footer {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }
</style>
