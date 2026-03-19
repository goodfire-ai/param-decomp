<script lang="ts">
    import { getContext, onMount } from "svelte";
    import { computeMaxAbsComponentAct } from "../lib/colors";
    import { mapLoadable } from "../lib/index";
    import { COMPONENT_CARD_CONSTANTS } from "../lib/componentCardConstants";
    import { anyCorrelationStatsEnabled, displaySettings } from "../lib/displaySettings.svelte";
    import type { ActivationContextsSummary, SubcomponentMetadata } from "../lib/promptAttributionsTypes";
    import { useComponentData } from "../lib/useComponentData.svelte";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import ActivationContextsPagedTable, { type ActivationExamplesData } from "./ActivationContextsPagedTable.svelte";
    import ComponentProbeInput from "./ComponentProbeInput.svelte";
    import ComponentCorrelationMetrics from "./ui/ComponentCorrelationMetrics.svelte";
    import ComponentFrequencyCurve from "./ui/ComponentFrequencyCurve.svelte";
    import GraphInterpBadge from "./ui/GraphInterpBadge.svelte";
    import InterpretationBadge from "./ui/InterpretationBadge.svelte";
    import SectionHeader from "./ui/SectionHeader.svelte";
    import StatusText from "./ui/StatusText.svelte";
    import TokenStatsSection from "./ui/TokenStatsSection.svelte";
    import DatasetAttributionsSection from "./ui/DatasetAttributionsSection.svelte";

    type Props = {
        activationContextsSummary: ActivationContextsSummary;
    };

    let { activationContextsSummary }: Props = $props();

    const runState = getContext<RunContext>(RUN_KEY);

    let availableLayers = $derived(Object.keys(activationContextsSummary).sort());
    let currentPage = $state(0);
    let selectedLayer = $state<string>(Object.keys(activationContextsSummary)[0]);

    // Layer metadata is already sorted by mean_ci desc from backend
    // Filter by mean CI cutoff for ordered navigation (but not for "go to index" search)
    let currentLayerMetadata = $derived(
        activationContextsSummary[selectedLayer].filter((m) => m.mean_ci >= displaySettings.meanCiCutoff),
    );
    let totalPages = $derived(currentLayerMetadata.length);
    let currentMetadata = $derived<SubcomponentMetadata>(currentLayerMetadata[currentPage]);
    let currentIntruderScore = $derived(
        currentMetadata ? runState.getIntruderScore(`${selectedLayer}:${currentMetadata.subcomponent_idx}`) : null,
    );
    let currentGraphInterpLabel = $derived(
        currentMetadata ? runState.getGraphInterpLabel(`${selectedLayer}:${currentMetadata.subcomponent_idx}`) : null,
    );

    // Component data hook - call load() explicitly when component changes
    const componentData = useComponentData();
    const DEBOUNCE_MS = 250;
    let loadTimeout: ReturnType<typeof setTimeout> | null = null;

    // Load data for current component (debounced to avoid spamming on rapid navigation)
    function loadCurrentComponent() {
        if (loadTimeout) clearTimeout(loadTimeout);
        componentData.reset(); // Clear stale data immediately
        loadTimeout = setTimeout(() => {
            const cIdx = currentMetadata?.subcomponent_idx;
            if (cIdx !== undefined) {
                componentData.load(selectedLayer, cIdx);
            }
        }, DEBOUNCE_MS);
    }

    // Initial load on mount, cleanup on unmount
    onMount(() => {
        loadCurrentComponent();
        return () => {
            if (loadTimeout) clearTimeout(loadTimeout);
        };
    });

    // Reset page if cutoff changes cause current page to be out of bounds
    $effect(() => {
        if (currentPage >= totalPages && totalPages > 0) {
            currentPage = 0;
            loadCurrentComponent();
        }
    });

    function handlePageInput(event: Event) {
        const target = event.target as HTMLInputElement;
        if (target.value === "") return;
        const value = parseInt(target.value);
        if (!isNaN(value) && value >= 1 && value <= totalPages) {
            currentPage = value - 1;
            loadCurrentComponent();
        }
    }

    // Search for a specific subcomponent index
    let searchValue = $state("");
    let searchError = $state<string | null>(null);

    function handleSearchInput(event: Event) {
        const target = event.target as HTMLInputElement;
        searchValue = target.value;
        searchError = null;

        if (searchValue === "") return;

        const targetIdx = parseInt(searchValue);
        if (isNaN(targetIdx)) {
            searchError = "Invalid number";
            return;
        }

        // Search in unfiltered metadata to allow finding any component
        const fullMetadata = activationContextsSummary[selectedLayer];
        const component = fullMetadata.find((m) => m.subcomponent_idx === targetIdx);

        if (!component) {
            searchError = `Not found`;
            return;
        }

        // Find the page index in filtered list
        const pageIndex = currentLayerMetadata.findIndex((m) => m.subcomponent_idx === targetIdx);

        if (pageIndex === -1) {
            searchError = `Below cutoff (${component.mean_ci.toExponential(2)})`;
            return;
        }

        currentPage = pageIndex;
        loadCurrentComponent();
    }

    function previousPage() {
        if (currentPage > 0) {
            currentPage--;
            loadCurrentComponent();
        }
    }

    function nextPage() {
        if (currentPage < totalPages - 1) {
            currentPage++;
            loadCurrentComponent();
        }
    }

    function handleLayerChange(event: Event) {
        const target = event.target as HTMLSelectElement;
        selectedLayer = target.value;
        currentPage = 0;
        loadCurrentComponent();
    }

    // Derive token lists from loaded tokenStats (null if not loaded or no data)
    const inputTokenLists = $derived.by(() => {
        const ts = componentData.tokenStats;
        if (ts.status !== "loaded" || ts.data === null) return null;
        return [
            // TODO clean this up, but for now Top Recall is honestly not useful
            // {
            //     title: "Top Recall",
            //     mathNotation: "P(token | component fires)",
            //     items: ts.data.input.top_recall
            //         .slice(0, COMPONENT_CARD_CONSTANTS.N_INPUT_TOKENS)
            //         .map(([token, value]) => ({ token, value })),
            //     maxScale: 1,
            // },
            {
                title: "Top Precision",
                mathNotation: "P(component fires | token)",
                items: ts.data.input.top_precision.map(([token, value]) => ({ token, value })),
                maxScale: 1,
            },
        ];
    });

    const outputTokenLists = $derived.by(() => {
        const ts = componentData.tokenStats;
        if (ts.status !== "loaded" || ts.data === null) return null;
        // Compute max absolute PMI for scaling
        const maxAbsPmi = Math.max(
            ts.data.output.top_pmi[0]?.[1] ?? 0,
            Math.abs(ts.data.output.bottom_pmi?.[0]?.[1] ?? 0),
        );
        return [
            {
                title: "Top PMI",
                mathNotation: "positive association with predictions",
                items: ts.data.output.top_pmi.map(([token, value]) => ({ token, value })),
                maxScale: maxAbsPmi,
            },
            {
                title: "Bottom PMI",
                mathNotation: "negative association with predictions",
                items: ts.data.output.bottom_pmi.map(([token, value]) => ({ token, value })),
                maxScale: maxAbsPmi,
            },
        ];
    });

    // Format mean CI for display
    function formatMeanCi(ci: number): string {
        return ci < 0.001 ? ci.toExponential(2) : ci.toFixed(3);
    }

    function handlePlotSelect(subcomponentIdx: number) {
        const pageIndex = currentLayerMetadata.findIndex((m) => m.subcomponent_idx === subcomponentIdx);
        if (pageIndex === -1) return;
        currentPage = pageIndex;
        loadCurrentComponent();
    }

    // Compute global max absolute component act for normalization (used by both activating examples and probe)
    const maxAbsComponentAct = $derived.by(() => {
        if (componentData.componentDetail.status !== "loaded") return 1;
        return computeMaxAbsComponentAct(componentData.componentDetail.data.example_component_acts);
    });

    const activationExamples = $derived(
        mapLoadable(
            componentData.componentDetail,
            (d): ActivationExamplesData => ({
                tokens: d.example_tokens,
                ci: d.example_ci,
                componentActs: d.example_component_acts,
                maxAbsComponentAct,
            }),
        ),
    );
</script>

<div class="viewer-content">
    <div class="controls-row">
        <div class="layer-select">
            <label for="layer-select">Layer:</label>
            <select id="layer-select" value={selectedLayer} onchange={handleLayerChange}>
                {#each availableLayers as layer (layer)}
                    <option value={layer}>{layer}</option>
                {/each}
            </select>
        </div>

        <div class="pagination">
            <label for="page-input">Subcomponent:</label>
            <button onclick={previousPage} disabled={currentPage === 0}>&lt;</button>
            <input
                id="page-input"
                type="number"
                min="1"
                max={totalPages}
                value={currentPage + 1}
                oninput={handlePageInput}
                class="page-input"
            />
            <span>of {totalPages}</span>
            <button onclick={nextPage} disabled={currentPage === totalPages - 1}>&gt;</button>
        </div>

        <div class="search-box">
            <label for="search-input">Go to index:</label>
            <input
                id="search-input"
                type="number"
                placeholder="e.g. 42"
                value={searchValue}
                oninput={handleSearchInput}
                class="search-input"
            />
            {#if searchError}
                <span class="search-error">{searchError}</span>
            {/if}
        </div>
    </div>

    <ComponentFrequencyCurve
        metadata={activationContextsSummary[selectedLayer]}
        currentSubcomponentIdx={currentMetadata?.subcomponent_idx ?? null}
        onSelect={handlePlotSelect}
    />

    <div class="component-section">
        <SectionHeader title="Subcomponent {currentMetadata.subcomponent_idx}" level="h4">
            <span class="mean-ci">Mean CI: {formatMeanCi(currentMetadata.mean_ci)}</span>
            {#if currentIntruderScore !== null}
                <span class="mean-ci">Intruder: {Math.round(currentIntruderScore * 100)}%</span>
            {/if}
        </SectionHeader>

        <div class="interpretation-badges">
            <InterpretationBadge
                interpretation={componentData.interpretation}
                interpretationDetail={componentData.interpretationDetail}
                onGenerate={componentData.generateInterpretation}
            />
            {#if currentGraphInterpLabel && componentData.graphInterpDetail.status === "loaded" && componentData.graphInterpDetail.data}
                <GraphInterpBadge headline={currentGraphInterpLabel} detail={componentData.graphInterpDetail.data} />
            {/if}
        </div>

        <!-- Activation examples -->
        {#if activationExamples.status === "error"}
            <StatusText>Error loading component data: {String(activationExamples.error)}</StatusText>
        {:else}
            <ActivationContextsPagedTable data={activationExamples} />
        {/if}

        <ComponentProbeInput
            layer={selectedLayer}
            componentIdx={currentMetadata.subcomponent_idx}
            {maxAbsComponentAct}
        />

        <!-- Dataset attributions -->
        {#if componentData.datasetAttributions?.status === "loaded" && componentData.datasetAttributions.data}
            <DatasetAttributionsSection attributions={componentData.datasetAttributions.data} />
        {:else if componentData.datasetAttributions?.status === "loading"}
            <div class="dataset-attributions-loading">
                <SectionHeader title="Dataset Attributions" />
                <StatusText>Loading...</StatusText>
            </div>
        {:else if componentData.datasetAttributions?.status === "error"}
            <div class="dataset-attributions-loading">
                <SectionHeader title="Dataset Attributions" />
                <StatusText>Error: {String(componentData.datasetAttributions.error)}</StatusText>
            </div>
        {/if}

        <div class="token-stats-row">
            {#if componentData.tokenStats.status === "uninitialized" || componentData.tokenStats.status === "loading"}
                <StatusText>Loading token stats...</StatusText>
            {:else if componentData.tokenStats.status === "error"}
                <StatusText>Error: {String(componentData.tokenStats.error)}</StatusText>
            {:else}
                <TokenStatsSection
                    sectionTitle="Input Tokens"
                    sectionSubtitle="(what activates this component)"
                    lists={inputTokenLists}
                />

                <TokenStatsSection
                    sectionTitle="Output Tokens"
                    sectionSubtitle="(what this component predicts)"
                    lists={outputTokenLists}
                />
            {/if}
        </div>

        <!-- Component correlations -->
        {#if anyCorrelationStatsEnabled()}
            <div class="correlations-section">
                <SectionHeader title="Correlated Components" />
                {#if componentData.correlations.status === "uninitialized" || componentData.correlations.status === "loading"}
                    <StatusText>Loading...</StatusText>
                {:else if componentData.correlations.status === "error"}
                    <StatusText>Error loading correlations: {String(componentData.correlations.error)}</StatusText>
                {:else if componentData.correlations.data === null}
                    <StatusText>No correlations data. Run harvest pipeline first.</StatusText>
                {:else}
                    <ComponentCorrelationMetrics
                        correlations={componentData.correlations.data}
                        pageSize={COMPONENT_CARD_CONSTANTS.CORRELATIONS_PAGE_SIZE}
                    />
                {/if}
            </div>
        {/if}
    </div>
</div>

<style>
    .viewer-content {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
    }

    .controls-row {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        flex-wrap: wrap;
    }

    .layer-select {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .viewer-content label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
    }

    #layer-select {
        border: 1px solid var(--border-default);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        cursor: pointer;
        min-width: 180px;
    }

    #layer-select:focus {
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
        background: var(--bg-surface);
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

    .search-box {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .search-input {
        width: 80px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        appearance: textfield;
    }

    .search-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .search-input::-webkit-inner-spin-button,
    .search-input::-webkit-outer-spin-button {
        appearance: none;
        margin: 0;
    }

    .search-input::placeholder {
        color: var(--text-muted);
    }

    .search-error {
        font-size: var(--text-xs);
        color: var(--semantic-error);
        white-space: nowrap;
    }

    .page-input {
        width: 50px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
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

    .component-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        padding: var(--space-4);
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
    }

    .mean-ci {
        font-weight: 400;
        color: var(--text-muted);
        font-family: var(--font-mono);
        margin-left: var(--space-2);
    }

    .token-stats-row {
        display: flex;
        gap: var(--space-4);
    }

    .token-stats-row > :global(*) {
        flex: 1;
        min-width: 0;
    }

    .correlations-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .interpretation-badges {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .dataset-attributions-loading {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }
</style>
