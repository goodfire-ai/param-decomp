<script lang="ts">
    import { onMount } from "svelte";
    import { SvelteMap } from "svelte/reactivity";
    import type { Loadable } from "../lib";
    import { computeMaxAbsComponentAct } from "../lib/colors";
    import { mapLoadable } from "../lib/index";
    import { COMPONENT_CARD_CONSTANTS } from "../lib/componentCardConstants";
    import { anyCorrelationStatsEnabled, displaySettings } from "../lib/displaySettings.svelte";
    import type { ActivationContextsSummary, SubcomponentMetadata } from "../lib/promptAttributionsTypes";
    import { useComponentData } from "../lib/useComponentData.svelte";
    import {
        getSubrunInterpretationDetail,
        type CompareInterpretationDetail,
        type CompareInterpretationHeadline,
        type SubrunSummary,
    } from "../lib/api";
    import ActivationContextsPagedTable, { type ActivationExamplesData } from "./ActivationContextsPagedTable.svelte";
    import ComponentProbeInput from "./ComponentProbeInput.svelte";
    import ComponentCorrelationMetrics from "./ui/ComponentCorrelationMetrics.svelte";
    import ComponentFrequencyCurve from "./ui/ComponentFrequencyCurve.svelte";
    import SectionHeader from "./ui/SectionHeader.svelte";
    import StatusText from "./ui/StatusText.svelte";
    import SubrunInterpCard from "./ui/SubrunInterpCard.svelte";
    import TokenStatsSection from "./ui/TokenStatsSection.svelte";
    import DatasetAttributionsSection from "./ui/DatasetAttributionsSection.svelte";

    interface Props {
        activationContextsSummary: ActivationContextsSummary;
        selectedSubruns: SubrunSummary[];
        headlinesCache: Map<string, Record<string, CompareInterpretationHeadline>>;
    }

    let { activationContextsSummary, selectedSubruns, headlinesCache }: Props = $props();

    let availableLayers = $derived(Object.keys(activationContextsSummary).sort());
    let currentPage = $state(0);
    let selectedLayer = $state<string>(Object.keys(activationContextsSummary)[0]);

    let currentLayerMetadata = $derived(
        activationContextsSummary[selectedLayer].filter((m) => m.mean_ci >= displaySettings.meanCiCutoff),
    );
    let totalPages = $derived(currentLayerMetadata.length);
    let currentMetadata = $derived<SubcomponentMetadata | undefined>(currentLayerMetadata[currentPage]);

    // Component data hook for right panel
    const componentData = useComponentData();
    const DEBOUNCE_MS = 250;
    let loadTimeout: ReturnType<typeof setTimeout> | null = null;

    // Detail cache: subrunId → (componentKey → Loadable<detail>)
    let detailCache = new SvelteMap<string, SvelteMap<string, Loadable<CompareInterpretationDetail | null>>>();

    function currentComponentKey(): string | null {
        if (!currentMetadata) return null;
        return `${selectedLayer}:${currentMetadata.subcomponent_idx}`;
    }

    function loadCurrentComponent() {
        if (loadTimeout) clearTimeout(loadTimeout);
        componentData.reset();
        loadTimeout = setTimeout(() => {
            if (!currentMetadata) return;
            componentData.load(selectedLayer, currentMetadata.subcomponent_idx);
            loadDetails();
        }, DEBOUNCE_MS);
    }

    function loadDetails() {
        const key = currentComponentKey();
        if (!key || !currentMetadata) return;
        for (const subrun of selectedSubruns) {
            let subrunDetails = detailCache.get(subrun.subrun_id);
            if (!subrunDetails) {
                subrunDetails = new SvelteMap();
                detailCache.set(subrun.subrun_id, subrunDetails);
            }
            if (subrunDetails.has(key)) continue;
            subrunDetails.set(key, { status: "loading" });

            getSubrunInterpretationDetail(subrun.subrun_id, selectedLayer, currentMetadata.subcomponent_idx).then(
                (data) => {
                    detailCache.get(subrun.subrun_id)?.set(key!, { status: "loaded", data });
                },
                (error) => {
                    detailCache.get(subrun.subrun_id)?.set(key!, { status: "error", error });
                },
            );
        }
    }

    onMount(() => {
        loadCurrentComponent();
        return () => {
            if (loadTimeout) clearTimeout(loadTimeout);
        };
    });

    $effect(() => {
        if (currentPage >= totalPages && totalPages > 0) {
            currentPage = 0;
            loadCurrentComponent();
        }
    });

    // Reload details when selectedSubruns change
    $effect(() => {
        // read selectedSubruns to establish reactive dependency
        void selectedSubruns.length;
        if (currentMetadata) {
            loadDetails();
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

        const fullMetadata = activationContextsSummary[selectedLayer];
        const component = fullMetadata.find((m) => m.subcomponent_idx === targetIdx);
        if (!component) {
            searchError = "Not found";
            return;
        }

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
        selectedLayer = (event.target as HTMLSelectElement).value;
        currentPage = 0;
        loadCurrentComponent();
    }

    function getHeadline(subrunId: string): CompareInterpretationHeadline | null {
        const key = currentComponentKey();
        if (!key) return null;
        const headlines = headlinesCache.get(subrunId);
        if (!headlines) return null;
        return headlines[key] ?? null;
    }

    function getDetail(subrunId: string): Loadable<CompareInterpretationDetail | null> {
        const key = currentComponentKey();
        if (!key) return { status: "uninitialized" };
        return detailCache.get(subrunId)?.get(key) ?? { status: "uninitialized" };
    }

    function formatMeanCi(ci: number): string {
        return ci < 0.001 ? ci.toExponential(2) : ci.toFixed(3);
    }

    function handlePlotSelect(subcomponentIdx: number) {
        const pageIndex = currentLayerMetadata.findIndex((m) => m.subcomponent_idx === subcomponentIdx);
        if (pageIndex === -1) return;
        currentPage = pageIndex;
        loadCurrentComponent();
    }

    // Activation examples data
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

    // Token stats
    const inputTokenLists = $derived.by(() => {
        const ts = componentData.tokenStats;
        if (ts.status !== "loaded" || ts.data === null) return null;
        return [
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
</script>

{#if currentMetadata}
    <div class="comparer">
        <div class="controls-row">
            <div class="layer-select">
                <label for="compare-layer-select">Layer:</label>
                <select id="compare-layer-select" value={selectedLayer} onchange={handleLayerChange}>
                    {#each availableLayers as layer (layer)}
                        <option value={layer}>{layer}</option>
                    {/each}
                </select>
            </div>

            <div class="pagination">
                <label for="compare-page-input">Subcomponent:</label>
                <button onclick={previousPage} disabled={currentPage === 0}>&lt;</button>
                <input
                    id="compare-page-input"
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
                <label for="compare-search-input">Go to index:</label>
                <input
                    id="compare-search-input"
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

        <div class="two-panel">
            <div class="left-panel">
                <SectionHeader title="Subcomponent {currentMetadata.subcomponent_idx}" level="h4">
                    <span class="mean-ci">Mean CI: {formatMeanCi(currentMetadata.mean_ci)}</span>
                </SectionHeader>

                <div class="interp-cards">
                    {#each selectedSubruns as subrun (subrun.subrun_id)}
                        <SubrunInterpCard
                            subrunId={subrun.subrun_id}
                            strategy={subrun.strategy}
                            llmModel={subrun.llm_model}
                            note={subrun.note}
                            headline={getHeadline(subrun.subrun_id)}
                            detail={getDetail(subrun.subrun_id)}
                        />
                    {/each}
                </div>
            </div>

            <div class="right-panel">
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

                {#if componentData.datasetAttributions?.status === "loaded" && componentData.datasetAttributions.data}
                    <DatasetAttributionsSection attributions={componentData.datasetAttributions.data} />
                {:else if componentData.datasetAttributions?.status === "loading"}
                    <div class="section-loading">
                        <SectionHeader title="Dataset Attributions" />
                        <StatusText>Loading...</StatusText>
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

                {#if anyCorrelationStatsEnabled()}
                    <div class="correlations-section">
                        <SectionHeader title="Correlated Components" />
                        {#if componentData.correlations.status === "uninitialized" || componentData.correlations.status === "loading"}
                            <StatusText>Loading...</StatusText>
                        {:else if componentData.correlations.status === "error"}
                            <StatusText>Error loading correlations</StatusText>
                        {:else if componentData.correlations.data === null}
                            <StatusText>No correlations data.</StatusText>
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
    </div>
{:else}
    <StatusText>No components above CI cutoff</StatusText>
{/if}

<style>
    .comparer {
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

    .comparer label {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
    }

    #compare-layer-select {
        border: 1px solid var(--border-default);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-elevated);
        color: var(--text-primary);
        cursor: pointer;
        min-width: 180px;
    }

    #compare-layer-select:focus {
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

    .two-panel {
        display: flex;
        gap: var(--space-4);
        min-height: 0;
    }

    .left-panel {
        flex: 2;
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        overflow-y: auto;
    }

    .right-panel {
        flex: 3;
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        overflow-y: auto;
    }

    .interp-cards {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
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

    .section-loading {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }
</style>
