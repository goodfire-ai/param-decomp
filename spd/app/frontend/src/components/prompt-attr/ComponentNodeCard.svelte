<script lang="ts">
    import { getContext, onMount } from "svelte";
    import { computeMaxAbsComponentAct } from "../../lib/colors";
    import { mapLoadable } from "../../lib/index";
    import type { ActivationExamplesData } from "../ActivationContextsPagedTable.svelte";
    import { COMPONENT_CARD_CONSTANTS } from "../../lib/componentCardConstants";
    import { anyCorrelationStatsEnabled, displaySettings } from "../../lib/displaySettings.svelte";
    import { topEdgeAttributions, type EdgeData, type OutputProbability } from "../../lib/promptAttributionsTypes";
    import { useComponentDataExpectCached } from "../../lib/useComponentDataExpectCached.svelte";
    import { RUN_KEY, type RunContext } from "../../lib/useRun.svelte";
    import ActivationContextsPagedTable from "../ActivationContextsPagedTable.svelte";
    import ComponentProbeInput from "../ComponentProbeInput.svelte";
    import ComponentCorrelationMetrics from "../ui/ComponentCorrelationMetrics.svelte";
    import DatasetAttributionsSection from "../ui/DatasetAttributionsSection.svelte";
    import EdgeAttributionGrid from "../ui/EdgeAttributionGrid.svelte";
    import GraphInterpBadge from "../ui/GraphInterpBadge.svelte";
    import InterpretationBadge from "../ui/InterpretationBadge.svelte";
    import SectionHeader from "../ui/SectionHeader.svelte";
    import StatusText from "../ui/StatusText.svelte";
    import TokenStatsSection from "../ui/TokenStatsSection.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    type Props = {
        layer: string;
        cIdx: number;
        seqIdx: number;
        ciVal: number | null;
        subcompAct: number | null;
        token: string;
        edgesBySource: Map<string, EdgeData[]>;
        edgesByTarget: Map<string, EdgeData[]>;
        tokens: string[];
        outputProbs: Record<string, OutputProbability>;
        onPinComponent?: (layer: string, cIdx: number, seqIdx: number) => void;
    };

    let {
        layer,
        cIdx,
        seqIdx,
        ciVal,
        subcompAct,
        token,
        edgesBySource,
        edgesByTarget,
        tokens,
        outputProbs,
        onPinComponent,
    }: Props = $props();

    const clusterId = $derived(runState.clusterMapping?.data[`${layer}:${cIdx}`]);
    const intruderScore = $derived(runState.getIntruderScore(`${layer}:${cIdx}`));
    const graphInterpLabel = $derived(runState.getGraphInterpLabel(`${layer}:${cIdx}`));

    // Handle clicking a correlated component - parse key and pin it at same seqIdx
    function handleCorrelationClick(componentKey: string) {
        if (!onPinComponent) return;
        // componentKey format: "layer:cIdx" e.g. "h.0.attn.q_proj:5"
        const [clickedLayer, clickedCIdx] = componentKey.split(":");
        onPinComponent(clickedLayer, parseInt(clickedCIdx), seqIdx);
    }

    // Component data hook - call load() explicitly on mount.
    // Parents use {#key} or {#each} keys to remount this component when layer/cIdx change,
    // so we only need to load once on mount (no effect watching props).
    // Reads from prefetched cache for activation contexts, correlations, token stats.
    // Dataset attributions and interpretation details are fetched on-demand.
    const componentData = useComponentDataExpectCached();

    onMount(() => {
        componentData.load(layer, cIdx);
    });

    // Derive token lists from loaded tokenStats (null if not loaded or no data)
    const inputTokenLists = $derived.by(() => {
        const tokenStats = componentData.tokenStats;
        if (tokenStats.status !== "loaded" || tokenStats.data === null) return null;
        return [
            // TODO clean this up, but for now Top Recall is honestly not useful
            // {
            //     title: "Top Recall",
            //     mathNotation: "P(token | component fires)",
            //     items: tokenStats.data.input.top_recall
            //         .slice(0, COMPONENT_CARD_CONSTANTS.N_INPUT_TOKENS)
            //         .map(([token, value]) => ({ token, value })),
            //     maxScale: 1,
            // },
            {
                title: "Top Precision",
                mathNotation: "P(component fires | token)",
                items: tokenStats.data.input.top_precision.map(([token, value]) => ({
                    token,
                    value,
                })),
                maxScale: 1,
            },
        ];
    });

    const outputTokenLists = $derived.by(() => {
        const tokenStats = componentData.tokenStats;
        if (tokenStats.status !== "loaded" || tokenStats.data === null) return null;
        // Compute max absolute PMI for scaling
        const maxAbsPmi = Math.max(
            tokenStats.data.output.top_pmi[0]?.[1] ?? 0,
            Math.abs(tokenStats.data.output.bottom_pmi?.[0]?.[1] ?? 0),
        );
        return [
            {
                title: "Top PMI",
                mathNotation: "positive association with predictions",
                items: tokenStats.data.output.top_pmi.map(([token, value]) => ({ token, value })),
                maxScale: maxAbsPmi,
            },
            {
                title: "Bottom PMI",
                mathNotation: "negative association with predictions",
                items: tokenStats.data.output.bottom_pmi.map(([token, value]) => ({
                    token,
                    value,
                })),
                maxScale: maxAbsPmi,
            },
        ];
    });

    // Format mean CI or subcomponent activation for display
    function formatNumericalValue(val: number): string {
        return Math.abs(val) < 0.001 ? val.toExponential(2) : val.toFixed(3);
    }

    // === Edge attribution lists ===
    const currentNodeKey = $derived(`${layer}:${seqIdx}:${cIdx}`);
    const N_EDGES_TO_DISPLAY = 20;

    function resolveTokenStr(nodeKey: string): string | null {
        const parts = nodeKey.split(":");
        if (parts.length !== 3) return null;
        const [layer, seqStr, cIdx] = parts;
        const seqIdx = parseInt(seqStr);
        if (layer === "embed") return tokens[seqIdx] ?? null;
        if (layer === "output") return outputProbs[`${seqIdx}:${cIdx}`]?.token ?? null;
        return null;
    }

    const incoming = $derived(
        topEdgeAttributions(edgesByTarget.get(currentNodeKey) ?? [], (e) => e.src, N_EDGES_TO_DISPLAY, resolveTokenStr),
    );

    const outgoing = $derived(
        topEdgeAttributions(edgesBySource.get(currentNodeKey) ?? [], (e) => e.tgt, N_EDGES_TO_DISPLAY, resolveTokenStr),
    );

    const hasAnyEdges = $derived(incoming.length > 0 || outgoing.length > 0);

    // Handle clicking an edge node - parse key and pin it
    function handleEdgeNodeClick(nodeKey: string) {
        if (!onPinComponent) return;
        // nodeKey format: "layer:seq:cIdx"
        const [clickedLayer, clickedSeqIdx, clickedCIdx] = nodeKey.split(":");
        onPinComponent(clickedLayer, parseInt(clickedCIdx), parseInt(clickedSeqIdx));
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
                maxAbsComponentAct: computeMaxAbsComponentAct(d.example_component_acts),
            }),
        ),
    );
</script>

<div class="component-node-card">
    <div class="card-header">
        <h3 class="node-identifier">{layer}:{seqIdx}:{cIdx}</h3>
        <div class="token-display">"{token}"</div>
        <div class="header-metrics">
            {#if ciVal !== null}
                <span class="metric">CI: {formatNumericalValue(ciVal)}</span>
            {/if}
            {#if subcompAct !== null}
                <span class="metric">Subcomp Act: {formatNumericalValue(subcompAct)}</span>
            {/if}
            {#if clusterId !== undefined}
                <span class="metric">Cluster: {clusterId ?? "null"}</span>
            {/if}
            {#if componentData.componentDetail.status === "loaded"}
                <span class="metric">Mean CI: {formatNumericalValue(componentData.componentDetail.data.mean_ci)}</span>
            {/if}
            {#if intruderScore !== null}
                <span class="metric">Intruder: {Math.round(intruderScore * 100)}%</span>
            {/if}
        </div>
    </div>

    <div class="interpretation-badges">
        <InterpretationBadge
            interpretation={componentData.interpretation}
            interpretationDetail={componentData.interpretationDetail}
            onGenerate={componentData.generateInterpretation}
        />
        {#if graphInterpLabel && componentData.graphInterpDetail.status === "loaded" && componentData.graphInterpDetail.data}
            <GraphInterpBadge headline={graphInterpLabel} detail={componentData.graphInterpDetail.data} />
        {/if}
    </div>

    <!-- Activating examples (from harvest data) -->
    <div class="activating-examples-section">
        <SectionHeader title="Activating Examples" />
        {#if activationExamples.status === "error"}
            <StatusText>Error loading details: {String(activationExamples.error)}</StatusText>
        {:else if activationExamples.status === "loaded" && activationExamples.data.tokens.length === 0}
            <!-- no examples -->
        {:else}
            <ActivationContextsPagedTable data={activationExamples} />
        {/if}
    </div>

    <ComponentProbeInput {layer} componentIdx={cIdx} {maxAbsComponentAct} />

    <!-- Prompt attributions -->
    {#if displaySettings.showEdgeAttributions && hasAnyEdges}
        <EdgeAttributionGrid
            title="Prompt Attributions"
            incomingLabel="Incoming"
            outgoingLabel="Outgoing"
            {incoming}
            {outgoing}
            pageSize={COMPONENT_CARD_CONSTANTS.PROMPT_ATTRIBUTIONS_PAGE_SIZE}
            onClick={handleEdgeNodeClick}
        />
    {/if}

    <!-- Dataset attributions  -->
    {#if componentData.datasetAttributions.status === "loading" || componentData.datasetAttributions.status === "uninitialized"}
        <div class="dataset-attributions-loading">
            <SectionHeader title="Dataset Attributions" />
            <div class="skeleton-group">
                <div class="skeleton skeleton-line wide"></div>
                <div class="skeleton skeleton-line medium"></div>
            </div>
        </div>
    {:else if componentData.datasetAttributions.status === "loaded"}
        {#if componentData.datasetAttributions.data !== null}
            <DatasetAttributionsSection
                attributions={componentData.datasetAttributions.data}
                onComponentClick={handleCorrelationClick}
            />
        {/if}
    {:else if componentData.datasetAttributions.status === "error"}
        <div class="dataset-attributions-loading">
            <SectionHeader title="Dataset Attributions" />
            <StatusText>Error: {String(componentData.datasetAttributions.error)}</StatusText>
        </div>
    {/if}

    <div class="token-stats-section">
        <SectionHeader title="Token Statistics" />
        <div class="token-stats-row">
            {#if componentData.tokenStats === null || componentData.tokenStats.status === "loading"}
                <div class="skeleton-group">
                    <div class="skeleton skeleton-line medium"></div>
                    <div class="skeleton skeleton-line short"></div>
                    <div class="skeleton skeleton-line medium"></div>
                    <div class="skeleton skeleton-line short"></div>
                </div>
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
    </div>

    <!-- Component correlations -->
    {#if anyCorrelationStatsEnabled()}
        <div class="correlations-section">
            <SectionHeader title="Correlated Components" />
            {#if componentData.correlations.status === "loading"}
                <div class="skeleton-group">
                    <div class="skeleton skeleton-line wide"></div>
                    <div class="skeleton skeleton-line medium"></div>
                </div>
            {:else if componentData.correlations.status === "loaded" && componentData.correlations.data}
                <ComponentCorrelationMetrics
                    correlations={componentData.correlations.data}
                    pageSize={16}
                    onComponentClick={handleCorrelationClick}
                />
            {:else if componentData.correlations.status === "error"}
                <StatusText>Error loading correlations: {String(componentData.correlations.error)}</StatusText>
            {:else}
                <StatusText>No correlations available.</StatusText>
            {/if}
        </div>
    {/if}
</div>

<style>
    .component-node-card {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        font-family: var(--font-sans);
        color: var(--text-primary);
    }

    .interpretation-badges {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .card-header {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-default);
    }

    .node-identifier {
        font-size: var(--text-base);
        font-family: var(--font-mono);
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }

    .token-display {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .header-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-3);
    }

    .metric {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
        font-weight: 600;
    }

    .token-stats-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
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

    .dataset-attributions-loading {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .skeleton-group {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .skeleton-line {
        height: 12px;
        border-radius: var(--radius-sm);
        background: var(--border-default);
        opacity: 0.4;
        animation: skeleton-pulse 1.2s ease-in-out infinite;
    }

    .skeleton-line.wide {
        width: 100%;
    }

    .skeleton-line.medium {
        width: 65%;
    }

    .skeleton-line.short {
        width: 40%;
    }

    @keyframes skeleton-pulse {
        0%,
        100% {
            opacity: 0.4;
        }
        50% {
            opacity: 0.15;
        }
    }
</style>
