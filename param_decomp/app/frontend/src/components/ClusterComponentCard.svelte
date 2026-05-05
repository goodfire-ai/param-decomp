<script lang="ts">
    import { getContext, onMount } from "svelte";
    import { computeMaxAbsComponentAct } from "../lib/colors";
    import { mapLoadable } from "../lib/index";
    import { anyCorrelationStatsEnabled } from "../lib/displaySettings.svelte";
    import { useComponentDataExpectCached } from "../lib/useComponentDataExpectCached.svelte";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import ActivationContextsPagedTable, { type ActivationExamplesData } from "./ActivationContextsPagedTable.svelte";
    import ComponentProbeInput from "./ComponentProbeInput.svelte";
    import ComponentCorrelationMetrics from "./ui/ComponentCorrelationMetrics.svelte";
    import DatasetAttributionsSection from "./ui/DatasetAttributionsSection.svelte";
    import InterpretationBadge from "./ui/InterpretationBadge.svelte";
    import SectionHeader from "./ui/SectionHeader.svelte";
    import StatusText from "./ui/StatusText.svelte";
    import TokenStatsSection from "./ui/TokenStatsSection.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    type Props = {
        layer: string;
        cIdx: number;
    };

    let { layer, cIdx }: Props = $props();

    const intruderScore = $derived(runState.getIntruderScore(`${layer}:${cIdx}`));

    const componentData = useComponentDataExpectCached();

    onMount(() => {
        componentData.load(layer, cIdx);
    });

    const inputTokenLists = $derived.by(() => {
        const tokenStats = componentData.tokenStats;
        if (tokenStats.status !== "loaded" || tokenStats.data === null) return null;
        return [
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

    function formatNumericalValue(val: number): string {
        return Math.abs(val) < 0.001 ? val.toExponential(2) : val.toFixed(3);
    }

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

<div class="component-node-card">
    <div class="card-header">
        <h3 class="node-identifier">{layer}:{cIdx}</h3>
        <div class="header-metrics">
            {#if componentData.componentDetail.status === "loaded"}
                <span class="metric">Mean CI: {formatNumericalValue(componentData.componentDetail.data.mean_ci)}</span>
            {/if}
            {#if intruderScore !== null}
                <span class="metric">Intruder: {Math.round(intruderScore * 100)}%</span>
            {/if}
        </div>
    </div>

    <InterpretationBadge
        interpretation={componentData.interpretation}
        interpretationDetail={componentData.interpretationDetail}
        onGenerate={componentData.generateInterpretation}
    />

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

    {#if componentData.datasetAttributions.status === "uninitialized"}
        <StatusText>uninitialized</StatusText>
    {:else if componentData.datasetAttributions.status === "loaded"}
        {#if componentData.datasetAttributions.data !== null}
            <DatasetAttributionsSection attributions={componentData.datasetAttributions.data} />
        {:else}
            <StatusText>No dataset attributions available.</StatusText>
        {/if}
    {:else if componentData.datasetAttributions.status === "loading"}
        <div class="dataset-attributions-loading">
            <SectionHeader title="Dataset Attributions" />
            <StatusText>Loading...</StatusText>
        </div>
    {:else if componentData.datasetAttributions.status === "error"}
        <div class="dataset-attributions-loading">
            <SectionHeader title="Dataset Attributions" />
            <StatusText>Error: {String(componentData.datasetAttributions.error)}</StatusText>
        </div>
    {/if}

    <div class="token-stats-section">
        <SectionHeader title="Token Statistics" />
        <div class="token-stats-row">
            {#if componentData.tokenStats.status === "loading" || componentData.tokenStats.status === "uninitialized"}
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
    </div>

    {#if anyCorrelationStatsEnabled()}
        <div class="correlations-section">
            <SectionHeader title="Correlated Components" />
            {#if componentData.correlations.status === "loading"}
                <StatusText>Loading...</StatusText>
            {:else if componentData.correlations.status === "loaded" && componentData.correlations.data}
                <ComponentCorrelationMetrics correlations={componentData.correlations.data} pageSize={16} />
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
</style>
