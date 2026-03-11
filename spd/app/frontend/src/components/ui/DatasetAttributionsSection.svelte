<script lang="ts">
    /**
     * Dataset attributions for a single component.
     *
     * Terminology:
     * - "Incoming" = sources that attribute TO this component (this component is the target)
     * - "Outgoing" = targets that this component attributes TO (this component is the source)
     */
    import { COMPONENT_CARD_CONSTANTS } from "../../lib/componentCardConstants";
    import type { EdgeAttribution } from "../../lib/promptAttributionsTypes";
    import type { DatasetAttributions } from "../../lib/useComponentData.svelte";
    import type { AttrMetric, DatasetAttributionEntry } from "../../lib/api/datasetAttributions";
    import EdgeAttributionList from "./EdgeAttributionList.svelte";
    import SectionHeader from "./SectionHeader.svelte";

    type Props = {
        attributions: DatasetAttributions;
        onComponentClick?: (componentKey: string) => void;
    };

    let { attributions, onComponentClick }: Props = $props();
    let selectedMetric = $state<AttrMetric>("attr_abs");

    function handleClick(key: string) {
        if (onComponentClick) {
            onComponentClick(key);
        }
    }

    function toEdgeAttribution(entries: DatasetAttributionEntry[], maxAbsValue: number): EdgeAttribution[] {
        return entries.map((e) => ({
            key: e.component_key,
            value: e.value,
            normalizedMagnitude: Math.abs(e.value) / (maxAbsValue || 1),
            tokenStr: e.token_str,
        }));
    }

    function maxAbs(...vals: number[]): number {
        return Math.max(...vals.map(Math.abs));
    }

    const selected = $derived(attributions[selectedMetric]);

    const maxSource = $derived(
        maxAbs(selected.positive_sources[0]?.value ?? 0, selected.negative_sources[0]?.value ?? 0),
    );
    const maxTarget = $derived(
        maxAbs(selected.positive_targets[0]?.value ?? 0, selected.negative_targets[0]?.value ?? 0),
    );

    const incomingPositive = $derived(toEdgeAttribution(selected.positive_sources, maxSource));
    const incomingNegative = $derived(toEdgeAttribution(selected.negative_sources, maxSource));
    const outgoingPositive = $derived(toEdgeAttribution(selected.positive_targets, maxTarget));
    const outgoingNegative = $derived(toEdgeAttribution(selected.negative_targets, maxTarget));

    const hasAnyIncoming = $derived(incomingPositive.length > 0 || incomingNegative.length > 0);
    const hasAnyOutgoing = $derived(outgoingPositive.length > 0 || outgoingNegative.length > 0);

    const pageSize = COMPONENT_CARD_CONSTANTS.DATASET_ATTRIBUTIONS_PAGE_SIZE;
</script>

<div class="section">
    <div class="metric-selector">
        <label class="radio-item">
            <input
                type="radio"
                name="dataset-attr-metric"
                checked={selectedMetric === "attr_abs"}
                onchange={() => (selectedMetric = "attr_abs")}
            />
            <span class="stat-label">Abs Target</span>
        </label>
        <label class="radio-item">
            <input
                type="radio"
                name="dataset-attr-metric"
                checked={selectedMetric === "attr"}
                onchange={() => (selectedMetric = "attr")}
            />
            <span class="stat-label">Signed</span>
        </label>
    </div>

    {#if hasAnyIncoming}
        <div class="edge-list-group">
            <SectionHeader title="Dataset Attributions – Incoming" />
            <div class="pos-neg-row">
                {#if incomingPositive.length > 0}
                    <div class="edge-list">
                        <EdgeAttributionList
                            items={incomingPositive}
                            {pageSize}
                            onClick={handleClick}
                            direction="positive"
                            title="Positive"
                        />
                    </div>
                {/if}
                {#if incomingNegative.length > 0}
                    <div class="edge-list">
                        <EdgeAttributionList
                            items={incomingNegative}
                            {pageSize}
                            onClick={handleClick}
                            direction="negative"
                            title="Negative"
                        />
                    </div>
                {/if}
            </div>
        </div>
    {/if}

    {#if hasAnyOutgoing}
        <div class="edge-list-group">
            <SectionHeader title="Dataset Attributions – Outgoing" />
            <div class="pos-neg-row">
                {#if outgoingPositive.length > 0}
                    <div class="edge-list">
                        <EdgeAttributionList
                            items={outgoingPositive}
                            {pageSize}
                            onClick={handleClick}
                            direction="positive"
                            title="Positive"
                        />
                    </div>
                {/if}
                {#if outgoingNegative.length > 0}
                    <div class="edge-list">
                        <EdgeAttributionList
                            items={outgoingNegative}
                            {pageSize}
                            onClick={handleClick}
                            direction="negative"
                            title="Negative"
                        />
                    </div>
                {/if}
            </div>
        </div>
    {/if}
</div>

<style>
    .section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .metric-selector {
        display: flex;
        gap: var(--space-3);
        font-size: var(--text-sm);
    }

    .radio-item {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        cursor: pointer;
        padding: var(--space-1);
        border-radius: var(--radius-sm);
    }

    .radio-item:hover {
        background: var(--bg-inset);
    }

    .radio-item input {
        margin: 0;
        cursor: pointer;
        accent-color: var(--accent-primary);
    }

    .stat-label {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-primary);
    }

    .edge-list-group {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .pos-neg-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-3);
    }

    .edge-list {
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }
</style>
