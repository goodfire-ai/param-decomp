<script lang="ts">
    import type { SubcomponentCorrelationsResponse } from "../../lib/promptAttributionsTypes";
    import { displaySettings } from "../../lib/displaySettings.svelte";
    import ComponentCorrelationPills from "../prompt-attr/ComponentCorrelationPills.svelte";

    type Props = {
        correlations: SubcomponentCorrelationsResponse;
        pageSize: number;
        onComponentClick?: (componentKey: string) => void;
    };

    let { correlations, pageSize, onComponentClick }: Props = $props();
</script>

<div class="correlations-grid">
    {#if displaySettings.showPmi}
        <ComponentCorrelationPills title="PMI" items={correlations.pmi} {onComponentClick} {pageSize}>
            {#snippet mathNotation()}
                log(P(<span class="color-this">this</span>, <span class="color-that">that</span>) / P(<span
                    class="color-this">this</span
                >)P(<span class="color-that">that</span>))
            {/snippet}
        </ComponentCorrelationPills>
    {/if}
    {#if displaySettings.showPrecision}
        <ComponentCorrelationPills title="Predictors" items={correlations.precision} {onComponentClick} {pageSize}>
            {#snippet mathNotation()}
                <span class="color-both">P</span>(<span class="color-this">this</span> |
                <span class="color-that">that</span>)
            {/snippet}
        </ComponentCorrelationPills>
    {/if}
    {#if displaySettings.showRecall}
        <ComponentCorrelationPills title="Predictees" items={correlations.recall} {onComponentClick} {pageSize}>
            {#snippet mathNotation()}
                <span class="color-both">P</span>(<span class="color-that">that</span> |
                <span class="color-this">this</span>)
            {/snippet}
        </ComponentCorrelationPills>
    {/if}
    {#if displaySettings.showJaccard}
        <ComponentCorrelationPills title="Jaccard" items={correlations.jaccard} {onComponentClick} {pageSize}>
            {#snippet mathNotation()}
                <span class="color-both">this</span> ∩ <span class="color-both">that</span> / (<span class="color-this"
                    >this</span
                >
                ∪ <span class="color-that">that</span>)
            {/snippet}
        </ComponentCorrelationPills>
    {/if}
</div>

<style>
    .correlations-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: var(--space-4);
    }
</style>
