<script lang="ts">
    import type { GraphInterpComponentDetail, GraphInterpHeadline } from "../../lib/api";
    import { formatComponentKey } from "../../lib/componentKeys";

    interface Props {
        headline: GraphInterpHeadline;
        detail: GraphInterpComponentDetail | null;
    }

    let { headline, detail }: Props = $props();

    let expanded = $state(false);

    const incomingEdges = $derived(
        detail?.edges
            .filter((e) => e.pass_name === "input")
            .sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution)) ?? [],
    );

    const outgoingEdges = $derived(
        detail?.edges
            .filter((e) => e.pass_name === "output")
            .sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution)) ?? [],
    );
</script>

<div class="graph-interp-container">
    <button class="graph-interp-badge" onclick={() => (expanded = !expanded)} type="button">
        <div class="badge-header">
            <span class="badge-label">{headline.label}</span>
            <span class="confidence confidence-{headline.confidence}">{headline.confidence}</span>
            <span class="source-tag">graph</span>
        </div>
        {#if expanded && (headline.output_label || headline.input_label)}
            <div class="sub-labels">
                {#if headline.output_label}
                    <span class="sub-label"><span class="sub-tag">out</span> {headline.output_label}</span>
                {/if}
                {#if headline.input_label}
                    <span class="sub-label"><span class="sub-tag">in</span> {headline.input_label}</span>
                {/if}
            </div>
        {/if}
    </button>

    {#if expanded && detail}
        <div class="detail-section">
            <div class="detail-columns">
                <div class="detail-column">
                    <span class="column-title">Input</span>
                    {#if detail.input?.reasoning}
                        <p class="reasoning-text">{detail.input.reasoning}</p>
                    {/if}
                    {#each incomingEdges as edge (edge.related_key)}
                        <div class="edge-row">
                            <span class="edge-key">{formatComponentKey(edge.related_key, edge.token_str)}</span>
                            <span
                                class="edge-attr"
                                class:positive={edge.attribution > 0}
                                class:negative={edge.attribution < 0}
                            >
                                {edge.attribution > 0 ? "+" : ""}{edge.attribution.toFixed(3)}
                            </span>
                            {#if edge.related_label}
                                <span class="edge-label">{edge.related_label}</span>
                            {/if}
                        </div>
                    {/each}
                </div>
                <div class="detail-column">
                    <span class="column-title">Output</span>
                    {#if detail.output?.reasoning}
                        <p class="reasoning-text">{detail.output.reasoning}</p>
                    {/if}
                    {#each outgoingEdges as edge (edge.related_key)}
                        <div class="edge-row">
                            <span class="edge-key">{formatComponentKey(edge.related_key, edge.token_str)}</span>
                            <span
                                class="edge-attr"
                                class:positive={edge.attribution > 0}
                                class:negative={edge.attribution < 0}
                            >
                                {edge.attribution > 0 ? "+" : ""}{edge.attribution.toFixed(3)}
                            </span>
                            {#if edge.related_label}
                                <span class="edge-label">{edge.related_label}</span>
                            {/if}
                        </div>
                    {/each}
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .graph-interp-container {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .graph-interp-badge {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-inset);
        border-radius: var(--radius-md);
        border-left: 3px solid var(--status-positive-bright);
        border-top: none;
        border-right: none;
        border-bottom: none;
        cursor: pointer;
        text-align: left;
        font: inherit;
        width: 100%;
    }

    .graph-interp-badge:hover {
        background: var(--bg-surface);
    }

    .badge-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .badge-label {
        font-weight: 500;
        color: var(--text-primary);
        font-size: var(--text-sm);
    }

    .confidence {
        font-size: var(--text-xs);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
        text-transform: uppercase;
        font-weight: 600;
    }

    .confidence-high {
        background: color-mix(in srgb, var(--status-positive-bright) 20%, transparent);
        color: var(--status-positive-bright);
    }

    .confidence-medium {
        background: color-mix(in srgb, var(--status-warning) 20%, transparent);
        color: var(--status-warning);
    }

    .confidence-low {
        background: color-mix(in srgb, var(--text-muted) 20%, transparent);
        color: var(--text-muted);
    }

    .source-tag {
        font-size: var(--text-xs);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
        background: color-mix(in srgb, var(--status-positive-bright) 15%, transparent);
        color: var(--status-positive-bright);
        font-weight: 600;
        text-transform: uppercase;
        margin-left: auto;
    }

    .sub-labels {
        display: flex;
        flex-direction: column;
        gap: 2px;
        padding-top: var(--space-1);
    }

    .sub-label {
        font-size: var(--text-xs);
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .sub-tag {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--text-muted);
        min-width: 24px;
    }

    .detail-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        padding: var(--space-3);
        background: var(--bg-elevated);
        border-radius: var(--radius-md);
        border: 1px solid var(--border-default);
    }

    .detail-columns {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-3);
    }

    .detail-column {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .column-title {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--text-muted);
        padding-bottom: var(--space-1);
        border-bottom: 1px solid var(--border-subtle);
    }

    .reasoning-text {
        font-size: var(--text-xs);
        color: var(--text-secondary);
        line-height: 1.5;
        margin: 0;
    }

    .edge-row {
        display: flex;
        align-items: baseline;
        gap: var(--space-2);
        font-size: var(--text-xs);
    }

    .edge-key {
        font-family: var(--font-mono);
        color: var(--text-secondary);
        flex-shrink: 0;
    }

    .edge-attr {
        font-family: var(--font-mono);
        font-weight: 600;
        flex-shrink: 0;
    }

    .edge-attr.positive {
        color: var(--status-positive-bright);
    }

    .edge-attr.negative {
        color: var(--semantic-error);
    }

    .edge-label {
        color: var(--text-muted);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
</style>
