<script lang="ts">
    import { getContext } from "svelte";
    import { colors } from "../../lib/colors";
    import { isTokenNode } from "../../lib/componentKeys";
    import type { EdgeAttribution } from "../../lib/promptAttributionsTypes";
    import { RUN_KEY, type InterpretationBackendState, type RunContext } from "../../lib/useRun.svelte";
    import { lerp } from "../prompt-attr/graphUtils";

    const runState = getContext<RunContext>(RUN_KEY);

    type Props = {
        items: EdgeAttribution[];
        onClick: (key: string) => void;
        pageSize: number;
        direction?: "positive" | "negative";
        title?: string;
    };

    let { items, onClick, pageSize, direction, title }: Props = $props();

    function getComponentKey(key: string): string {
        const parts = key.split(":");
        if (parts.length === 3) {
            return `${parts[0]}:${parts[2]}`;
        }
        return key;
    }

    function getLayerLabel(key: string): string {
        return key.split(":")[0];
    }

    function getInterpretation(key: string): InterpretationBackendState {
        const componentKey = getComponentKey(key);
        const interp = runState.getInterpretation(componentKey);
        if (interp.status === "loaded" && interp.data.status === "generated") return interp.data;
        return { status: "none" };
    }

    function getTokenTypeLabel(key: string): string {
        const layer = key.split(":")[0];
        return layer === "embed" ? "Input token" : "Output token";
    }

    let currentPage = $state(0);
    const totalPages = $derived(Math.ceil(items.length / pageSize));
    const paginatedItems = $derived(items.slice(currentPage * pageSize, (currentPage + 1) * pageSize));

    let hoveredKey = $state<string | null>(null);
    let tooltipPosition = $state<{ top: number; left: number } | null>(null);

    function handleMouseEnter(key: string, event: MouseEvent) {
        hoveredKey = key;
        const target = event.currentTarget as HTMLElement;
        const rect = target.getBoundingClientRect();
        tooltipPosition = { top: rect.top, left: rect.left };
    }

    function handleMouseLeave() {
        hoveredKey = null;
        tooltipPosition = null;
    }

    $effect(() => {
        items; // eslint-disable-line @typescript-eslint/no-unused-expressions
        currentPage = 0;
    });

    function getBgColor(normalizedMagnitude: number): string {
        const intensity = lerp(0, 0.8, normalizedMagnitude);
        const rgb = direction === "negative" ? colors.negativeRgb : colors.positiveRgb;
        return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${intensity})`;
    }

    async function copyToClipboard(text: string) {
        await navigator.clipboard.writeText(text);
    }
</script>

<div class="edge-attribution-list">
    <div class="header-row">
        {#if title}
            <span class="list-title">{title}</span>
        {/if}
        {#if totalPages > 1}
            <div class="pagination">
                <button onclick={() => currentPage--} disabled={currentPage === 0}>&lt;</button>
                <span>{currentPage + 1} / {totalPages}</span>
                <button onclick={() => currentPage++} disabled={currentPage >= totalPages - 1}>&gt;</button>
            </div>
        {/if}
    </div>
    <div class="items">
        {#each paginatedItems as { key, value, normalizedMagnitude, tokenStr } (key)}
            {@const bgColor = getBgColor(normalizedMagnitude)}
            {@const textColor = normalizedMagnitude > 0.8 ? "white" : "var(--text-primary)"}
            {@const formattedKey = key}
            {@const isToken = isTokenNode(key)}
            {@const interp = isToken ? undefined : getInterpretation(key)}
            {@const hasInterpretation = interp?.status === "generated"}
            {@const polarity = value >= 0 ? "+" : "\u2212"}
            <div class="pill-container" onmouseenter={(e) => handleMouseEnter(key, e)} onmouseleave={handleMouseLeave}>
                <button class="edge-pill" style="background: {bgColor};" onclick={() => onClick(key)}>
                    {#if !direction}
                        <span class="polarity" style="color: {textColor};">{polarity}</span>
                    {/if}
                    {#if hasInterpretation}
                        <span class="pill-content">
                            <span class="interp-label" style="color: {textColor};">{interp.data.label}</span>
                            <span class="layer-label" style="color: {textColor};">{getLayerLabel(key)}</span>
                        </span>
                    {:else if isToken && tokenStr}
                        <span class="pill-content">
                            <span class="interp-label" style="color: {textColor};">'{tokenStr}'</span>
                            <span class="layer-label" style="color: {textColor};">{getTokenTypeLabel(key)}</span>
                        </span>
                    {:else}
                        <span class="node-key" style="color: {textColor};">{formattedKey}</span>
                    {/if}
                </button>
                {#if hoveredKey === key && tooltipPosition}
                    <!-- svelte-ignore a11y_no_static_element_interactions -->
                    <div
                        class="tooltip"
                        style="top: {tooltipPosition.top}px; left: {tooltipPosition.left}px;"
                        onmouseenter={() => (hoveredKey = key)}
                        onmouseleave={handleMouseLeave}
                    >
                        <div class="tooltip-value">Attribution: {value.toFixed(3)}</div>
                        {#if hasInterpretation}
                            <div class="tooltip-label">{interp.data.label}</div>
                            <button class="tooltip-node-key copyable" onclick={() => copyToClipboard(formattedKey)}>
                                {formattedKey}
                                <svg
                                    class="copy-icon"
                                    width="12"
                                    height="12"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    stroke-width="2"
                                >
                                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                </svg>
                            </button>
                            <div class="tooltip-confidence">Confidence: {interp.data.confidence}</div>
                        {:else if isToken && tokenStr}
                            <div class="tooltip-token">Token: '{tokenStr}'</div>
                        {/if}
                    </div>
                {/if}
            </div>
        {/each}
    </div>
</div>

<style>
    .edge-attribution-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .header-row {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        min-height: 1.25rem;
    }

    .list-title {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-style: italic;
    }

    .items {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-2);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        background: var(--bg-elevated);
        padding: var(--space-2);
        border: 1px solid var(--border-default);
    }

    .pagination {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .pagination button {
        padding: 2px 6px;
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-secondary);
        cursor: pointer;
        font-size: var(--text-xs);
    }

    .pagination button:hover:not(:disabled) {
        background: var(--bg-surface);
        border-color: var(--border-strong);
    }

    .pagination button:disabled {
        opacity: 0.4;
        cursor: default;
    }

    .pill-container {
        position: relative;
        flex: 0 1 auto;
        min-width: 0;
        max-width: 100%;
    }

    .edge-pill {
        display: inline-flex;
        align-items: flex-start;
        gap: var(--space-1);
        padding: var(--space-1) var(--space-1);
        border-radius: var(--radius-sm);
        cursor: default;
        border: 1px solid var(--border-default);
        font-family: inherit;
        font-size: inherit;
        max-width: 100%;
    }

    .polarity {
        font-weight: 700;
        font-size: 10px;
        line-height: 1;
        flex-shrink: 0;
        opacity: 0.7;
    }

    .node-key {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        overflow-wrap: break-word;
        text-align: left;
    }

    .pill-content {
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: baseline;
        gap: var(--space-2);
    }

    .interp-label {
        font-family: var(--font-sans);
        font-size: var(--text-xs);
        font-weight: 500;
        overflow-wrap: break-word;
        text-align: left;
    }

    .layer-label {
        font-family: var(--font-mono);
        font-size: 9px;
        opacity: 0.7;
        text-align: right;
    }

    .tooltip {
        position: fixed;
        transform: translateY(-100%);
        margin-top: -8px;
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-sm);
        box-shadow: var(--shadow-lg);
        z-index: 10000;
        min-width: 200px;
        max-width: 350px;
    }

    .tooltip-value {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: var(--space-1);
    }

    .tooltip-token {
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-secondary);
    }

    .tooltip-label {
        font-family: var(--font-sans);
        font-weight: 600;
        font-size: var(--text-sm);
        color: var(--text-primary);
        margin-bottom: var(--space-1);
        word-wrap: break-word;
    }

    .tooltip-node-key {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-secondary);
        margin-bottom: var(--space-1);
    }

    .tooltip-node-key.copyable {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        background: none;
        border: none;
        padding: var(--space-1) var(--space-1);
        margin: calc(-1 * var(--space-1)) calc(-1 * var(--space-1));
        margin-bottom: var(--space-1);
        border-radius: var(--radius-sm);
        cursor: pointer;
        text-align: left;
    }

    .tooltip-node-key.copyable:hover {
        background: var(--bg-surface);
    }

    .tooltip-node-key.copyable .copy-icon {
        opacity: 0.4;
        flex-shrink: 0;
    }

    .tooltip-node-key.copyable:hover .copy-icon {
        opacity: 0.8;
    }

    .tooltip-confidence {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
    }
</style>
