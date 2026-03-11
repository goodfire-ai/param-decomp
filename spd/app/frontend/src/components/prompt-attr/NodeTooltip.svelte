<script lang="ts">
    import { getContext } from "svelte";
    import { topEdgeAttributions, type OutputProbability, type EdgeData } from "../../lib/promptAttributionsTypes";
    import type { TooltipPos } from "./graphUtils";
    import ComponentNodeCard from "./ComponentNodeCard.svelte";
    import OutputNodeCard from "./OutputNodeCard.svelte";
    import EdgeAttributionGrid from "../ui/EdgeAttributionGrid.svelte";
    import { displaySettings } from "../../lib/displaySettings.svelte";
    import { COMPONENT_CARD_CONSTANTS } from "../../lib/componentCardConstants";
    import { RUN_KEY, type RunContext } from "../../lib/useRun.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    type HoveredNode = {
        layer: string;
        seqIdx: number;
        cIdx: number;
    };

    type Props = {
        hoveredNode: HoveredNode;
        tooltipPos: TooltipPos;
        hideNodeCard?: boolean;
        outputProbs: Record<string, OutputProbability>;
        nodeCiVals: Record<string, number>;
        nodeSubcompActs: Record<string, number>;
        tokens: string[];
        edgesBySource: Map<string, EdgeData[]>;
        edgesByTarget: Map<string, EdgeData[]>;
        onMouseEnter: () => void;
        onMouseLeave: () => void;
        onPinComponent?: (layer: string, cIdx: number, seqIdx: number) => void;
    };

    let {
        hoveredNode,
        tooltipPos,
        hideNodeCard = false,
        outputProbs,
        nodeCiVals,
        nodeSubcompActs,
        tokens,
        edgesBySource,
        edgesByTarget,
        onMouseEnter,
        onMouseLeave,
        onPinComponent,
    }: Props = $props();

    const isWte = $derived(hoveredNode.layer === "embed");
    const isOutput = $derived(hoveredNode.layer === "output");
    const isComponent = $derived(!isWte && !isOutput);
    const clusterId = $derived(isComponent ? runState.getClusterId(hoveredNode.layer, hoveredNode.cIdx) : undefined);

    // Get CI value for component nodes
    const ciVal = $derived.by(() => {
        if (!isComponent) return null;
        const key = `${hoveredNode.layer}:${hoveredNode.seqIdx}:${hoveredNode.cIdx}`;
        return nodeCiVals[key] ?? null;
    });

    // Get subcomponent activation for component nodes
    const subcompAct = $derived.by(() => {
        if (!isComponent) return null;
        const key = `${hoveredNode.layer}:${hoveredNode.seqIdx}:${hoveredNode.cIdx}`;
        return nodeSubcompActs[key] ?? null;
    });

    const token = $derived.by(() => {
        if (hoveredNode.seqIdx >= tokens.length) {
            throw new Error(
                `NodeTooltip: seqIdx ${hoveredNode.seqIdx} out of bounds for tokens length ${tokens.length}`,
            );
        }
        return tokens[hoveredNode.seqIdx];
    });

    const positionStyle = $derived.by(() => {
        const parts: string[] = [];
        if (tooltipPos.left !== undefined) parts.push(`left: ${tooltipPos.left}px`);
        if (tooltipPos.right !== undefined) parts.push(`right: ${tooltipPos.right}px`);
        if (tooltipPos.top !== undefined) parts.push(`top: ${tooltipPos.top}px`);
        if (tooltipPos.bottom !== undefined) parts.push(`bottom: ${tooltipPos.bottom}px`);
        if (tooltipPos.maxHeight !== undefined) parts.push(`max-height: ${tooltipPos.maxHeight}px`);
        return parts.join("; ");
    });

    const wteNodeKey = $derived(`embed:${hoveredNode.seqIdx}:0`);
    const wteOutgoing = $derived(
        isWte ? topEdgeAttributions(edgesBySource.get(wteNodeKey) ?? [], (e) => e.tgt, 20) : [],
    );
</script>

<div
    class="node-tooltip"
    style={positionStyle}
    onmouseenter={onMouseEnter}
    onmouseleave={onMouseLeave}
    onwheel={(e) => e.stopPropagation()}
>
    <h3>{hoveredNode.layer}:{hoveredNode.seqIdx}:{hoveredNode.cIdx}</h3>
    {#if isComponent && ciVal !== null}
        <div class="ci-value">CI: {ciVal.toFixed(3)}</div>
    {/if}
    {#if isComponent && subcompAct !== null}
        <div class="subcomp-act">Subcomp Act: {subcompAct.toFixed(3)}</div>
    {/if}
    {#if clusterId !== undefined}
        <div class="cluster-id">Cluster: {clusterId ?? "null"}</div>
    {/if}
    {#if isWte}
        <p class="wte-info">Input embedding at position {hoveredNode.seqIdx}</p>
        <div class="wte-content">
            <div class="wte-token">"{token}"</div>
            <p class="wte-stats">
                <strong>Position:</strong>
                {hoveredNode.seqIdx}
            </p>
        </div>
        {#if displaySettings.showEdgeAttributions && wteOutgoing.length > 0}
            <EdgeAttributionGrid
                title="Prompt Attributions"
                incomingLabel="Incoming"
                outgoingLabel="Outgoing"
                incoming={[]}
                outgoing={wteOutgoing}
                pageSize={COMPONENT_CARD_CONSTANTS.PROMPT_ATTRIBUTIONS_PAGE_SIZE}
                onClick={() => {}}
            />
        {/if}
    {:else if isOutput}
        <OutputNodeCard cIdx={hoveredNode.cIdx} {outputProbs} seqIdx={hoveredNode.seqIdx} {edgesByTarget} />
    {:else if !hideNodeCard}
        <!-- Key forces remount when component identity changes, so ComponentNodeCard can load on mount -->
        {#key `${hoveredNode.layer}:${hoveredNode.cIdx}`}
            <ComponentNodeCard
                layer={hoveredNode.layer}
                cIdx={hoveredNode.cIdx}
                seqIdx={hoveredNode.seqIdx}
                {ciVal}
                {subcompAct}
                {token}
                {edgesBySource}
                {edgesByTarget}
                {tokens}
                {outputProbs}
                {onPinComponent}
            />
        {/key}
    {/if}
</div>

<style>
    .node-tooltip {
        position: fixed;
        z-index: 1000;
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        padding: var(--space-3);
        width: fit-content;
        max-width: 800px;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: var(--shadow-md);
    }

    .node-tooltip h3 {
        font-size: var(--text-base);
        font-family: var(--font-mono);
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 var(--space-2) 0;
    }

    .wte-info {
        margin: var(--space-2) 0 0 0;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }
</style>
