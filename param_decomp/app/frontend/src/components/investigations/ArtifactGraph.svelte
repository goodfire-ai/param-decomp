<script lang="ts">
    /**
     * Graph component for rendering artifacts in research reports.
     * Includes tooltips using the same NodeTooltip as the main graph.
     */

    import { getContext } from "svelte";
    import type { NodePosition, EdgeData, OutputProbability, HoveredNode } from "../../lib/promptAttributionsTypes";
    import { buildEdgeIndexes } from "../../lib/promptAttributionsTypes";
    import type { ArtifactGraphData } from "../../lib/api/investigations";
    import { getRowKey, sortRows, getRowLabel } from "../../lib/graphLayout";
    import { colors, getEdgeColor, rgbToCss } from "../../lib/colors";
    import {
        lerp,
        sortComponentsByImportance,
        computeComponentOffsets,
        calcTooltipPos,
        type TooltipPos,
    } from "../prompt-attr/graphUtils";
    import { useZoomPan } from "../../lib/useZoomPan.svelte";
    import ZoomControls from "../../lib/ZoomControls.svelte";
    import NodeTooltip from "../prompt-attr/NodeTooltip.svelte";
    import { RUN_KEY, type RunContext } from "../../lib/useRun.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    const COMPONENT_SIZE = 8;
    const HIT_AREA_PADDING = 4;
    const MARGIN = { top: 60, right: 40, bottom: 20, left: 20 };
    const LABEL_WIDTH = 100;

    type Props = {
        data: ArtifactGraphData;
        caption?: string;
        topK?: number;
        componentGap?: number;
        layerGap?: number;
    };

    let { data, caption, topK = 200, componentGap = 4, layerGap = 24 }: Props = $props();

    let hoveredNode = $state<HoveredNode | null>(null);
    let isHoveringTooltip = $state(false);
    let tooltipPos = $state<TooltipPos>({ left: 0, top: 0 });
    let hoverTimeout: ReturnType<typeof setTimeout> | null = null;

    let innerContainer: HTMLDivElement;
    const zoom = useZoomPan(() => innerContainer);

    function computeLayout(graphData: ArtifactGraphData, edges: EdgeData[], compGap: number, lGap: number) {
        // Only lay out nodes that appear in edges
        // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local variable, not reactive state
        const activeNodes = new Set<string>();
        for (const edge of edges) {
            activeNodes.add(edge.src);
            activeNodes.add(edge.tgt);
        }

        const nodesPerLayerSeq: Record<string, number[]> = {};
        // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local variable, not reactive state
        const allLayers = new Set<string>();
        // eslint-disable-next-line svelte/prefer-svelte-reactivity -- local variable, not reactive state
        const allRows = new Set<string>();

        for (const nodeKey of activeNodes) {
            const [layer, seqIdx, cIdx] = nodeKey.split(":");
            allLayers.add(layer);
            allRows.add(getRowKey(layer));
            const key = `${layer}:${seqIdx}`;
            if (!nodesPerLayerSeq[key]) nodesPerLayerSeq[key] = [];
            nodesPerLayerSeq[key].push(+cIdx);
        }

        const rows = sortRows(Array.from(allRows));

        // Assign Y positions
        const rowYPositions: Record<string, number> = {};
        for (let i = 0; i < rows.length; i++) {
            const distanceFromEnd = rows.length - 1 - i;
            rowYPositions[rows[i]] = MARGIN.top + distanceFromEnd * (COMPONENT_SIZE + lGap);
        }

        const layerYPositions: Record<string, number> = {};
        for (const layer of allLayers) {
            layerYPositions[layer] = rowYPositions[getRowKey(layer)];
        }

        // Calculate column widths
        const tokens = graphData.tokens;
        const maxComponentsPerSeq = tokens.map((_: string, seqIdx: number) => {
            let maxAtSeq = 0;
            for (const layer of allLayers) {
                const nodes = nodesPerLayerSeq[`${layer}:${seqIdx}`] ?? [];
                maxAtSeq = Math.max(maxAtSeq, nodes.length);
            }
            return maxAtSeq;
        });

        const MIN_COL_WIDTH = 30;
        const COL_PADDING = 16;
        const seqWidths = maxComponentsPerSeq.map((n: number) =>
            Math.max(MIN_COL_WIDTH, n * (COMPONENT_SIZE + compGap) + COL_PADDING * 2),
        );
        const seqXStarts = [MARGIN.left];
        for (let i = 0; i < seqWidths.length - 1; i++) {
            seqXStarts.push(seqXStarts[i] + seqWidths[i]);
        }

        // Position nodes
        const nodePositions: Record<string, NodePosition> = {};
        for (const layer of allLayers) {
            for (let seqIdx = 0; seqIdx < tokens.length; seqIdx++) {
                const nodes = nodesPerLayerSeq[`${layer}:${seqIdx}`];
                if (!nodes) continue;

                const baseX = seqXStarts[seqIdx] + COL_PADDING;
                const baseY = layerYPositions[layer];

                const sorted = sortComponentsByImportance(
                    nodes,
                    layer,
                    seqIdx,
                    graphData.nodeCiVals,
                    graphData.outputProbs,
                );
                const offsets = computeComponentOffsets(sorted, COMPONENT_SIZE, compGap);

                for (const cIdx of nodes) {
                    nodePositions[`${layer}:${seqIdx}:${cIdx}`] = {
                        x: baseX + offsets[cIdx] + COMPONENT_SIZE / 2,
                        y: baseY + COMPONENT_SIZE / 2,
                    };
                }
            }
        }

        const totalSeqWidth = seqXStarts[seqXStarts.length - 1] + seqWidths[seqWidths.length - 1];
        const width = totalSeqWidth + MARGIN.right;
        const maxY = Math.max(...Object.values(layerYPositions), 0) + COMPONENT_SIZE;
        const height = maxY + MARGIN.bottom;

        return { nodePositions, layerYPositions, seqXStarts, width, height };
    }

    function getFilteredEdges(edges: EdgeData[], k: number): EdgeData[] {
        return [...edges].sort((a, b) => Math.abs(b.val) - Math.abs(a.val)).slice(0, k);
    }

    function computeNodeStyles(
        positions: Record<string, NodePosition>,
        nodeCiVals: Record<string, number>,
        outputProbs: Record<string, OutputProbability>,
    ): Record<string, { fill: string; opacity: number }> {
        const styles: Record<string, { fill: string; opacity: number }> = {};
        const maxCi = Math.max(...Object.values(nodeCiVals), 1);

        for (const nodeKey of Object.keys(positions)) {
            const [layer, seqIdx, cIdx] = nodeKey.split(":");
            let fill: string = colors.nodeDefault;
            let opacity = 0.2;

            if (layer === "output") {
                const probEntry = outputProbs[`${seqIdx}:${cIdx}`];
                if (probEntry) {
                    fill = rgbToCss(colors.outputBase);
                    opacity = 0.2 + probEntry.prob * 0.8;
                }
            } else {
                const ci = nodeCiVals[nodeKey] || 0;
                opacity = 0.2 + (ci / maxCi) * 0.8;
            }

            styles[nodeKey] = { fill, opacity };
        }

        return styles;
    }

    function buildEdgesSvg(edges: EdgeData[], positions: Record<string, NodePosition>, maxAbsAttr: number): string {
        let svg = "";
        for (let i = edges.length - 1; i >= 0; i--) {
            const edge = edges[i];
            const p1 = positions[edge.src];
            const p2 = positions[edge.tgt];
            if (p1 && p2) {
                const color = getEdgeColor(edge.val);
                const w = lerp(1, 4, Math.abs(edge.val) / maxAbsAttr);
                const op = lerp(0.1, 0.6, Math.abs(edge.val) / maxAbsAttr);
                const dy = Math.abs(p2.y - p1.y);
                const curveOffset = Math.max(20, dy * 0.4);
                const d = `M ${p1.x},${p1.y} C ${p1.x},${p1.y - curveOffset} ${p2.x},${p2.y + curveOffset} ${p2.x},${p2.y}`;
                svg += `<path d="${d}" stroke="${color}" stroke-width="${w}" opacity="${op}" fill="none"/>`;
            }
        }
        return svg;
    }

    const filteredEdges = $derived(getFilteredEdges(data.edges, topK));
    const layout = $derived(computeLayout(data, filteredEdges, componentGap, layerGap));
    const nodeStyles = $derived(computeNodeStyles(layout.nodePositions, data.nodeCiVals, data.outputProbs));
    const edgesSvg = $derived(buildEdgesSvg(filteredEdges, layout.nodePositions, data.maxAbsAttr || 1));
    const edgeIndexes = $derived(buildEdgeIndexes(filteredEdges));

    const svgWidth = $derived(layout.width * zoom.scale + Math.max(zoom.translateX, 0));
    const svgHeight = $derived(layout.height * zoom.scale + Math.max(zoom.translateY, 0));

    // Hover handlers
    function handleNodeHover(event: MouseEvent, layer: string, seqIdx: number, cIdx: number) {
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
            hoverTimeout = null;
        }
        hoveredNode = { layer, seqIdx, cIdx };
        const size = layer === "embed" || layer === "output" ? "small" : "large";
        tooltipPos = calcTooltipPos(event.clientX, event.clientY, size);
    }

    function handleNodeLeave() {
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
        }
        hoverTimeout = setTimeout(() => {
            if (!isHoveringTooltip) {
                hoveredNode = null;
            }
            hoverTimeout = null;
        }, 100);
    }

    function handlePanStart(event: MouseEvent) {
        if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
            zoom.startPan(event);
        }
    }
</script>

<div class="artifact-graph">
    {#if caption}
        <div class="caption">{caption}</div>
    {/if}

    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
        class="graph-wrapper"
        class:panning={zoom.isPanning}
        onmousedown={handlePanStart}
        onmousemove={zoom.updatePan}
        onmouseup={zoom.endPan}
        onmouseleave={zoom.endPan}
    >
        <ZoomControls scale={zoom.scale} onZoomIn={zoom.zoomIn} onZoomOut={zoom.zoomOut} onReset={zoom.reset} />

        <div class="layer-labels-container" style="width: {LABEL_WIDTH}px;">
            <svg width={LABEL_WIDTH} height={svgHeight} style="display: block;">
                <g transform="translate(0, {zoom.translateY}) scale(1, {zoom.scale})">
                    {#each Object.entries(layout.layerYPositions) as [layer, y] (layer)}
                        <text
                            x={LABEL_WIDTH - 10}
                            y={y + COMPONENT_SIZE / 2}
                            text-anchor="end"
                            dominant-baseline="middle"
                            font-size="10"
                            font-weight="500"
                            font-family="'Berkeley Mono', 'SF Mono', monospace"
                            fill={colors.textSecondary}
                        >
                            {getRowLabel(getRowKey(layer))}
                        </text>
                    {/each}
                </g>
            </svg>
        </div>

        <div class="graph-container" bind:this={innerContainer}>
            <svg width={svgWidth} height={svgHeight}>
                <g transform="translate({zoom.translateX}, {zoom.translateY}) scale({zoom.scale})">
                    <g class="edges-layer">
                        <!-- eslint-disable-next-line svelte/no-at-html-tags -->
                        {@html edgesSvg}
                    </g>

                    <g class="nodes-layer">
                        {#each Object.entries(layout.nodePositions) as [key, pos] (key)}
                            {@const style = nodeStyles[key]}
                            {@const [layer, seqIdxStr, cIdxStr] = key.split(":")}
                            {@const seqIdx = parseInt(seqIdxStr)}
                            {@const cIdx = parseInt(cIdxStr)}
                            <!-- Hit area for easier hovering -->
                            <rect
                                x={pos.x - COMPONENT_SIZE / 2 - HIT_AREA_PADDING}
                                y={pos.y - COMPONENT_SIZE / 2 - HIT_AREA_PADDING}
                                width={COMPONENT_SIZE + HIT_AREA_PADDING * 2}
                                height={COMPONENT_SIZE + HIT_AREA_PADDING * 2}
                                fill="transparent"
                                onmouseenter={(e) => handleNodeHover(e, layer, seqIdx, cIdx)}
                                onmouseleave={handleNodeLeave}
                            />
                            <!-- Visible node -->
                            <rect
                                x={pos.x - COMPONENT_SIZE / 2}
                                y={pos.y - COMPONENT_SIZE / 2}
                                width={COMPONENT_SIZE}
                                height={COMPONENT_SIZE}
                                fill={style.fill}
                                rx="1"
                                opacity={style.opacity}
                                pointer-events="none"
                            />
                        {/each}
                    </g>
                </g>
            </svg>

            <div class="token-labels-container">
                <svg width={svgWidth} height="50" style="display: block;">
                    <g transform="translate({zoom.translateX}, 0) scale({zoom.scale}, 1)">
                        {#each data.tokens as token, i (i)}
                            {@const colLeft = layout.seqXStarts[i] + 8}
                            <text
                                x={colLeft}
                                y="20"
                                text-anchor="start"
                                font-size="11"
                                font-family="'Berkeley Mono', 'SF Mono', monospace"
                                font-weight="500"
                                fill={colors.textPrimary}
                            >
                                {token}
                            </text>
                            <text
                                x={colLeft}
                                y="36"
                                text-anchor="start"
                                font-size="9"
                                font-family="'Berkeley Mono', 'SF Mono', monospace"
                                fill={colors.textMuted}>[{i}]</text
                            >
                        {/each}
                    </g>
                </svg>
            </div>
        </div>
    </div>

    <div class="stats">
        L0: {data.l0_total} · Edges: {filteredEdges.length}
    </div>

    <!-- Node tooltip -->
    {#if hoveredNode && runState}
        <NodeTooltip
            {hoveredNode}
            {tooltipPos}
            hideNodeCard={false}
            outputProbs={data.outputProbs}
            nodeCiVals={data.nodeCiVals}
            nodeSubcompActs={data.nodeSubcompActs}
            tokens={data.tokens}
            edgesBySource={edgeIndexes.edgesBySource}
            edgesByTarget={edgeIndexes.edgesByTarget}
            onMouseEnter={() => (isHoveringTooltip = true)}
            onMouseLeave={() => {
                isHoveringTooltip = false;
                hoveredNode = null;
            }}
        />
    {/if}
</div>

<style>
    .artifact-graph {
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        overflow: visible;
        margin: var(--space-3) 0;
        background: var(--bg-surface);
    }

    .caption {
        padding: var(--space-2) var(--space-3);
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-secondary);
        background: var(--bg-elevated);
        border-bottom: 1px solid var(--border-default);
    }

    .graph-wrapper {
        display: flex;
        overflow: visible;
        position: relative;
    }

    .graph-wrapper.panning {
        cursor: grabbing;
    }

    .layer-labels-container {
        position: sticky;
        left: 0;
        background: var(--bg-surface);
        border-right: 1px solid var(--border-default);
        z-index: 11;
        flex-shrink: 0;
    }

    .graph-container {
        overflow: visible;
        flex: 1;
        position: relative;
        background: var(--bg-inset);
    }

    .token-labels-container {
        position: sticky;
        bottom: 0;
        background: var(--bg-surface);
        border-top: 1px solid var(--border-default);
        z-index: 10;
    }

    svg {
        display: block;
    }

    .stats {
        padding: var(--space-1) var(--space-3);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        background: var(--bg-elevated);
        border-top: 1px solid var(--border-default);
    }
</style>
