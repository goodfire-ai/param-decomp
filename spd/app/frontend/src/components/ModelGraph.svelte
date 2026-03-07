<script lang="ts">
    import { SvelteMap } from "svelte/reactivity";
    import type { GraphNode, GraphEdge } from "../lib/api";
    import { sortRows, getRowKey } from "../lib/graphLayout";
    import { useZoomPan } from "../lib/useZoomPan.svelte";

    interface Props {
        nodes: GraphNode[];
        edges: GraphEdge[];
    }

    let { nodes, edges }: Props = $props();

    let innerContainer: HTMLDivElement | undefined = $state();
    const zoom = useZoomPan(() => innerContainer ?? null);

    // -- Controls --
    let minAttrThreshold = $state(0.1);
    let searchText = $state("");
    let showConfidence = $state<Record<string, boolean>>({ high: true, medium: true, low: true });
    let selectedNodeKey = $state<string | null>(null);
    let hoveredNodeKey = $state<string | null>(null);
    let showAllEdges = $state(false);

    // -- Layout computation --
    type LayoutNode = {
        key: string;
        label: string;
        confidence: string;
        x: number;
        y: number;
        rowKey: string;
    };

    const ROW_HEIGHT = 60;
    const NODE_RADIUS = 6;
    const PADDING_X = 80;
    const PADDING_Y = 40;

    const layout = $derived.by(() => {
        // Group nodes by row (layer group)
        const rowGroups = new SvelteMap<string, GraphNode[]>();
        for (const node of nodes) {
            const layer = node.component_key.split(":")[0];
            const rk = getRowKey(layer);
            if (!rowGroups.has(rk)) rowGroups.set(rk, []);
            rowGroups.get(rk)!.push(node);
        }

        // Sort rows topologically
        const sortedRowKeys = sortRows([...rowGroups.keys()]);

        // Assign positions
        const layoutNodes = new SvelteMap<string, LayoutNode>();
        const rowYMap = new SvelteMap<string, number>();

        // Rows go bottom-to-top (early layers at bottom)
        const totalRows = sortedRowKeys.length;

        for (let ri = 0; ri < sortedRowKeys.length; ri++) {
            const rk = sortedRowKeys[ri];
            const y = PADDING_Y + (totalRows - 1 - ri) * ROW_HEIGHT;
            rowYMap.set(rk, y);

            const rowNodes = rowGroups.get(rk) ?? [];
            // Sort nodes within row by component index
            rowNodes.sort((a, b) => {
                const aIdx = parseInt(a.component_key.split(":")[1] ?? "0");
                const bIdx = parseInt(b.component_key.split(":")[1] ?? "0");
                return aIdx - bIdx;
            });

            const spacing = Math.max(4, Math.min(14, 800 / Math.max(rowNodes.length, 1)));
            const rowWidth = (rowNodes.length - 1) * spacing;
            const startX = PADDING_X + Math.max(0, (800 - rowWidth) / 2);

            for (let ni = 0; ni < rowNodes.length; ni++) {
                const node = rowNodes[ni];
                layoutNodes.set(node.component_key, {
                    key: node.component_key,
                    label: node.label,
                    confidence: node.confidence,
                    x: startX + ni * spacing,
                    y,
                    rowKey: rk,
                });
            }
        }

        const width = 800 + PADDING_X * 2;
        const height = PADDING_Y * 2 + (totalRows - 1) * ROW_HEIGHT;

        return { nodes: layoutNodes, rowYMap, sortedRowKeys, width, height };
    });

    // -- Filtering --
    const filteredNodes = $derived.by(() => {
        const result: LayoutNode[] = [];
        for (const node of layout.nodes.values()) {
            if (!showConfidence[node.confidence]) continue;
            if (searchText && !node.label.toLowerCase().includes(searchText.toLowerCase())) continue;
            result.push(node);
        }
        return result;
    });

    const filteredNodeKeys = $derived(new Set(filteredNodes.map((n) => n.key)));

    const visibleEdges = $derived.by(() => {
        const activeKey = selectedNodeKey ?? hoveredNodeKey;
        return edges.filter((e) => {
            if (Math.abs(e.attribution) < minAttrThreshold) return false;
            if (!filteredNodeKeys.has(e.source) || !filteredNodeKeys.has(e.target)) return false;
            if (!showAllEdges && activeKey) {
                return e.source === activeKey || e.target === activeKey;
            }
            return showAllEdges;
        });
    });

    // -- Tooltip --
    const tooltipNode = $derived(hoveredNodeKey ? layout.nodes.get(hoveredNodeKey) : null);

    const selectedNodeEdges = $derived.by(() => {
        if (!selectedNodeKey) return [];
        return edges
            .filter((e) => e.source === selectedNodeKey || e.target === selectedNodeKey)
            .sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution))
            .slice(0, 20);
    });

    function confidenceColor(conf: string): string {
        switch (conf) {
            case "high":
                return "var(--status-positive-bright)";
            case "medium":
                return "var(--status-warning)";
            default:
                return "var(--text-muted)";
        }
    }

    function edgeColor(attr: number): string {
        return attr >= 0 ? "var(--accent-primary)" : "var(--status-negative)";
    }

    function edgePath(e: GraphEdge): string {
        const src = layout.nodes.get(e.source);
        const tgt = layout.nodes.get(e.target);
        if (!src || !tgt) return "";
        const midY = (src.y + tgt.y) / 2;
        return `M ${src.x} ${src.y} C ${src.x} ${midY}, ${tgt.x} ${midY}, ${tgt.x} ${tgt.y}`;
    }
</script>

<div class="model-graph-container">
    <!-- Controls bar -->
    <div class="controls-bar">
        <div class="control-group">
            <label class="control-label">
                <input type="checkbox" bind:checked={showConfidence.high} /> High
            </label>
            <label class="control-label">
                <input type="checkbox" bind:checked={showConfidence.medium} /> Med
            </label>
            <label class="control-label">
                <input type="checkbox" bind:checked={showConfidence.low} /> Low
            </label>
        </div>
        <div class="control-group">
            <label class="control-label">
                Min attr:
                <input type="range" min="0" max="1" step="0.01" bind:value={minAttrThreshold} class="range-input" />
                <span class="range-value">{minAttrThreshold.toFixed(2)}</span>
            </label>
        </div>
        <div class="control-group">
            <input type="text" placeholder="Search labels..." bind:value={searchText} class="search-input" />
        </div>
        <div class="control-group">
            <label class="control-label">
                <input type="checkbox" bind:checked={showAllEdges} /> Show all edges
            </label>
        </div>
        <div class="control-group stats">
            {filteredNodes.length} nodes, {visibleEdges.length} edges
        </div>
        <div class="control-group">
            <button class="zoom-btn" onclick={zoom.zoomIn}>+</button>
            <button class="zoom-btn" onclick={zoom.zoomOut}>-</button>
            <button class="zoom-btn" onclick={zoom.reset}>Reset</button>
        </div>
    </div>

    <!-- Graph viewport -->
    <div class="graph-viewport" bind:this={innerContainer} role="img" aria-label="Model graph">
        <svg
            width={layout.width}
            height={layout.height}
            style="transform: translate({zoom.translateX}px, {zoom.translateY}px) scale({zoom.scale})"
        >
            <!-- Row labels -->
            {#each layout.sortedRowKeys as rk (rk)}
                {@const y = layout.rowYMap.get(rk)}
                {#if y !== undefined}
                    <text x={10} {y} dy="4" class="row-label">{rk}</text>
                {/if}
            {/each}

            <!-- Edges -->
            {#each visibleEdges as e, i (i)}
                <path
                    d={edgePath(e)}
                    fill="none"
                    stroke={edgeColor(e.attribution)}
                    stroke-width={Math.max(0.5, Math.abs(e.attribution) * 2)}
                    opacity={Math.min(0.8, 0.2 + Math.abs(e.attribution) * 0.6)}
                />
            {/each}

            <!-- Nodes -->
            {#each filteredNodes as node (node.key)}
                <circle
                    cx={node.x}
                    cy={node.y}
                    r={NODE_RADIUS}
                    fill={confidenceColor(node.confidence)}
                    opacity={selectedNodeKey === node.key ? 1 : 0.7}
                    stroke={selectedNodeKey === node.key ? "white" : "none"}
                    stroke-width={selectedNodeKey === node.key ? 2 : 0}
                    style="cursor: pointer"
                    onmouseenter={() => (hoveredNodeKey = node.key)}
                    onmouseleave={() => (hoveredNodeKey = null)}
                    onclick={() => (selectedNodeKey = selectedNodeKey === node.key ? null : node.key)}
                    role="button"
                    tabindex="0"
                    aria-label={node.label}
                />
            {/each}
        </svg>
    </div>

    <!-- Tooltip -->
    {#if tooltipNode}
        <div
            class="tooltip"
            style="left: {tooltipNode.x * zoom.scale + zoom.translateX + 12}px; top: {tooltipNode.y * zoom.scale +
                zoom.translateY -
                8}px"
        >
            <div class="tooltip-label">{tooltipNode.label}</div>
            <div class="tooltip-meta">
                <span class="tooltip-confidence confidence-{tooltipNode.confidence}">{tooltipNode.confidence}</span>
                <span class="tooltip-key">{tooltipNode.key}</span>
            </div>
        </div>
    {/if}

    <!-- Selected node detail -->
    {#if selectedNodeKey}
        {@const selectedNode = layout.nodes.get(selectedNodeKey)}
        {#if selectedNode}
            <div class="detail-panel">
                <div class="detail-header">
                    <span class="detail-label">{selectedNode.label}</span>
                    <span class="confidence confidence-{selectedNode.confidence}">{selectedNode.confidence}</span>
                    <button class="close-btn" onclick={() => (selectedNodeKey = null)}>x</button>
                </div>
                <div class="detail-key">{selectedNode.key}</div>
                <div class="detail-edges">
                    {#if selectedNodeEdges.length > 0}
                        <div class="edge-list">
                            {#each selectedNodeEdges as e, i (i)}
                                {@const other = e.source === selectedNodeKey ? e.target : e.source}
                                {@const otherNode = layout.nodes.get(other)}
                                <div class="edge-item">
                                    <span class="edge-dir">{e.source === selectedNodeKey ? "to" : "from"}</span>
                                    <span class="edge-label">{otherNode?.label ?? other}</span>
                                    <span class="edge-attr" style="color: {edgeColor(e.attribution)}"
                                        >{e.attribution.toFixed(3)}</span
                                    >
                                </div>
                            {/each}
                        </div>
                    {:else}
                        <span class="no-edges">No edges</span>
                    {/if}
                </div>
            </div>
        {/if}
    {/if}
</div>

<style>
    .model-graph-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
        position: relative;
    }

    .controls-bar {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
        flex-shrink: 0;
        flex-wrap: wrap;
    }

    .control-group {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        font-size: var(--text-xs);
        color: var(--text-secondary);
        font-family: var(--font-sans);
    }

    .control-group.stats {
        color: var(--text-muted);
        margin-left: auto;
    }

    .control-label {
        display: flex;
        align-items: center;
        gap: 4px;
        cursor: pointer;
        white-space: nowrap;
    }

    .range-input {
        width: 80px;
    }

    .range-value {
        font-family: var(--font-mono);
        min-width: 32px;
    }

    .search-input {
        padding: 2px var(--space-2);
        font-size: var(--text-xs);
        background: var(--bg-inset);
        color: var(--text-primary);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        width: 140px;
    }

    .zoom-btn {
        padding: 2px var(--space-2);
        font-size: var(--text-xs);
        background: var(--bg-inset);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        cursor: pointer;
    }

    .zoom-btn:hover {
        background: var(--bg-surface);
    }

    .graph-viewport {
        flex: 1;
        overflow: hidden;
        background: var(--bg-base);
        position: relative;
    }

    .graph-viewport svg {
        transform-origin: 0 0;
    }

    .row-label {
        font-size: 9px;
        fill: var(--text-muted);
        font-family: var(--font-mono);
    }

    /* Tooltip */
    .tooltip {
        position: absolute;
        pointer-events: none;
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        padding: var(--space-2);
        z-index: 10;
        max-width: 280px;
    }

    .tooltip-label {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-primary);
    }

    .tooltip-meta {
        display: flex;
        gap: var(--space-2);
        align-items: center;
        margin-top: 2px;
    }

    .tooltip-confidence {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .tooltip-key {
        font-size: 10px;
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .confidence-high {
        color: var(--status-positive-bright);
    }
    .confidence-medium {
        color: var(--status-warning);
    }
    .confidence-low {
        color: var(--text-muted);
    }

    /* Detail panel */
    .detail-panel {
        position: absolute;
        right: var(--space-3);
        top: 60px;
        width: 320px;
        max-height: calc(100% - 80px);
        overflow-y: auto;
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        padding: var(--space-3);
        z-index: 10;
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .detail-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .detail-label {
        font-weight: 600;
        font-size: var(--text-sm);
        color: var(--text-primary);
        flex: 1;
    }

    .close-btn {
        background: none;
        border: none;
        color: var(--text-muted);
        cursor: pointer;
        font-size: var(--text-sm);
        padding: 0 var(--space-1);
    }

    .detail-key {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .edge-list {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .edge-item {
        display: flex;
        gap: var(--space-2);
        font-size: var(--text-xs);
        align-items: baseline;
    }

    .edge-dir {
        color: var(--text-muted);
        min-width: 28px;
    }

    .edge-label {
        flex: 1;
        color: var(--text-secondary);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .edge-attr {
        font-family: var(--font-mono);
        font-weight: 600;
    }

    .no-edges {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .confidence {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
    }
</style>
