<script lang="ts">
    import { getContext } from "svelte";
    import { SvelteSet } from "svelte/reactivity";
    import { colors, getEdgeColor, getNextTokenProbBgColor, rgbaToCss } from "../../lib/colors";
    import type { Loadable } from "../../lib/index";
    import type { NormalizeType } from "../../lib/api";
    import { isInterventableNode, getActiveEdges, type NodePosition } from "../../lib/promptAttributionsTypes";
    import { displaySettings } from "../../lib/displaySettings.svelte";
    import { RUN_KEY, type RunContext } from "../../lib/useRun.svelte";
    import {
        parseLayer,
        getRowKey as _getRowKey,
        getRowLabel as _getRowLabel,
        sortRows,
        getGroupProjections,
        buildLayerAddress,
    } from "../../lib/graphLayout";

    const runState = getContext<RunContext>(RUN_KEY);
    import {
        calcTooltipPos,
        computeClusterSpans,
        computeComponentOffsets,
        lerp,
        sortComponentsByCluster,
        sortComponentsByImportance,
        type ClusterSpan,
        type TooltipPos,
    } from "./graphUtils";
    import NodeTooltip from "./NodeTooltip.svelte";
    // TokenDropdown import removed — fork modal disabled
    import type { StoredGraph } from "./types";
    import ViewControls from "./ViewControls.svelte";
    import { useZoomPan } from "../../lib/useZoomPan.svelte";
    import ZoomControls from "../../lib/ZoomControls.svelte";

    // Layout constants
    const COMPONENT_SIZE = 6;
    const HIT_AREA_PADDING = 4;
    const PRED_ROW_HEIGHT = 36;
    const PRED_ROW_GAP = 6;
    const BASE_MARGIN_TOP = 60;
    const LABEL_WIDTH = 100;
    const CLUSTER_BAR_HEIGHT = 3;
    const CLUSTER_BAR_GAP = 2;
    const LAYER_X_OFFSET = 3; // Horizontal offset per layer to avoid edge overlap

    import {
        EVAL_PGD_N_STEPS,
        EVAL_PGD_STEP_SIZE,
        type InterventionState,
        type InterventionRun,
        type InterventionResult,
        type TokenPrediction,
    } from "../../lib/interventionTypes";

    type Props = {
        graph: StoredGraph;
        interventionState: InterventionState;
        tokens: string[];
        // View settings (shared with main graph)
        topK: number;
        componentGap: number;
        layerGap: number;
        normalizeEdges: NormalizeType;
        ciThreshold: Loadable<number>;
        hideUnpinnedEdges: boolean;
        hideNodeCard: boolean;
        onTopKChange: (value: number) => void;
        onComponentGapChange: (value: number) => void;
        onLayerGapChange: (value: number) => void;
        onNormalizeChange: (value: NormalizeType) => void;
        onCiThresholdChange: (value: number) => void;
        onHideUnpinnedEdgesChange: (value: boolean) => void;
        onHideNodeCardChange: (value: boolean) => void;
        // Actions
        runningIntervention: boolean;
        generatingSubgraph: boolean;
        onSelectionChange: (selection: Set<string>) => void;
        onForwardDraft: (advPgd: { n_steps: number; step_size: number }) => void;
        onCloneRun: () => void;
        onSelectVersion: (index: number) => void;
        onDeleteRun: (runId: number) => void;
        onGenerateGraphFromSelection: () => void;
        onHoveredNodeChange?: (node: { layer: string; seqIdx: number; cIdx: number } | null) => void;
    };

    let {
        graph,
        interventionState,
        tokens,
        topK,
        componentGap,
        layerGap,
        normalizeEdges,
        ciThreshold,
        hideUnpinnedEdges,
        hideNodeCard,
        onTopKChange,
        onComponentGapChange,
        onLayerGapChange,
        onNormalizeChange,
        onCiThresholdChange,
        onHideUnpinnedEdgesChange,
        onHideNodeCardChange,
        runningIntervention,
        generatingSubgraph,
        onSelectionChange,
        onForwardDraft,
        onCloneRun,
        onSelectVersion,
        onDeleteRun,
        onGenerateGraphFromSelection,
        onHoveredNodeChange,
    }: Props = $props();

    // Derived: active run and editability
    const activeRun = $derived(interventionState.runs[interventionState.activeIndex]);
    const isEditable = $derived(activeRun.kind === "draft");
    // All interventable nodes = the base baked run's selection (single source of truth)
    const allInterventableNodes = $derived.by(() => {
        const baseRun = interventionState.runs[0];
        if (baseRun.kind !== "baked") throw new Error("First intervention run must be a baked base run");
        return baseRun.selectedNodes;
    });
    const effectiveSelection = $derived(activeRun.selectedNodes);

    let advPgdNSteps = $state(EVAL_PGD_N_STEPS);
    let advPgdStepSize = $state(EVAL_PGD_STEP_SIZE);

    // Intervention result from baked run, null for draft
    const interventionResult = $derived.by((): InterventionResult | null => {
        if (activeRun.kind === "baked") return activeRun.result;
        return null;
    });

    // Prediction rows for rendering: [{label, preds, labelPred}] ordered top-to-bottom (Adv, Stoch, CI)
    type PredRow = { label: string; preds: TokenPrediction[][]; labelPred: TokenPrediction | null };
    const predRows = $derived.by((): PredRow[] | null => {
        if (!interventionResult) return null;
        const lbl = interventionResult.label;
        const rows: PredRow[] = [];
        if (interventionResult.adversarial.length > 0)
            rows.push({ label: "Adv", preds: interventionResult.adversarial, labelPred: lbl?.adversarial ?? null });
        if (interventionResult.stochastic.length > 0)
            rows.push({ label: "Stoch", preds: interventionResult.stochastic, labelPred: lbl?.stochastic ?? null });
        rows.push({ label: "CI", preds: interventionResult.ci, labelPred: lbl?.ci ?? null });
        if (interventionResult.ablated && interventionResult.ablated.length > 0)
            rows.push({ label: "T\\S", preds: interventionResult.ablated, labelPred: lbl?.ablated ?? null });
        return rows;
    });

    const predRowCount = $derived(predRows ? predRows.length : 0);
    const PRED_AREA_HEIGHT = $derived(
        predRowCount > 0 ? predRowCount * (PRED_ROW_HEIGHT + PRED_ROW_GAP) + PRED_ROW_GAP : 0,
    );

    const MARGIN = $derived({
        top: BASE_MARGIN_TOP,
        right: 40,
        bottom: 20,
        left: 20,
    });

    // Optimization target highlight
    const optimizationTarget = $derived.by(() => {
        const opt = graph.data.optimization;
        if (!opt) return null;
        return {
            position: opt.loss.position,
            label: opt.loss.type === "ce" ? opt.loss.label_str : null,
        };
    });

    // Version list identity key
    function runIdentityKey(run: InterventionRun, index: number): string {
        switch (run.kind) {
            case "draft":
                return `draft-${index}`;
            case "baked":
                return `baked-${run.id}`;
        }
    }

    // Hover state for composer
    let hoveredNode = $state<{ layer: string; seqIdx: number; cIdx: number } | null>(null);
    let hoveredBarClusterId = $state<number | null>(null);
    let isHoveringTooltip = $state(false);
    let tooltipPos = $state<TooltipPos>({ left: 0, top: 0 });
    let hoverTimeout: ReturnType<typeof setTimeout> | null = null;

    $effect(() => {
        onHoveredNodeChange?.(hoveredNode);
    });

    // Hover state for prediction chips
    type HoveredPred = { pred: TokenPrediction; rowLabel: string; seqIdx: number };
    let hoveredPred = $state<HoveredPred | null>(null);
    let predTooltipPos = $state<TooltipPos>({ left: 0, top: 0 });

    function handlePredMouseEnter(e: MouseEvent, pred: TokenPrediction, rowLabel: string, seqIdx: number) {
        hoveredPred = { pred, rowLabel, seqIdx };
        predTooltipPos = calcTooltipPos(e.clientX, e.clientY, "small");
    }

    function handlePredMouseLeave() {
        hoveredPred = null;
    }

    // Refs
    let graphContainer: HTMLDivElement;

    // Zoom/pan
    const zoom = useZoomPan(() => graphContainer);

    // Get cluster ID of hovered node or bar (for cluster-wide rotation effect)
    const hoveredClusterId = $derived.by(() => {
        if (hoveredBarClusterId !== null) return hoveredBarClusterId;
        if (!hoveredNode) return undefined;
        return runState.getClusterId(hoveredNode.layer, hoveredNode.cIdx);
    });

    // Check if a node is in the same cluster as the hovered node (for cluster rotation effect)
    function isNodeInSameCluster(nodeKey: string): boolean {
        // Only trigger if hovered node has a numeric cluster ID (not singleton/no mapping)
        if (hoveredClusterId === undefined || hoveredClusterId === null) return false;
        const [layer, , cIdxStr] = nodeKey.split(":");
        const cIdx = parseInt(cIdxStr);
        const nodeClusterId = runState.getClusterId(layer, cIdx);
        return nodeClusterId === hoveredClusterId;
    }

    // Check if a node matches the hovered component (same layer:cIdx across any seqIdx)
    function nodeMatchesHoveredComponent(nodeKey: string): boolean {
        if (!hoveredNode) return false;
        const [layer, , cIdxStr] = nodeKey.split(":");
        const cIdx = parseInt(cIdxStr);
        return layer === hoveredNode.layer && cIdx === hoveredNode.cIdx;
    }

    // Drag-to-select state
    let isDragging = $state(false);
    let dragStart = $state<{ x: number; y: number } | null>(null);
    let dragCurrent = $state<{ x: number; y: number } | null>(null);
    let svgElement: SVGSVGElement | null = null;

    function getRowKey(layer: string): string {
        return _getRowKey(layer);
    }

    // All nodes from the graph (for rendering)
    const allNodes = $derived(new SvelteSet(Object.keys(graph.data.nodeCiVals)));

    // Select active edge variant
    const active = $derived(getActiveEdges(graph.data, displaySettings.edgeVariant));
    const activeEdges = $derived(active.edges);
    const activeEdgesBySource = $derived(active.bySource);
    const activeEdgesByTarget = $derived(active.byTarget);

    // Filter edges for rendering (topK by magnitude, optionally hide edges not connected to selected nodes).
    // Edges arrive pre-sorted by abs(val) desc from backend, so filter preserves order and we just slice.
    const filteredEdges = $derived.by(() => {
        let edges = activeEdges;
        if (hideUnpinnedEdges && effectiveSelection.size > 0) {
            edges = edges.filter((e) => effectiveSelection.has(e.src) || effectiveSelection.has(e.tgt));
        }
        return edges.slice(0, topK);
    });

    // Edge count for ViewControls
    const filteredEdgeCount = $derived(filteredEdges.length);

    // Compute layout for composer
    const layout = $derived.by(() => {
        const nodesPerLayerSeq: Record<string, number[]> = {};
        const allLayers = new SvelteSet<string>();
        const allRows = new SvelteSet<string>();

        for (const nodeKey of allNodes) {
            const [layer, seqIdx, cIdx] = nodeKey.split(":");
            allLayers.add(layer);
            allRows.add(getRowKey(layer));
            const key = `${layer}:${seqIdx}`;
            if (!nodesPerLayerSeq[key]) nodesPerLayerSeq[key] = [];
            nodesPerLayerSeq[key].push(+cIdx);
        }

        // Sort rows
        const rows = sortRows(Array.from(allRows));

        const numTokens = tokens.length;

        // Calculate column widths
        const maxComponentsPerSeq = Array.from({ length: numTokens }, (_, seqIdx) => {
            let maxAtSeq = 1;
            for (const row of rows) {
                let totalInRow = 0;
                for (const layer of allLayers) {
                    if (getRowKey(layer) === row) {
                        totalInRow += (nodesPerLayerSeq[`${layer}:${seqIdx}`] ?? []).length;
                    }
                }
                const rowParts = row.split(".");
                const isGroupedRow = rowParts.length >= 3 && rowParts[2].includes("_");
                if (isGroupedRow) {
                    const groupProjs = getGroupProjections(rowParts[1]);
                    if (groupProjs && groupProjs.length > 1) {
                        totalInRow += groupProjs.length - 1;
                    }
                }
                maxAtSeq = Math.max(maxAtSeq, totalInRow);
            }
            return maxAtSeq;
        });

        const COL_PADDING = 12;
        const MIN_COL_WIDTH = 60;
        const seqWidths = maxComponentsPerSeq.map((n) =>
            Math.max(MIN_COL_WIDTH, n * (COMPONENT_SIZE + componentGap) + COL_PADDING * 2),
        );
        const seqXStarts = [MARGIN.left];
        for (let i = 0; i < seqWidths.length - 1; i++) {
            seqXStarts.push(seqXStarts[i] + seqWidths[i]);
        }

        // Assign Y positions (output at top, wte at bottom)
        const rowYPositions: Record<string, number> = {};
        for (let i = 0; i < rows.length; i++) {
            const distanceFromEnd = rows.length - 1 - i;
            rowYPositions[rows[i]] = MARGIN.top + distanceFromEnd * (COMPONENT_SIZE + layerGap);
        }

        // Map each layer to its row's Y position and X offset
        // X offset: output (last row) at center, others alternate +/- based on distance
        const layerYPositions: Record<string, number> = {};
        const layerXOffsets: Record<string, number> = {};
        for (const layer of allLayers) {
            const rowKey = getRowKey(layer);
            layerYPositions[layer] = rowYPositions[rowKey];
            const rowIdx = rows.indexOf(rowKey);
            const distanceFromOutput = rows.length - 1 - rowIdx;
            if (distanceFromOutput === 0 || layer === "embed") {
                layerXOffsets[layer] = 0;
            } else {
                layerXOffsets[layer] = distanceFromOutput % 2 === 1 ? LAYER_X_OFFSET : -LAYER_X_OFFSET;
            }
        }

        // Position nodes and compute cluster spans
        const nodePositions: Record<string, NodePosition> = {};
        const allClusterSpans: ClusterSpan[] = [];
        for (const layer of allLayers) {
            const info = parseLayer(layer);
            const groupProjs = info.sublayer ? getGroupProjections(info.sublayer) : null;
            const isGrouped = groupProjs !== null && info.projection !== null && groupProjs.includes(info.projection);

            for (let seqIdx = 0; seqIdx < numTokens; seqIdx++) {
                const layerNodes = nodesPerLayerSeq[`${layer}:${seqIdx}`];
                if (!layerNodes) continue;

                let baseX = seqXStarts[seqIdx] + COL_PADDING + layerXOffsets[layer];
                const baseY = layerYPositions[layer];

                if (isGrouped && groupProjs && info.projection) {
                    const projIdx = groupProjs.indexOf(info.projection);
                    for (let i = 0; i < projIdx; i++) {
                        const prevLayer = buildLayerAddress(info.block, info.sublayer, groupProjs[i]);
                        baseX +=
                            (nodesPerLayerSeq[`${prevLayer}:${seqIdx}`]?.length ?? 0) * (COMPONENT_SIZE + componentGap);
                        baseX += COMPONENT_SIZE + componentGap;
                    }
                }

                // Output nodes always sort by probability; internal nodes sort by cluster if mapping loaded, else by CI
                const sorted =
                    layer === "output" || !runState.clusterMapping
                        ? sortComponentsByImportance(
                              layerNodes,
                              layer,
                              seqIdx,
                              graph.data.nodeCiVals,
                              graph.data.outputProbs,
                          )
                        : sortComponentsByCluster(
                              layerNodes,
                              layer,
                              seqIdx,
                              graph.data.nodeCiVals,
                              runState.getClusterId,
                          );
                const offsets = computeComponentOffsets(sorted, COMPONENT_SIZE, componentGap);
                for (const cIdx of layerNodes) {
                    nodePositions[`${layer}:${seqIdx}:${cIdx}`] = {
                        x: baseX + offsets[cIdx] + COMPONENT_SIZE / 2,
                        y: baseY + COMPONENT_SIZE / 2,
                    };
                }

                // Compute cluster spans for this layer/seqIdx (skip output layer)
                if (layer !== "output" && runState.clusterMapping) {
                    const spans = computeClusterSpans(
                        sorted,
                        layer,
                        seqIdx,
                        baseX,
                        baseY,
                        COMPONENT_SIZE,
                        offsets,
                        runState.getClusterId,
                    );
                    allClusterSpans.push(...spans);
                }
            }
        }

        const totalSeqWidth = seqXStarts[seqXStarts.length - 1] + seqWidths[seqWidths.length - 1];
        const width = totalSeqWidth + MARGIN.right;
        const maxY = rows.length > 0 ? Math.max(...Object.values(layerYPositions)) + COMPONENT_SIZE : MARGIN.top;
        const height = maxY + MARGIN.bottom + 40;

        return {
            nodePositions,
            layerYPositions,
            seqWidths,
            seqXStarts,
            width,
            height,
            nodesPerLayerSeq,
            allLayers,
            rows,
            clusterSpans: allClusterSpans,
        };
    });

    // Derived values
    const maxAbsAttr = $derived(active.maxAbsAttr);
    const selectedCount = $derived(effectiveSelection.size);
    const interventableCount = $derived(allInterventableNodes.size);

    // Selection helpers
    function isNodeSelected(nodeKey: string): boolean {
        return effectiveSelection.has(nodeKey);
    }

    function toggleNode(nodeKey: string) {
        if (!isEditable) return;
        if (!isInterventableNode(nodeKey)) return;
        const newSelection = new SvelteSet(effectiveSelection);
        if (newSelection.has(nodeKey)) {
            newSelection.delete(nodeKey);
        } else {
            newSelection.add(nodeKey);
        }
        onSelectionChange(newSelection);
    }

    function selectAll() {
        if (!isEditable) return;
        onSelectionChange(new SvelteSet(allInterventableNodes));
    }

    function selectAllKV() {
        if (!isEditable) return;
        const kvNodes = new SvelteSet<string>();
        for (const nodeKey of allInterventableNodes) {
            const layer = nodeKey.split(":")[0];
            if (layer.endsWith("k_proj") || layer.endsWith("v_proj")) {
                kvNodes.add(nodeKey);
            }
        }
        onSelectionChange(kvNodes);
    }

    function clearSelection() {
        if (!isEditable) return;
        onSelectionChange(new SvelteSet());
    }

    // Hover handlers
    function handleNodeMouseEnter(event: MouseEvent, layer: string, seqIdx: number, cIdx: number) {
        if (isDragging) return;
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
            hoverTimeout = null;
        }
        hoveredNode = { layer, seqIdx, cIdx };
        const size = layer === "embed" || layer === "output" ? "small" : "large";
        tooltipPos = calcTooltipPos(event.clientX, event.clientY, size);
    }

    function handleNodeMouseLeave() {
        if (hoverTimeout) clearTimeout(hoverTimeout);
        hoverTimeout = setTimeout(() => {
            if (!isHoveringTooltip) hoveredNode = null;
            hoverTimeout = null;
        }, 100);
    }

    function handleNodeClick(nodeKey: string) {
        toggleNode(nodeKey);
        hoveredNode = null;
    }

    // Drag-to-select handlers
    // Converts mouse event to SVG logical coordinates (accounting for zoom transform)
    function getSvgPoint(event: MouseEvent): { x: number; y: number } | null {
        if (!svgElement) return null;
        const container = svgElement.parentElement!;
        const rect = container.getBoundingClientRect();
        // Get container-relative position (with scroll offset)
        const containerX = event.clientX - rect.left + container.scrollLeft;
        const containerY = event.clientY - rect.top + container.scrollTop;
        // Convert to logical SVG coordinates by reversing the zoom transform
        return {
            x: (containerX - zoom.translateX) / zoom.scale,
            y: (containerY - zoom.translateY) / zoom.scale,
        };
    }

    function handlePanStart(event: MouseEvent) {
        // Pan on shift+left-click or middle-click
        if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
            zoom.startPan(event);
            return;
        }
        // Otherwise handle drag-to-select
        if (event.button === 0 && !event.shiftKey) {
            handleSvgMouseDown(event);
        }
    }

    function handleSvgMouseDown(event: MouseEvent) {
        if (!isEditable) return;
        const target = event.target as Element;
        if (target.closest(".node-group")) return;

        event.preventDefault();
        const point = getSvgPoint(event);
        if (!point) return;

        hoveredNode = null;
        isDragging = true;
        dragStart = point;
        dragCurrent = point;
    }

    function handleSvgMouseMove(event: MouseEvent) {
        if (!isDragging) return;
        dragCurrent = getSvgPoint(event);
    }

    function handleSvgMouseUp() {
        if (!isDragging || !dragStart || !dragCurrent) {
            isDragging = false;
            dragStart = null;
            dragCurrent = null;
            return;
        }

        // Calculate selection rectangle bounds
        const minX = Math.min(dragStart.x, dragCurrent.x);
        const maxX = Math.max(dragStart.x, dragCurrent.x);
        const minY = Math.min(dragStart.y, dragCurrent.y);
        const maxY = Math.max(dragStart.y, dragCurrent.y);

        // Only select if drag was meaningful (more than a few pixels)
        const dragDistance = Math.sqrt((maxX - minX) ** 2 + (maxY - minY) ** 2);
        if (dragDistance > 5) {
            // Find nodes within the selection rectangle
            const nodesToToggle: string[] = [];
            for (const nodeKey of allInterventableNodes) {
                const pos = layout.nodePositions[nodeKey];
                if (!pos) continue;

                // Check if node center is within selection rect
                if (pos.x >= minX && pos.x <= maxX && pos.y >= minY && pos.y <= maxY) {
                    nodesToToggle.push(nodeKey);
                }
            }

            // Toggle selection for nodes in rect
            if (nodesToToggle.length > 0) {
                const newSelection = new SvelteSet(effectiveSelection);
                for (const nodeKey of nodesToToggle) {
                    if (newSelection.has(nodeKey)) {
                        newSelection.delete(nodeKey);
                    } else {
                        newSelection.add(nodeKey);
                    }
                }
                onSelectionChange(newSelection);
            }
        }

        isDragging = false;
        dragStart = null;
        dragCurrent = null;
    }

    // Derived selection rectangle for rendering
    const selectionRect = $derived.by(() => {
        if (!isDragging || !dragStart || !dragCurrent) return null;
        return {
            x: Math.min(dragStart.x, dragCurrent.x),
            y: Math.min(dragStart.y, dragCurrent.y),
            width: Math.abs(dragCurrent.x - dragStart.x),
            height: Math.abs(dragCurrent.y - dragStart.y),
        };
    });

    // Edge rendering
    function getEdgePath(src: string, tgt: string): string {
        const srcPos = layout.nodePositions[src];
        const tgtPos = layout.nodePositions[tgt];
        if (!srcPos || !tgtPos) return "";

        const midY = (srcPos.y + tgtPos.y) / 2;
        return `M ${srcPos.x} ${srcPos.y} C ${srcPos.x} ${midY}, ${tgtPos.x} ${midY}, ${tgtPos.x} ${tgtPos.y}`;
    }

    function getEdgeOpacity(val: number): number {
        const normalized = Math.abs(val) / maxAbsAttr;
        return lerp(0.1, 0.8, Math.sqrt(normalized));
    }

    function getEdgeWidth(val: number): number {
        const normalized = Math.abs(val) / maxAbsAttr;
        return lerp(0.5, 3, Math.sqrt(normalized));
    }

    // Run history helpers
    function formatTime(timestamp: string): string {
        return new Date(timestamp).toLocaleTimeString();
    }

    function getRowLabel(layer: string): string {
        return _getRowLabel(layer);
    }
</script>

<div class="interventions-view">
    <!-- Composer Graph (Left) -->
    <div class="composer-graph">
        <!-- Shared view controls -->
        <ViewControls
            {topK}
            {componentGap}
            {layerGap}
            {filteredEdgeCount}
            {normalizeEdges}
            {ciThreshold}
            {hideUnpinnedEdges}
            {hideNodeCard}
            {onTopKChange}
            {onComponentGapChange}
            {onLayerGapChange}
            {onNormalizeChange}
            {onCiThresholdChange}
            {onHideUnpinnedEdgesChange}
            {onHideNodeCardChange}
        />

        <!-- Intervention controls -->
        <div class="intervention-controls">
            <span class="node-count"
                >{selectedCount} / {interventableCount} nodes{#if !isEditable}&nbsp;(read-only){/if}</span
            >
            <span
                class="info-icon"
                data-tooltip="NOTE: Biases in each layer that have them are always active, regardless of which components are selected"
                >?</span
            >
            <div class="button-group">
                {#if isEditable}
                    <button onclick={selectAll}>Select All</button>
                    <button onclick={selectAllKV}>Select All K/V</button>
                    <button onclick={clearSelection}>Clear</button>
                    <button
                        class="generate-btn"
                        onclick={onGenerateGraphFromSelection}
                        disabled={generatingSubgraph ||
                            selectedCount === 0 ||
                            (interventableCount > 0 && selectedCount === interventableCount)}
                        title={selectedCount === 0
                            ? "Select components to include in subgraph"
                            : "Generate a subgraph showing only attributions between selected components"}
                    >
                        {generatingSubgraph ? "Generating..." : "Generate subgraph"}
                    </button>
                    <button
                        class="run-btn"
                        onclick={() => onForwardDraft({ n_steps: advPgdNSteps, step_size: advPgdStepSize })}
                        disabled={runningIntervention || selectedCount === 0}
                    >
                        {runningIntervention ? "Forwarding..." : "Forward"}
                    </button>
                    <span class="pgd-inputs">
                        <label>PGD steps <input type="number" min="0" max="50" bind:value={advPgdNSteps} /></label>
                        <label
                            >step size <input
                                type="number"
                                min="0"
                                max="10"
                                step="0.1"
                                bind:value={advPgdStepSize}
                            /></label
                        >
                    </span>
                {:else}
                    <button class="run-btn" onclick={onCloneRun}>Clone</button>
                {/if}
            </div>
        </div>

        <!-- Graph wrapper for sticky layout -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
            class="graph-wrapper"
            class:panning={zoom.isPanning}
            onmousedown={handlePanStart}
            onmousemove={zoom.updatePan}
            onmouseup={zoom.endPan}
            onmouseleave={zoom.endPan}
        >
            <!-- Sticky layer labels (left) -->
            <div class="layer-labels-container" style="width: {LABEL_WIDTH}px; padding-top: {PRED_AREA_HEIGHT}px;">
                <svg
                    width={LABEL_WIDTH}
                    height={layout.height * zoom.scale + Math.max(zoom.translateY, 0)}
                    style="display: block;"
                >
                    <g transform="translate(0, {zoom.translateY}) scale(1, {zoom.scale})">
                        {#each Object.entries(layout.layerYPositions) as [layer, y] (layer)}
                            {@const yCenter = y + COMPONENT_SIZE / 2}
                            <text
                                x={LABEL_WIDTH - 10}
                                y={yCenter}
                                text-anchor="end"
                                dominant-baseline="middle"
                                font-size="10"
                                font-weight="500"
                                font-family="'Berkeley Mono', 'SF Mono', monospace"
                                fill={colors.textSecondary}>{getRowLabel(layer)}</text
                            >
                        {/each}
                    </g>
                </svg>
            </div>

            <!-- Scrollable graph area -->
            <div class="graph-container" bind:this={graphContainer}>
                <!-- Sticky prediction rows (top) -->
                {#if predRows && PRED_AREA_HEIGHT > 0}
                    <div class="pred-rows-container" style="height: {PRED_AREA_HEIGHT}px;">
                        <svg
                            width={layout.width * zoom.scale + Math.max(zoom.translateX, 0)}
                            height={PRED_AREA_HEIGHT}
                            style="display: block;"
                        >
                            <g transform="translate({zoom.translateX}, 0) scale({zoom.scale}, 1)">
                                {#each predRows as row, rowIdx (`pred-${row.label}`)}
                                    {@const rowY = rowIdx * (PRED_ROW_HEIGHT + PRED_ROW_GAP) + PRED_ROW_GAP}
                                    <!-- Row label -->
                                    <text
                                        x={layout.seqXStarts[0] - 4}
                                        y={rowY + PRED_ROW_HEIGHT / 2 + 3}
                                        text-anchor="end"
                                        font-size="9"
                                        font-weight="500"
                                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                                        fill={colors.textMuted}>{row.label}</text
                                    >
                                    <!-- Predictions per position -->
                                    {#each row.preds as preds, seqIdx (seqIdx)}
                                        {@const colX = layout.seqXStarts[seqIdx]}
                                        {@const colW = layout.seqWidths[seqIdx]}
                                        {@const chipW = 48}
                                        {@const chipH = PRED_ROW_HEIGHT}
                                        {@const chipGap = 1}
                                        {@const isLabelPos =
                                            interventionResult?.label != null &&
                                            seqIdx === interventionResult.label.position}
                                        {@const labelTokenId = isLabelPos ? (row.labelPred?.token_id ?? null) : null}
                                        {@const labelInTopk =
                                            labelTokenId != null && preds.some((p) => p.token_id === labelTokenId)}
                                        {@const maxChips = Math.min(
                                            preds.length,
                                            Math.max(1, Math.floor((colW - 2 + chipGap) / (chipW + chipGap))),
                                        )}
                                        {#each preds.slice(0, maxChips) as pred, rank (rank)}
                                            {@const cx = colX + rank * (chipW + chipGap)}
                                            {@const isLabel = labelTokenId != null && pred.token_id === labelTokenId}
                                            <!-- svelte-ignore a11y_no_static_element_interactions -->
                                            <g
                                                onmouseenter={(e) => handlePredMouseEnter(e, pred, row.label, seqIdx)}
                                                onmouseleave={handlePredMouseLeave}
                                            >
                                                <rect
                                                    x={cx}
                                                    y={rowY}
                                                    width={chipW}
                                                    height={chipH}
                                                    rx="2"
                                                    fill={getNextTokenProbBgColor(pred.prob)}
                                                    stroke={isLabel ? "#f59e0b" : "#ddd"}
                                                    stroke-width={isLabel ? "1.5" : "0.5"}
                                                />
                                                <text
                                                    x={cx + chipW / 2}
                                                    y={rowY + chipH / 2 + 3}
                                                    text-anchor="middle"
                                                    font-size="7"
                                                    font-family="'Berkeley Mono', 'SF Mono', monospace"
                                                    fill={pred.prob > 0.5 ? "white" : colors.textPrimary}
                                                    >{pred.token}</text
                                                >
                                            </g>
                                        {/each}
                                        <!-- Label token chip (when not in topk) -->
                                        {#if isLabelPos && !labelInTopk && row.labelPred}
                                            {@const cx = colX + maxChips * (chipW + chipGap) + chipGap}
                                            <!-- svelte-ignore a11y_no_static_element_interactions -->
                                            <g
                                                onmouseenter={(e) =>
                                                    handlePredMouseEnter(e, row.labelPred!, row.label, seqIdx)}
                                                onmouseleave={handlePredMouseLeave}
                                            >
                                                <rect
                                                    x={cx}
                                                    y={rowY}
                                                    width={chipW}
                                                    height={chipH}
                                                    rx="2"
                                                    fill={getNextTokenProbBgColor(row.labelPred.prob)}
                                                    stroke="#f59e0b"
                                                    stroke-width="1.5"
                                                    stroke-dasharray="3,2"
                                                />
                                                <text
                                                    x={cx + chipW / 2}
                                                    y={rowY + chipH / 2 + 3}
                                                    text-anchor="middle"
                                                    font-size="7"
                                                    font-family="'Berkeley Mono', 'SF Mono', monospace"
                                                    fill={row.labelPred.prob > 0.5 ? "white" : colors.textPrimary}
                                                    >{row.labelPred.token}</text
                                                >
                                            </g>
                                        {/if}
                                    {/each}
                                {/each}
                            </g>
                        </svg>
                    </div>
                {/if}

                <svg
                    bind:this={svgElement}
                    class:readonly={!isEditable}
                    width={layout.width * zoom.scale + Math.max(zoom.translateX, 0)}
                    height={layout.height * zoom.scale + Math.max(zoom.translateY, 0)}
                    style="display: block;"
                    onmousemove={handleSvgMouseMove}
                    onmouseup={handleSvgMouseUp}
                    onmouseleave={handleSvgMouseUp}
                >
                    <g transform="translate({zoom.translateX}, {zoom.translateY}) scale({zoom.scale})">
                        <!-- Edges -->
                        <g class="edges-layer" opacity="0.6">
                            {#each filteredEdges as edge (`${edge.src}-${edge.tgt}`)}
                                {@const path = getEdgePath(edge.src, edge.tgt)}
                                {#if path}
                                    <path
                                        d={path}
                                        stroke={getEdgeColor(edge.val)}
                                        stroke-width={getEdgeWidth(edge.val)}
                                        fill="none"
                                        opacity={getEdgeOpacity(edge.val)}
                                    />
                                {/if}
                            {/each}
                        </g>

                        <!-- Optimization target highlight -->
                        {#if optimizationTarget}
                            {@const pos = optimizationTarget.position}
                            {@const xStart = layout.seqXStarts[pos]}
                            {@const width = layout.seqWidths[pos]}
                            <rect
                                x={xStart}
                                y={MARGIN.top - 10}
                                {width}
                                height={layout.height - MARGIN.top - MARGIN.bottom + 20}
                                fill="none"
                                stroke={colors.accent}
                                stroke-width="1.5"
                                stroke-dasharray="6 3"
                                opacity="0.4"
                            />
                        {/if}

                        <!-- Cluster bars (below nodes) -->
                        <g class="cluster-bars-layer">
                            {#each layout.clusterSpans as span (`${span.layer}:${span.seqIdx}:${span.clusterId}`)}
                                {@const isHighlighted = hoveredClusterId === span.clusterId}
                                <!-- svelte-ignore a11y_no_static_element_interactions -->
                                <rect
                                    class="cluster-bar"
                                    class:highlighted={isHighlighted}
                                    x={span.xStart}
                                    y={span.y + CLUSTER_BAR_GAP}
                                    width={span.xEnd - span.xStart}
                                    height={CLUSTER_BAR_HEIGHT}
                                    rx="1"
                                    onmouseenter={() => (hoveredBarClusterId = span.clusterId)}
                                    onmouseleave={() => (hoveredBarClusterId = null)}
                                />
                            {/each}
                        </g>

                        <!-- Nodes -->
                        <g class="nodes-layer">
                            {#each allNodes as nodeKey (nodeKey)}
                                {@const pos = layout.nodePositions[nodeKey]}
                                {@const [layer, seqIdx, cIdx] = nodeKey.split(":")}
                                {@const interventable = isInterventableNode(nodeKey)}
                                {@const selected = interventable && isNodeSelected(nodeKey)}
                                {@const inSameCluster = isNodeInSameCluster(nodeKey)}
                                {@const isHoveredComponent = nodeMatchesHoveredComponent(nodeKey)}
                                {@const isDimmed =
                                    (hoveredNode !== null || hoveredBarClusterId !== null) &&
                                    !isHoveredComponent &&
                                    !inSameCluster &&
                                    !selected}
                                {#if pos}
                                    <g
                                        class="node-group"
                                        class:selected
                                        class:non-interventable={!interventable}
                                        onmouseenter={(e) => handleNodeMouseEnter(e, layer, +seqIdx, +cIdx)}
                                        onmouseleave={handleNodeMouseLeave}
                                        onclick={() => handleNodeClick(nodeKey)}
                                    >
                                        <rect
                                            x={pos.x - COMPONENT_SIZE / 2 - HIT_AREA_PADDING}
                                            y={pos.y - COMPONENT_SIZE / 2 - HIT_AREA_PADDING}
                                            width={COMPONENT_SIZE + HIT_AREA_PADDING * 2}
                                            height={COMPONENT_SIZE + HIT_AREA_PADDING * 2}
                                            fill="transparent"
                                        />
                                        <rect
                                            class="node"
                                            class:cluster-hovered={inSameCluster}
                                            class:dimmed={isDimmed}
                                            x={pos.x - COMPONENT_SIZE / 2}
                                            y={pos.y - COMPONENT_SIZE / 2}
                                            width={COMPONENT_SIZE}
                                            height={COMPONENT_SIZE}
                                            fill={!interventable
                                                ? colors.textMuted
                                                : selected
                                                  ? colors.accent
                                                  : colors.nodeDefault}
                                            stroke={selected ? colors.accent : "none"}
                                            stroke-width={selected ? 2 : 0}
                                            rx="1"
                                            opacity={!interventable ? 0.3 : selected ? 1 : 0.4}
                                        />
                                    </g>
                                {/if}
                            {/each}
                        </g>

                        <!-- Selection rectangle -->
                        {#if selectionRect}
                            <rect
                                class="selection-rect"
                                x={selectionRect.x}
                                y={selectionRect.y}
                                width={selectionRect.width}
                                height={selectionRect.height}
                                fill={rgbaToCss(colors.positiveRgb, 0.1)}
                                stroke={colors.accent}
                                stroke-width="1"
                                stroke-dasharray="4 2"
                            />
                        {/if}
                    </g>
                </svg>

                <!-- Sticky token labels (bottom) -->
                <div class="token-labels-container">
                    <svg
                        width={layout.width * zoom.scale + Math.max(zoom.translateX, 0)}
                        height="50"
                        style="display: block;"
                    >
                        <g transform="translate({zoom.translateX}, 0) scale({zoom.scale}, 1)">
                            {#each tokens as token, i (i)}
                                {@const colX = layout.seqXStarts[i]}
                                <text
                                    x={colX}
                                    y="20"
                                    text-anchor="start"
                                    font-size="11"
                                    font-family="'Berkeley Mono', 'SF Mono', monospace"
                                    font-weight="500"
                                    fill={colors.textPrimary}
                                    style="white-space: pre"
                                >
                                    {token}
                                </text>
                                <text
                                    x={colX}
                                    y="36"
                                    text-anchor="start"
                                    font-size="9"
                                    font-family="'Berkeley Mono', 'SF Mono', monospace"
                                    fill={colors.textMuted}>[{i}]</text
                                >
                                {#if optimizationTarget && i === optimizationTarget.position}
                                    <rect x={colX} y="42" width="40" height="3" fill={colors.accent} rx="1" />
                                {/if}
                            {/each}
                        </g>
                    </svg>
                </div>
            </div>

            <!-- Zoom controls in the bottom-left corner -->
            <div class="zoom-corner">
                <ZoomControls scale={zoom.scale} onZoomIn={zoom.zoomIn} onZoomOut={zoom.zoomOut} onReset={zoom.reset} />
            </div>
        </div>
    </div>

    <!-- Version List Panel (Right) -->
    <div class="history-panel">
        <div class="history-header">
            <span class="title">Versions</span>
            <span class="run-count">{interventionState.runs.length}</span>
        </div>

        <div class="runs-list">
            {#each interventionState.runs as run, index (runIdentityKey(run, index))}
                {@const isActive = index === interventionState.activeIndex}
                <div
                    class="run-card"
                    class:active={isActive}
                    role="button"
                    tabindex="0"
                    onclick={() => onSelectVersion(index)}
                    onkeydown={(e) => e.key === "Enter" && onSelectVersion(index)}
                >
                    {#if run.kind === "draft"}
                        <div class="run-header">
                            <span class="run-time draft-label">Draft</span>
                            <span class="run-nodes">{run.selectedNodes.size} nodes</span>
                            <span class="draft-hint">(not forwarded)</span>
                        </div>
                    {:else}
                        <div class="run-header">
                            <span class="run-time">{index === 0 ? "Base" : formatTime(run.createdAt)}</span>
                            <span class="run-nodes">{run.selectedNodes.size} nodes</span>
                            {#if index > 0}
                                <button
                                    class="delete-btn"
                                    onclick={(e) => {
                                        e.stopPropagation();
                                        onDeleteRun(run.id);
                                    }}>✕</button
                                >
                            {/if}
                        </div>
                        {#if isActive}
                            {@const opt = graph.data.optimization}
                            {@const lossLabel = opt
                                ? opt.loss.type === "ce"
                                    ? `CE "${opt.loss.label_str}" @ ${opt.loss.position}`
                                    : `KL @ ${opt.loss.position}`
                                : "mean KL"}
                            <div class="opt-info">
                                <div class="opt-row" title="Loss with binarised CI masking">
                                    <span class="opt-key">CI</span>
                                    <span>{run.result.ci_loss.toFixed(3)}</span>
                                </div>
                                <div class="opt-row" title="Loss with stochastic sources on 0-CI nodes">
                                    <span class="opt-key">stoch</span>
                                    <span>{run.result.stochastic_loss.toFixed(3)}</span>
                                </div>
                                <div
                                    class="opt-row"
                                    title="Loss using adversarially optimized sources on deselected-but-alive nodes, and stochastic sources on 0-CI nodes"
                                >
                                    <span class="opt-key">adv</span>
                                    <span>{run.result.adversarial_loss.toFixed(3)}</span>
                                </div>
                                {#if run.result.ablated_loss != null}
                                    <div
                                        class="opt-row"
                                        title="Loss with unselected nodes ablated from target model weights — measures sufficiency (lower = selected nodes are more sufficient)"
                                    >
                                        <span class="opt-key">T\S</span>
                                        <span>{run.result.ablated_loss.toFixed(3)}</span>
                                    </div>
                                {/if}
                                <div
                                    class="opt-row"
                                    title="The loss function used: mean KL (standard graphs) or the specific loss from optimization"
                                >
                                    <span class="opt-key">metric</span>
                                    <span>{lossLabel}</span>
                                </div>
                                {#if opt}
                                    <div class="opt-row" title="Total active components in the optimized circuit">
                                        <span class="opt-key">L0</span>
                                        <span>{opt.metrics.l0_total.toFixed(1)}</span>
                                    </div>
                                {/if}
                            </div>
                        {/if}
                    {/if}
                </div>
            {/each}
        </div>

        <div class="version-actions">
            <button class="clone-btn" onclick={onCloneRun}>Clone</button>
        </div>
    </div>

    <!-- Node tooltip -->
    {#if hoveredNode}
        <NodeTooltip
            {hoveredNode}
            {tooltipPos}
            {hideNodeCard}
            outputProbs={graph.data.outputProbs}
            nodeCiVals={graph.data.nodeCiVals}
            nodeSubcompActs={graph.data.nodeSubcompActs}
            {tokens}
            edgesBySource={activeEdgesBySource}
            edgesByTarget={activeEdgesByTarget}
            onMouseEnter={() => (isHoveringTooltip = true)}
            onMouseLeave={() => {
                isHoveringTooltip = false;
                handleNodeMouseLeave();
            }}
        />
    {/if}

    <!-- Prediction chip tooltip -->
    {#if hoveredPred}
        {@const p = hoveredPred.pred}
        {@const pos = predTooltipPos}
        <div
            class="pred-tooltip"
            style:left={pos.left != null ? `${pos.left}px` : undefined}
            style:right={pos.right != null ? `${pos.right}px` : undefined}
            style:top={pos.top != null ? `${pos.top}px` : undefined}
            style:bottom={pos.bottom != null ? `${pos.bottom}px` : undefined}
        >
            <div class="pred-tooltip-token">{p.token}</div>
            <table class="pred-tooltip-table">
                <tbody>
                    <tr><td>prob</td><td class="val">{(p.prob * 100).toFixed(1)}%</td></tr>
                    <tr><td>logit</td><td class="val">{p.logit.toFixed(2)}</td></tr>
                    <tr class="sep"><td>target prob</td><td class="val">{(p.target_prob * 100).toFixed(1)}%</td></tr>
                    <tr><td>target logit</td><td class="val">{p.target_logit.toFixed(2)}</td></tr>
                </tbody>
            </table>
        </div>
    {/if}

    <!-- Fork Modal (disabled — functionality commented out) -->
</div>

<style>
    .interventions-view {
        display: flex;
        flex: 1;
        min-height: 0;
        gap: var(--space-4);
    }

    /* Composer Graph */
    .composer-graph {
        display: flex;
        flex-direction: column;
        border: 1px solid var(--border-default);
        background: var(--bg-surface);
        overflow: hidden;
    }

    .intervention-controls {
        display: flex;
        align-items: center;
        gap: var(--space-4);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
        flex-shrink: 0;
    }

    .node-count {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .button-group {
        display: flex;
        gap: var(--space-2);
        margin-left: auto;
    }

    .button-group button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
        font-size: var(--text-sm);
    }

    .button-group button:hover:not(:disabled) {
        background: var(--bg-inset);
        border-color: var(--border-strong);
    }

    .pgd-inputs {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        font-size: 11px;
        color: var(--text-secondary);
    }

    .pgd-inputs label {
        display: flex;
        align-items: center;
        gap: var(--space-1);
        white-space: nowrap;
    }

    .pgd-inputs input {
        width: 42px;
        padding: 1px 4px;
        font-size: 11px;
        font-family: "Berkeley Mono", "SF Mono", monospace;
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: 3px;
        color: var(--text-primary);
    }

    .generate-btn {
        background: var(--status-info) !important;
        color: white !important;
        border-color: var(--status-info) !important;
    }

    .generate-btn:hover:not(:disabled) {
        filter: brightness(1.1);
    }

    .generate-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .run-btn {
        background: var(--accent-primary) !important;
        color: white !important;
        border-color: var(--accent-primary) !important;
    }

    .run-btn:hover:not(:disabled) {
        background: var(--accent-primary-dim) !important;
    }

    .run-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .graph-wrapper {
        display: flex;
        overflow: hidden;
        position: relative;
    }

    .zoom-corner {
        position: sticky;
        bottom: 0;
        left: 0;
        width: 0;
        height: 0;
        z-index: 12;
    }

    .zoom-corner :global(.zoom-controls) {
        position: absolute;
        bottom: var(--space-2);
        left: var(--space-2);
        top: auto;
        right: auto;
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
        overflow: auto;
        flex: 1;
        position: relative;
        background: var(--bg-inset);
    }

    .pred-rows-container {
        position: sticky;
        top: 0;
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
        z-index: 10;
    }

    .token-labels-container {
        position: sticky;
        bottom: 0;
        background: var(--bg-surface);
        border-top: 1px solid var(--border-default);
        z-index: 10;
    }

    .graph-container svg {
        cursor: crosshair;
    }

    .graph-container svg.readonly {
        cursor: default;
    }

    .node-group {
        cursor: pointer;
    }

    .node-group .node {
        transform-box: fill-box;
        transform-origin: center;
        transition:
            opacity var(--transition-fast),
            fill var(--transition-fast),
            transform var(--transition-normal);
    }

    .node-group .node.cluster-hovered {
        transform: rotate(45deg);
    }

    .node-group .node.dimmed {
        transform: scale(0.5);
    }

    .node-group:hover .node {
        opacity: 1 !important;
        filter: brightness(1.2);
    }

    .node-group.non-interventable {
        cursor: default;
    }

    .node-group.non-interventable:hover .node {
        filter: none;
    }

    .cluster-bar {
        fill: var(--text-secondary);
        opacity: 0.5;
        cursor: pointer;
        transition:
            opacity var(--transition-normal),
            fill var(--transition-normal);
    }

    .cluster-bar:hover,
    .cluster-bar.highlighted {
        fill: var(--text-primary);
        opacity: 0.8;
    }

    /* History Panel */
    .history-panel {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-width: 300px;
        max-width: 400px;
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        padding: var(--space-3);
    }

    .history-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-2);
        padding-bottom: var(--space-2);
        border-bottom: 1px solid var(--border-subtle);
    }

    .history-header .title {
        font-weight: 600;
        font-family: var(--font-sans);
        color: var(--text-primary);
    }

    .run-count {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .runs-list {
        flex: 1;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .run-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        padding: var(--space-2);
        cursor: pointer;
        transition: border-color var(--transition-fast);
    }

    .run-card:hover {
        border-color: var(--border-strong);
    }

    .run-card.active {
        border-color: var(--accent-primary);
        background: var(--bg-inset);
    }

    .run-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        margin-bottom: var(--space-2);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .run-time {
        color: var(--text-secondary);
    }

    .run-nodes {
        color: var(--text-muted);
        margin-left: auto;
    }

    .delete-btn {
        padding: 2px 6px;
        background: transparent;
        border: none;
        color: var(--text-muted);
        font-size: var(--text-xs);
        cursor: pointer;
    }

    .delete-btn:hover {
        color: var(--status-negative);
    }

    .draft-label {
        color: var(--status-info);
        font-weight: 600;
    }

    .draft-hint {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-style: italic;
    }

    .opt-info {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
        padding: var(--space-2);
        margin-top: var(--space-2);
        background: var(--bg-inset);
        border-radius: var(--radius-sm);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .opt-row {
        display: flex;
        gap: var(--space-2);
    }

    .opt-key {
        color: var(--text-muted);
        min-width: 60px;
        flex-shrink: 0;
    }

    .opt-key::after {
        content: ":";
    }

    .top-logits {
        margin-top: var(--space-1);
        padding-top: var(--space-1);
        border-top: 1px solid var(--border-subtle);
    }

    .top-logits-header {
        color: var(--text-muted);
        font-size: var(--text-xs);
    }

    .top-logit-row {
        display: flex;
        justify-content: space-between;
        gap: var(--space-2);
    }

    .top-logit-token {
        white-space: pre;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100px;
    }

    .top-logit-prob {
        flex-shrink: 0;
        text-align: right;
        color: var(--text-primary);
    }

    .version-actions {
        display: flex;
        gap: var(--space-2);
        padding-top: var(--space-2);
        border-top: 1px solid var(--border-subtle);
        margin-top: var(--space-2);
    }

    .clone-btn {
        flex: 1;
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
        font-size: var(--text-sm);
        cursor: pointer;
    }

    .clone-btn:hover {
        background: var(--bg-inset);
        border-color: var(--border-strong);
    }

    .pred-tooltip {
        position: fixed;
        z-index: 1000;
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-sm);
        box-shadow: var(--shadow-md);
        padding: var(--space-2) var(--space-3);
        pointer-events: none;
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        min-width: 140px;
    }

    .pred-tooltip-token {
        font-weight: 600;
        font-size: var(--text-sm);
        color: var(--text-primary);
        margin-bottom: var(--space-1);
        border-bottom: 1px solid var(--border-default);
        padding-bottom: var(--space-1);
    }

    .pred-tooltip-table {
        border-spacing: 0;
    }

    .pred-tooltip-table td {
        padding: 1px 0;
        color: var(--text-muted);
    }

    .pred-tooltip-table td.val {
        padding-left: var(--space-2);
        text-align: right;
        color: var(--text-secondary);
        font-variant-numeric: tabular-nums;
    }

    .pred-tooltip-table tr.sep td {
        padding-top: var(--space-1);
    }
</style>
