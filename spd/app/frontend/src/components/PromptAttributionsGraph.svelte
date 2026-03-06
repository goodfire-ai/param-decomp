<script lang="ts">
    import { getContext, untrack } from "svelte";
    import { SvelteSet, SvelteMap } from "svelte/reactivity";
    import type {
        GraphData,
        EdgeData,
        PinnedNode,
        HoveredNode,
        HoveredEdge,
        NodePosition,
    } from "../lib/promptAttributionsTypes";
    import { getActiveEdges } from "../lib/promptAttributionsTypes";
    import { colors, getEdgeColor, getSubcompActColor, rgbToCss, getNextTokenProbBgColor } from "../lib/colors";
    import { displaySettings } from "../lib/displaySettings.svelte";
    import {
        lerp,
        calcTooltipPos,
        sortComponentsByImportance,
        sortComponentsByCluster,
        computeComponentOffsets,
        computeClusterSpans,
        type ClusterSpan,
        type TooltipPos,
    } from "./prompt-attr/graphUtils";
    import NodeTooltip from "./prompt-attr/NodeTooltip.svelte";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import { useZoomPan } from "../lib/useZoomPan.svelte";
    import ZoomControls from "../lib/ZoomControls.svelte";
    import {
        parseLayer,
        getRowKey as _getRowKey,
        getRowLabel as _getRowLabel,
        sortRows,
        getGroupProjections,
        buildLayerAddress,
    } from "../lib/graphLayout";

    const runState = getContext<RunContext>(RUN_KEY);

    // Constants
    const COMPONENT_SIZE = 8;
    const HIT_AREA_PADDING = 4;
    const MARGIN = { top: 60, right: 40, bottom: 20, left: 20 };
    const LABEL_WIDTH = 100;
    const CLUSTER_BAR_HEIGHT = 3;
    const CLUSTER_BAR_GAP = 2;
    const LAYER_X_OFFSET = 3; // Horizontal offset per layer to avoid edge overlap

    type Props = {
        data: GraphData;
        tokenIds: number[];
        topK: number;
        componentGap: number;
        layerGap: number;
        hideUnpinnedEdges: boolean;
        hideNodeCard: boolean;
        stagedNodes: PinnedNode[];
        onStagedNodesChange: (nodes: PinnedNode[]) => void;
        onEdgeCountChange?: (count: number) => void;
        onHoveredNodeChange?: (node: HoveredNode | null) => void;
    };

    let {
        data,
        tokenIds,
        topK,
        componentGap,
        layerGap,
        hideUnpinnedEdges,
        hideNodeCard,
        stagedNodes,
        onStagedNodesChange,
        onEdgeCountChange,
        onHoveredNodeChange,
    }: Props = $props();

    // Compute masked prediction probability of self given previous position.
    // For token at position i, we look up outputProbs[(i-1):tokenIds[i]] - how well
    // position i-1 predicted this token. First token has no previous, so null.
    // NOTE: outputProbs only includes tokens with >=1% probability (backend threshold).
    // If the correct token isn't found, it means the masked model gave it <1% probability.
    const maskedSelfProbs = $derived.by(() => {
        const probs: (number | null)[] = [];
        for (let i = 0; i < data.tokens.length; i++) {
            if (i === 0) {
                probs.push(null); // First token has no previous position
            } else {
                const thisTokenId = tokenIds[i];
                const entry = data.outputProbs[`${i - 1}:${thisTokenId}`];
                probs.push(entry?.prob ?? null);
            }
        }
        return probs;
    });

    // UI state
    let hoveredNode = $state<HoveredNode | null>(null);
    let hoveredEdge = $state<HoveredEdge | null>(null);
    let hoveredBarClusterId = $state<number | null>(null);
    let isHoveringTooltip = $state(false);
    let tooltipPos = $state<TooltipPos>({ left: 0, top: 0 });
    let edgeTooltipPos = $state({ x: 0, y: 0 });

    $effect(() => {
        onHoveredNodeChange?.(hoveredNode);
    });

    // Alt/Option key temporarily toggles hide unpinned edges
    let altHeld = $state(false);
    const effectiveHideUnpinned = $derived(altHeld ? !hideUnpinnedEdges : hideUnpinnedEdges);

    $effect(() => {
        function onKeyDown(e: KeyboardEvent) {
            if (e.key === "Alt") {
                altHeld = true;
            }
        }
        function onKeyUp(e: KeyboardEvent) {
            if (e.key === "Alt") {
                altHeld = false;
            }
        }
        function onBlur() {
            altHeld = false;
        }
        window.addEventListener("keydown", onKeyDown);
        window.addEventListener("keyup", onKeyUp);
        window.addEventListener("blur", onBlur);
        return () => {
            window.removeEventListener("keydown", onKeyDown);
            window.removeEventListener("keyup", onKeyUp);
            window.removeEventListener("blur", onBlur);
        };
    });

    // Refs
    let innerContainer: HTMLDivElement;
    let edgeCanvas: HTMLCanvasElement;

    type EdgeDrawItem = {
        edge: EdgeData;
        path: Path2D;
        color: string;
        width: number;
        opacity: number;
    };

    // Zoom/pan
    const zoom = useZoomPan(() => innerContainer);

    function getRowKey(layer: string): string {
        return _getRowKey(layer);
    }

    function getRowLabel(layer: string): string {
        return _getRowLabel(layer);
    }

    // Select active edge variant based on display setting
    const active = $derived(getActiveEdges(data, displaySettings.edgeVariant));
    const activeEdges = $derived(active.edges);
    const activeEdgesBySource = $derived(active.bySource);
    const activeEdgesByTarget = $derived(active.byTarget);

    // Use pre-computed values from backend, derive max CI
    const maxAbsAttr = $derived(active.maxAbsAttr);
    const maxCi = $derived.by(() => {
        let max = 0;
        for (const ci of Object.values(data.nodeCiVals)) {
            if (ci > max) max = ci;
        }
        return max || 1; // Avoid division by zero
    });
    // Check if nodeSubcompActs has actual data (empty object {} is truthy in JS)
    const hasSubcompActData = $derived(data.nodeSubcompActs && Object.keys(data.nodeSubcompActs).length > 0);
    const maxAbsSubcompAct = $derived(data.maxAbsSubcompAct);

    // All nodes from nodeCiVals (for layout and rendering)
    const allNodes = $derived(new SvelteSet(Object.keys(data.nodeCiVals)));

    // Pre-compute pinned node keys for efficient lookup
    const pinnedNodeKeys = $derived(new Set(stagedNodes.map((p) => `${p.layer}:${p.seqIdx}:${p.cIdx}`)));

    // For hover, we match by component (layer:cIdx), ignoring seqIdx
    const hoveredComponentKey = $derived(hoveredNode ? `${hoveredNode.layer}:${hoveredNode.cIdx}` : null);

    // Get cluster ID of hovered node or bar (for cluster-wide rotation effect)
    const hoveredClusterId = $derived.by(() => {
        if (hoveredBarClusterId !== null) return hoveredBarClusterId;
        if (!hoveredNode) return undefined;
        return runState.getClusterId(hoveredNode.layer, hoveredNode.cIdx);
    });

    // Filter edges by topK (for rendering). Edges arrive pre-sorted by abs(val) desc from backend.
    const filteredEdges = $derived(activeEdges.slice(0, topK));

    // Build layout
    const { nodePositions, layerYPositions, seqXStarts, width, height, clusterSpans } = $derived.by(() => {
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

        // Sort rows for Y positioning
        const rows = sortRows(Array.from(allRows));

        // Assign Y positions (output at top, embed at bottom)
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

        // Calculate column widths
        const tokens = data.tokens;
        const maxComponentsPerSeq = tokens.map((_, seqIdx) => {
            let maxAtSeq = 0;
            for (const row of rows) {
                // Count nodes in this row at this seq position
                // Rows are "block.sublayer" — find all layers that belong to this row
                let totalInRow = 0;
                for (const layer of allLayers) {
                    if (getRowKey(layer) === row) {
                        const nodes = nodesPerLayerSeq[`${layer}:${seqIdx}`] ?? [];
                        totalInRow += nodes.length;
                    }
                }
                // Add gaps between grouped projections (only for grouped rows)
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

        const MIN_COL_WIDTH = 30;
        const COL_PADDING = 16;
        const seqWidths = maxComponentsPerSeq.map((n) =>
            Math.max(MIN_COL_WIDTH, n * (COMPONENT_SIZE + componentGap) + COL_PADDING * 2),
        );
        const seqXStarts = [MARGIN.left];
        for (let i = 0; i < seqWidths.length - 1; i++) {
            seqXStarts.push(seqXStarts[i] + seqWidths[i]);
        }

        // Position nodes and compute cluster spans
        const nodePositions: Record<string, NodePosition> = {};
        const allClusterSpans: ClusterSpan[] = [];
        const QKV_GROUP_GAP = COMPONENT_SIZE + componentGap;

        for (const layer of allLayers) {
            const info = parseLayer(layer);
            const groupProjs = info.sublayer ? getGroupProjections(info.sublayer) : null;
            const isGrouped = groupProjs !== null && info.projection !== null && groupProjs.includes(info.projection);

            for (let seqIdx = 0; seqIdx < tokens.length; seqIdx++) {
                const nodes = nodesPerLayerSeq[`${layer}:${seqIdx}`];
                if (!nodes) continue;

                let baseX = seqXStarts[seqIdx] + COL_PADDING + layerXOffsets[layer];
                const baseY = layerYPositions[layer];

                // For grouped projections (e.g. q/k/v), offset X based on position in group
                if (isGrouped && groupProjs && info.projection) {
                    const projIdx = groupProjs.indexOf(info.projection);
                    for (let i = 0; i < projIdx; i++) {
                        const prevLayer = buildLayerAddress(info.block, info.sublayer, groupProjs[i]);
                        const prevLayerNodes = nodesPerLayerSeq[`${prevLayer}:${seqIdx}`] ?? [];
                        baseX += prevLayerNodes.length * (COMPONENT_SIZE + componentGap);
                        baseX += QKV_GROUP_GAP;
                    }
                }

                // Output nodes always sort by probability; internal nodes sort by cluster if mapping loaded, else by CI
                const sorted =
                    layer === "output" || !runState.clusterMapping
                        ? sortComponentsByImportance(nodes, layer, seqIdx, data.nodeCiVals, data.outputProbs)
                        : sortComponentsByCluster(nodes, layer, seqIdx, data.nodeCiVals, runState.getClusterId);
                const offsets = computeComponentOffsets(sorted, COMPONENT_SIZE, componentGap);

                for (const cIdx of nodes) {
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
        const widthVal = totalSeqWidth + MARGIN.right;
        const maxY = Math.max(...Object.values(layerYPositions), 0) + COMPONENT_SIZE;
        const heightVal = maxY + MARGIN.bottom;

        return {
            nodePositions,
            layerYPositions,
            seqXStarts,
            width: widthVal,
            height: heightVal,
            clusterSpans: allClusterSpans,
        };
    });

    // Derived SVG dimensions (fixes negative translate bug)
    const svgWidth = $derived(width * zoom.scale + Math.max(zoom.translateX, 0));
    const svgHeight = $derived(height * zoom.scale + Math.max(zoom.translateY, 0));

    const EDGE_HIT_AREA_WIDTH = 12; // Wide stroke for canvas isPointInStroke hit testing

    // Check if a node key matches the currently hovered component (same layer:cIdx, any seqIdx)
    // For embed nodes: match by token value (highlight same tokens across positions)
    // For other nodes: match by layer:cIdx (highlight same component across positions)
    function nodeMatchesHoveredComponent(nodeKey: string): boolean {
        if (!hoveredNode) return false;
        const [layer, seqIdxStr, cIdx] = nodeKey.split(":");
        const seqIdx = parseInt(seqIdxStr);

        // For embed nodes, match by token value
        if (hoveredNode.layer === "embed") {
            if (layer !== "embed") return false;
            return data.tokens[seqIdx] === data.tokens[hoveredNode.seqIdx];
        }

        // For other nodes, match by component key (layer:cIdx)
        return `${layer}:${cIdx}` === hoveredComponentKey;
    }

    // Check if a node is in the same cluster as the hovered node (for cluster rotation effect)
    function isNodeInSameCluster(nodeKey: string): boolean {
        // Only trigger if hovered node has a numeric cluster ID (not singleton/no mapping)
        if (hoveredClusterId === undefined || hoveredClusterId === null) return false;
        const [layer, , cIdxStr] = nodeKey.split(":");
        const cIdx = parseInt(cIdxStr);
        const nodeClusterId = runState.getClusterId(layer, cIdx);
        return nodeClusterId === hoveredClusterId;
    }

    // -- Node interaction state machine --
    // Global mode computed once; each node looks up its role via O(1) set/map membership.

    type SpotlightConnected = {
        role: "spotlight_connected";
        color: string;
        opacity: number;
    };

    type NodeRole =
        | "default"
        | "highlighted"
        | "cluster_hovered"
        | "dimmed"
        | "hidden"
        | "spotlight_source"
        | SpotlightConnected;

    type InteractionMode =
        | { mode: "spotlight"; connected: Map<string, SpotlightConnected>; hoveredKey: string }
        | { mode: "focusing"; accentKeys: Set<string>; clusterKeys: Set<string> }
        | { mode: "resting"; accentKeys: Set<string> };

    const interactionMode = $derived.by((): InteractionMode => {
        const isHovering = hoveredNode !== null || hoveredBarClusterId !== null;
        const hasPinned = pinnedNodeKeys.size > 0;

        // Spotlight: hideUnpinned + hovering a node + no pinned nodes
        if (effectiveHideUnpinned && hoveredNode && !hasPinned) {
            const nodeKey = `${hoveredNode.layer}:${hoveredNode.seqIdx}:${hoveredNode.cIdx}`;
            const maxAttr = maxAbsAttr;
            const connected = new SvelteMap<string, SpotlightConnected>();

            const addEdge = (edge: EdgeData, connectedKey: string) => {
                const absVal = Math.abs(edge.val);
                const existing = connected.get(connectedKey);
                if (!existing || absVal > existing.opacity) {
                    const val = edge.val;
                    connected.set(connectedKey, {
                        role: "spotlight_connected",
                        color: getEdgeColor(val),
                        opacity: lerp(0, 0.5, absVal / maxAttr),
                    });
                }
            };

            for (const edge of activeEdgesBySource.get(nodeKey) ?? []) addEdge(edge, edge.tgt);
            for (const edge of activeEdgesByTarget.get(nodeKey) ?? []) addEdge(edge, edge.src);

            return { mode: "spotlight", connected, hoveredKey: nodeKey };
        }

        // Focusing: something is being hovered — accented nodes highlighted, rest dimmed
        if (isHovering) {
            const accentKeys = new SvelteSet<string>();
            const clusterKeys = new SvelteSet<string>();
            for (const key of Object.keys(nodePositions)) {
                if (pinnedNodeKeys.has(key) || nodeMatchesHoveredComponent(key)) {
                    accentKeys.add(key);
                } else if (isNodeInSameCluster(key)) {
                    clusterKeys.add(key);
                }
            }
            return { mode: "focusing", accentKeys, clusterKeys };
        }

        // Resting: nothing hovered — pinned nodes accented, everything else default
        return { mode: "resting", accentKeys: pinnedNodeKeys };
    });

    function getNodeRole(key: string, mode: InteractionMode): NodeRole {
        switch (mode.mode) {
            case "spotlight": {
                if (key === mode.hoveredKey) return "spotlight_source";
                return mode.connected.get(key) ?? "hidden";
            }
            case "focusing":
                if (mode.accentKeys.has(key)) return "highlighted";
                if (mode.clusterKeys.has(key)) return "cluster_hovered";
                return "dimmed";
            case "resting":
                if (mode.accentKeys.has(key)) return "highlighted";
                return "default";
        }
    }

    type EdgeState = "normal" | "highlighted" | "hidden";

    // Hover acts as a "promotion": hidden → normal → highlighted
    function getEdgeState(src: string, tgt: string): EdgeState {
        const hasPinned = pinnedNodeKeys.size > 0;
        const connectedToPinned = pinnedNodeKeys.has(src) || pinnedNodeKeys.has(tgt);
        const connectedToHoveredNode = nodeMatchesHoveredComponent(src) || nodeMatchesHoveredComponent(tgt);
        const isThisEdgeHovered = hoveredEdge?.src === src && hoveredEdge?.tgt === tgt;

        // No pinned nodes
        if (!hasPinned) {
            // Spotlight mode: hide all edges, nodes carry the color instead
            if (effectiveHideUnpinned && hoveredNode) return "hidden";
            return connectedToHoveredNode ? "highlighted" : "normal";
        }

        // Has pinned nodes
        if (effectiveHideUnpinned) {
            if (!connectedToPinned) {
                // Show (not highlighted) edges connected to hovered component
                if (connectedToHoveredNode) return "normal";
                return "hidden";
            }
            // Highlight edges connected to pinned on edge/node hover
            if (isThisEdgeHovered || connectedToHoveredNode) return "highlighted";
            return "normal";
        } else {
            // Show all edges, connected ones highlighted by default
            // Edge hover: only that edge highlighted, others normal
            if (hoveredEdge) {
                return isThisEdgeHovered ? "highlighted" : "normal";
            }
            // Node hover: highlight connected to hovered component OR pinned nodes
            if (hoveredNode) {
                return connectedToHoveredNode || connectedToPinned ? "highlighted" : "normal";
            }
            // No hover: connected to pinned are highlighted
            return connectedToPinned ? "highlighted" : "normal";
        }
    }

    // Precompute edge geometry for canvas rendering.
    // Only recomputes when data/layout changes, not on hover.
    // Ordered smallest-first (reverse of filteredEdges which is desc by abs val).
    const edgeDrawData = $derived.by(() => {
        // Register coarse reactive dependencies
        const edges = filteredEdges;
        const positions = nodePositions;
        const maxAttr = maxAbsAttr;

        // Hot loop in untrack() to avoid O(n²) fine-grained proxy dependency tracking.
        // Without this, each property access on Svelte's deep reactive proxies (edge.src,
        // edge.tgt, etc.) registers a fine-grained dependency, scaling quadratically.
        return untrack(() => {
            const items: EdgeDrawItem[] = [];
            for (let i = edges.length - 1; i >= 0; i--) {
                const edge = edges[i];
                const p1 = positions[edge.src];
                const p2 = positions[edge.tgt];
                if (!p1 || !p2) continue;
                const path = new Path2D();
                path.moveTo(p1.x, p1.y);
                if (displaySettings.curvedEdges) {
                    const dy = Math.abs(p2.y - p1.y);
                    const curveOffset = Math.max(20, dy * 0.4);
                    path.bezierCurveTo(p1.x, p1.y - curveOffset, p2.x, p2.y + curveOffset, p2.x, p2.y);
                } else {
                    path.lineTo(p2.x, p2.y);
                }
                items.push({
                    edge,
                    path,
                    color: getEdgeColor(edge.val),
                    width: lerp(1, 4, Math.abs(edge.val) / maxAttr),
                    opacity: lerp(0, 0.5, Math.abs(edge.val) / maxAttr),
                });
            }
            return items;
        });
    });

    // Canvas edge rendering effect
    $effect(() => {
        if (!edgeCanvas) return;
        const ctx = edgeCanvas.getContext("2d");
        if (!ctx) return;

        // Register coarse reactive deps (used by getEdgeState inside the loop)
        const items = edgeDrawData;
        const cw = svgWidth;
        const ch = svgHeight;
        const zs = zoom.scale;
        const ztx = zoom.translateX;
        const zty = zoom.translateY;
        void pinnedNodeKeys;
        void hoveredNode;
        void hoveredEdge;
        void effectiveHideUnpinned;
        void hoveredComponentKey;
        void interactionMode;

        const dpr = window.devicePixelRatio || 1;

        // Hot loop in untrack() — same O(n²) proxy issue as edgeDrawData
        untrack(() => {
            // Size canvas to match SVG
            edgeCanvas.width = cw * dpr;
            edgeCanvas.height = ch * dpr;
            edgeCanvas.style.width = `${cw}px`;
            edgeCanvas.style.height = `${ch}px`;

            ctx.clearRect(0, 0, edgeCanvas.width, edgeCanvas.height);

            // Apply zoom/pan transform matching the SVG <g> transform
            ctx.setTransform(zs * dpr, 0, 0, zs * dpr, ztx * dpr, zty * dpr);

            // Draw edges
            for (const item of items) {
                const state = getEdgeState(item.edge.src, item.edge.tgt);
                if (state === "hidden") continue;

                const isHighlighted = state === "highlighted";
                ctx.strokeStyle = item.color;
                ctx.lineWidth = isHighlighted ? 3 : item.width;
                ctx.globalAlpha = isHighlighted ? 1 : item.opacity;
                ctx.stroke(item.path);
            }

            // Reset transform
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.globalAlpha = 1;
        });
    });

    // Pre-compute node styles (fill, opacity) - only recomputes when data/layout changes, not on hover
    const nodeStyles = $derived.by(() => {
        const styles: Record<string, { fill: string; opacity: number }> = {};

        for (const nodeKey of Object.keys(nodePositions)) {
            const [layer, seqIdxStr, cIdxStr] = nodeKey.split(":");
            const seqIdx = parseInt(seqIdxStr);
            const cIdx = parseInt(cIdxStr);

            let fill: string = colors.nodeDefault;
            let opacity = 0.2;

            if (layer === "output") {
                const probEntry = data.outputProbs[`${seqIdx}:${cIdx}`];
                if (probEntry) {
                    fill = rgbToCss(colors.outputBase);
                    opacity = 0.2 + probEntry.prob * 0.8;
                } else {
                    // remove me. we should just assert this should be present
                    console.error(`OutputNodeCard: no entry for ${seqIdx}:${cIdx}`);
                }
            } else {
                // Component nodes: color/opacity based on CI or subcomp activation
                if (displaySettings.nodeColorMode === "ci" || !hasSubcompActData) {
                    const ci = data.nodeCiVals[`${layer}:${seqIdx}:${cIdx}`] || 0;
                    const intensity = ci / maxCi;
                    if (intensity > 1) {
                        throw new Error(`Inconsistent state: intensity > 1: ${intensity}`);
                    }
                    opacity = 0.2 + intensity * 0.8;
                } else {
                    const subcompAct = data.nodeSubcompActs![`${layer}:${seqIdx}:${cIdx}`] ?? 0;
                    const intensity = subcompAct / maxAbsSubcompAct;
                    if (intensity > 1) {
                        throw new Error(`Inconsistent state: intensity > 1: ${intensity}`);
                    }
                    fill = getSubcompActColor(subcompAct);
                    opacity = 0.2 + intensity * 0.8;
                }
            }

            styles[nodeKey] = { fill, opacity };
        }

        return styles;
    });

    // Event handlers
    let hoverTimeout: ReturnType<typeof setTimeout> | null = null;

    function handleNodeMouseEnter(event: MouseEvent, layer: string, seqIdx: number, cIdx: number) {
        // Clear any pending leave timeout
        if (hoverTimeout) {
            clearTimeout(hoverTimeout);
            hoverTimeout = null;
        }

        hoveredNode = { layer, seqIdx, cIdx };
        const size = layer === "embed" || layer === "output" ? "small" : "large";
        tooltipPos = calcTooltipPos(event.clientX, event.clientY, size);
    }

    function handleNodeMouseLeave() {
        // Clear any existing timeout first
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

    function handleNodeClick(layer: string, seqIdx: number, cIdx: number) {
        toggleComponentPinned(layer, cIdx, seqIdx);
        hoveredNode = null;
    }

    function toggleComponentPinned(layer: string, cIdx: number, seqIdx: number) {
        const idx = stagedNodes.findIndex((p) => p.layer === layer && p.seqIdx === seqIdx && p.cIdx === cIdx);
        if (idx >= 0) {
            onStagedNodesChange(stagedNodes.filter((_, i) => i !== idx));
        } else {
            onStagedNodesChange([...stagedNodes, { layer, seqIdx, cIdx }]);
        }
    }

    // Cached offscreen context for edge hit testing
    let hitTestCtx: CanvasRenderingContext2D | null = null;

    function handleEdgeHitTest(event: MouseEvent) {
        if (pinnedNodeKeys.size === 0) {
            if (hoveredEdge) hoveredEdge = null;
            return;
        }

        // Convert screen coords to world coords
        const rect = innerContainer.getBoundingClientRect();
        const screenX = event.clientX - rect.left + innerContainer.scrollLeft;
        const screenY = event.clientY - rect.top + innerContainer.scrollTop;
        const worldX = (screenX - zoom.translateX) / zoom.scale;
        const worldY = (screenY - zoom.translateY) / zoom.scale;

        if (!hitTestCtx) {
            hitTestCtx = document.createElement("canvas").getContext("2d");
        }
        if (!hitTestCtx) return;
        hitTestCtx.lineWidth = EDGE_HIT_AREA_WIDTH / zoom.scale;

        // Test edges connected to pinned nodes (reverse order so largest edges match first)
        for (let i = edgeDrawData.length - 1; i >= 0; i--) {
            const { edge, path } = edgeDrawData[i];
            if (!pinnedNodeKeys.has(edge.src) && !pinnedNodeKeys.has(edge.tgt)) continue;
            if (hitTestCtx.isPointInStroke(path, worldX, worldY)) {
                hoveredEdge = { src: edge.src, tgt: edge.tgt, val: edge.val };
                edgeTooltipPos = { x: event.clientX + 10, y: event.clientY + 10 };
                return;
            }
        }

        if (hoveredEdge) hoveredEdge = null;
    }

    function handlePanStart(event: MouseEvent) {
        const target = event.target as Element;
        if (target.closest(".node-group") || target.closest(".cluster-bar")) return;
        // Pan on shift+left-click or middle-click
        if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
            zoom.startPan(event);
        }
    }

    // Notify parent of edge count changes
    $effect(() => {
        onEdgeCountChange?.(filteredEdges.length);
    });
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
    class="graph-wrapper"
    class:panning={zoom.isPanning}
    onmousedown={handlePanStart}
    onmousemove={zoom.updatePan}
    onmouseup={zoom.endPan}
    onmouseleave={zoom.endPan}
>
    <ZoomControls
        scale={zoom.scale}
        onZoomIn={zoom.zoomIn}
        onZoomOut={zoom.zoomOut}
        onReset={zoom.reset}
        hint="Shift+drag to pan, Shift+scroll to zoom"
    />

    <div class="layer-labels-container" style="width: {LABEL_WIDTH}px;">
        <svg width={LABEL_WIDTH} height={svgHeight} style="display: block;">
            <g transform="translate(0, {zoom.translateY}) scale(1, {zoom.scale})">
                {#each Object.entries(layerYPositions) as [layer, y] (layer)}
                    {@const yCenter = y + COMPONENT_SIZE / 2}
                    <text
                        x={LABEL_WIDTH - 10}
                        y={yCenter}
                        text-anchor="end"
                        dominant-baseline="middle"
                        font-size="10"
                        font-weight="500"
                        font-family="'Berkeley Mono', 'SF Mono', monospace"
                        fill={colors.textSecondary}
                    >
                        {getRowLabel(layer)}
                    </text>
                {/each}
            </g>
        </svg>
    </div>

    <div class="graph-container" bind:this={innerContainer} onmousemove={handleEdgeHitTest}>
        <canvas bind:this={edgeCanvas} class="edge-canvas" style="width: {svgWidth}px; height: {svgHeight}px;"></canvas>
        <svg width={svgWidth} height={svgHeight}>
            <g transform="translate({zoom.translateX}, {zoom.translateY}) scale({zoom.scale})">
                <!-- Cluster bars (below nodes) -->
                <g class="cluster-bars-layer">
                    {#each clusterSpans as span (`${span.layer}:${span.seqIdx}:${span.clusterId}`)}
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

                <!-- Nodes (reactive for interactivity) -->
                <g class="nodes-layer">
                    {#each Object.entries(nodePositions) as [key, pos] (key)}
                        {@const [layer, seqIdxStr, cIdxStr] = key.split(":")}
                        {@const seqIdx = parseInt(seqIdxStr)}
                        {@const cIdx = parseInt(cIdxStr)}
                        {@const role = getNodeRole(key, interactionMode)}
                        {@const style = nodeStyles[key]}
                        {@const isSpotlight = typeof role === "object"}
                        {#if role !== "hidden"}
                            <g
                                class="node-group"
                                onmouseenter={(e) => handleNodeMouseEnter(e, layer, seqIdx, cIdx)}
                                onmouseleave={handleNodeMouseLeave}
                                onclick={() => handleNodeClick(layer, seqIdx, cIdx)}
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
                                    class:highlighted={role === "highlighted"}
                                    class:cluster-hovered={role === "cluster_hovered"}
                                    class:dimmed={role === "dimmed"}
                                    x={pos.x - COMPONENT_SIZE / 2}
                                    y={pos.y - COMPONENT_SIZE / 2}
                                    width={COMPONENT_SIZE}
                                    height={COMPONENT_SIZE}
                                    fill={isSpotlight ? role.color : style.fill}
                                    rx="1"
                                    fill-opacity={isSpotlight ? role.opacity : style.opacity}
                                    stroke={isSpotlight ? colors.textSecondary : "none"}
                                    stroke-width={isSpotlight ? 1.5 : 0}
                                />
                            </g>
                        {/if}
                    {/each}
                </g>
            </g>
        </svg>

        <div class="token-labels-container">
            <svg width={svgWidth} height="50" style="display: block;">
                <g transform="translate({zoom.translateX}, 0) scale({zoom.scale}, 1)">
                    {#each data.tokens as token, i (i)}
                        {@const colLeft = seqXStarts[i] + 8}
                        {@const maskedProb = maskedSelfProbs[i]}
                        <text
                            x={colLeft}
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
                            x={colLeft}
                            y="36"
                            text-anchor="start"
                            font-size="9"
                            font-family="'Berkeley Mono', 'SF Mono', monospace"
                            fill={colors.textMuted}>[{i}]</text
                        >
                        <!-- Masked prediction probability dot: P(self | previous) -->
                        {@const isFirstToken = i === 0}
                        <circle
                            cx={colLeft + 24}
                            cy="33"
                            r="4"
                            fill={getNextTokenProbBgColor(maskedProb)}
                            stroke={colors.textMuted}
                            stroke-width="0.5"
                        >
                            <title
                                >{maskedProb !== null
                                    ? `P(self): ${(maskedProb * 100).toFixed(1)}%`
                                    : isFirstToken
                                      ? "First token"
                                      : "P(self): <1%"}</title
                            >
                        </circle>
                    {/each}
                </g>
            </svg>
        </div>
    </div>

    <!-- Edge tooltip -->
    {#if hoveredEdge}
        <div class="edge-tooltip" style="left: {edgeTooltipPos.x}px; top: {edgeTooltipPos.y}px;">
            <div class="edge-tooltip-row">
                <span class="edge-tooltip-label">Src</span>
                <code>{hoveredEdge.src}</code>
            </div>
            <div class="edge-tooltip-row">
                <span class="edge-tooltip-label">Tgt</span>
                <code>{hoveredEdge.tgt}</code>
            </div>
            <div class="edge-tooltip-row">
                <span class="edge-tooltip-label">Val</span>
                <span style="color: {getEdgeColor(hoveredEdge.val)}; font-weight: 600;">
                    {hoveredEdge.val.toFixed(4)}
                </span>
            </div>
        </div>
    {/if}

    <!-- Node tooltip -->
    {#if hoveredNode}
        <NodeTooltip
            {hoveredNode}
            {tooltipPos}
            {hideNodeCard}
            outputProbs={data.outputProbs}
            nodeCiVals={data.nodeCiVals}
            nodeSubcompActs={data.nodeSubcompActs}
            tokens={data.tokens}
            edgesBySource={activeEdgesBySource}
            edgesByTarget={activeEdgesByTarget}
            onMouseEnter={() => (isHoveringTooltip = true)}
            onMouseLeave={() => {
                isHoveringTooltip = false;
                handleNodeMouseLeave();
            }}
            onPinComponent={toggleComponentPinned}
        />
    {/if}
</div>

<style>
    .graph-wrapper {
        display: flex;
        background: var(--bg-surface);
        overflow: hidden;
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
        overflow: auto;
        flex: 1;
        position: relative;
        background: var(--bg-inset);
    }

    .edge-canvas {
        position: absolute;
        top: 0;
        left: 0;
        pointer-events: none;
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

    .node-group {
        cursor: pointer;
    }

    .node {
        transform-box: fill-box;
        transform-origin: center;
        transition: transform var(--transition-normal);
    }

    .node.cluster-hovered {
        transform: rotate(45deg);
    }

    .node.dimmed {
        transform: scale(0.5);
    }

    .node.highlighted {
        stroke: var(--accent-primary) !important;
        stroke-width: 2px !important;
        filter: brightness(1.2);
        opacity: 1 !important;
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

    .edge-tooltip {
        position: fixed;
        padding: var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        z-index: 1000;
        pointer-events: auto;
        font-family: var(--font-mono);
        font-size: var(--text-sm);
    }

    .edge-tooltip-row {
        margin: var(--space-1) 0;
        display: flex;
        gap: var(--space-2);
    }

    .edge-tooltip-label {
        color: var(--text-muted);
        font-size: var(--text-xs);
        letter-spacing: 0.05em;
        min-width: 4em;
    }

    .edge-tooltip code {
        color: var(--text-primary);
        font-size: var(--text-sm);
    }
</style>
