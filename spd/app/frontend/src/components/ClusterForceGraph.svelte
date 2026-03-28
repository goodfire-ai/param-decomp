<script lang="ts">
    import { getContext } from "svelte";
    import { fetchClusteringCoactivation, type ClusterPairwiseResponse } from "../lib/api/clusters";
    import ZoomControls from "../lib/ZoomControls.svelte";
    import { useZoomPan } from "../lib/useZoomPan.svelte";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";

    type ComponentMember = { layer: string; cIdx: number };

    type Props = {
        members: ComponentMember[];
        clusteringRunId: string;
    };

    type DataState =
        | { status: "loading" }
        | { status: "loaded"; response: ClusterPairwiseResponse }
        | { status: "error"; error: string };

    type GraphNode = {
        key: string;
        member: ComponentMember;
        count: number;
        radius: number;
        x: number;
        y: number;
        vx: number;
        vy: number;
    };

    type GraphEdge = {
        source: string;
        target: string;
        weight: number;
        countAB: number;
    };

    let { members, clusteringRunId }: Props = $props();

    const runState = getContext<RunContext>(RUN_KEY);

    let dataState: DataState = $state({ status: "loading" });
    let minJaccard = $state(0.08);
    let showIsolated = $state(true);
    let selectedNodeKey: string | null = $state(null);
    let hoveredNodeKey: string | null = $state(null);

    let graphContainer: HTMLDivElement | null = $state(null);
    const zoom = useZoomPan(() => graphContainer);

    const keys = $derived(members.map((member) => `${member.layer}:${member.cIdx}`));

    function getLabel(key: string): string {
        const loadable = runState.getInterpretation(key);
        if (loadable.status === "loaded" && loadable.data.status === "generated") {
            return loadable.data.data.label;
        }
        return key;
    }

    $effect(() => {
        const currentKeys = keys;
        const runId = clusteringRunId;
        if (currentKeys.length < 2) {
            dataState = { status: "loaded", response: { pairs: [], n_tokens: 0 } };
            return;
        }

        dataState = { status: "loading" };
        fetchClusteringCoactivation(currentKeys, runId)
            .then((response) => {
                dataState = { status: "loaded", response };
            })
            .catch((error) => {
                dataState = { status: "error", error: String(error) };
            });
    });

    const baseGraph = $derived.by(() => {
        const memberByKey = new Map<string, ComponentMember>();
        for (const member of members) {
            memberByKey.set(`${member.layer}:${member.cIdx}`, member);
        }

        const counts = new Map<string, number>();
        const allEdges: GraphEdge[] = [];
        let maxCount = 1;

        if (dataState.status === "loaded") {
            for (const pair of dataState.response.pairs) {
                counts.set(pair.key_a, pair.count_a);
                counts.set(pair.key_b, pair.count_b);
                maxCount = Math.max(maxCount, pair.count_a, pair.count_b);
                allEdges.push({
                    source: pair.key_a,
                    target: pair.key_b,
                    weight: pair.jaccard,
                    countAB: pair.count_ab,
                });
            }
        }

        const nodes = keys
            .map((key) => ({
                key,
                member: memberByKey.get(key)!,
                count: counts.get(key) ?? 0,
            }))
            .filter((node) => node.member !== undefined)
            .map((node) => ({
                ...node,
                radius: 6 + (Math.sqrt(node.count) / Math.sqrt(maxCount || 1)) * 10,
            }));

        return { nodes, allEdges };
    });

    const visibleGraph = $derived.by(() => {
        const edges = baseGraph.allEdges.filter((edge) => edge.weight >= minJaccard);
        const connected = new Set<string>();
        for (const edge of edges) {
            connected.add(edge.source);
            connected.add(edge.target);
        }

        const nodes = showIsolated ? baseGraph.nodes : baseGraph.nodes.filter((node) => connected.has(node.key));

        const visibleNodeKeys = new Set(nodes.map((node) => node.key));
        return {
            nodes,
            edges: edges.filter((edge) => visibleNodeKeys.has(edge.source) && visibleNodeKeys.has(edge.target)),
        };
    });

    const nodeLabels = $derived(new Map(keys.map((key) => [key, getLabel(key)])));

    let renderedNodes: GraphNode[] = $state([]);
    let simNodes: GraphNode[] = [];
    let lastPositions = new Map<string, { x: number; y: number }>();

    const WIDTH = 920;
    const HEIGHT = 580;

    function pushRenderedNodes() {
        renderedNodes = simNodes.map((node: GraphNode) => ({ ...node }));
        lastPositions = new Map(simNodes.map((node: GraphNode) => [node.key, { x: node.x, y: node.y }]));
    }

    $effect(() => {
        const graph = visibleGraph;
        const edgeCount = graph.edges.length;
        const nodeCount = graph.nodes.length;
        if (nodeCount === 0) {
            renderedNodes = [];
            return;
        }

        simNodes = graph.nodes.map((node, index) => {
            const saved = lastPositions.get(node.key);
            const angle = (index / Math.max(nodeCount, 1)) * Math.PI * 2;
            return {
                ...node,
                x: saved?.x ?? WIDTH / 2 + Math.cos(angle) * Math.min(220, 60 + nodeCount * 8),
                y: saved?.y ?? HEIGHT / 2 + Math.sin(angle) * Math.min(180, 50 + nodeCount * 6),
                vx: 0,
                vy: 0,
            };
        });

        let frame = 0;
        let raf = 0;

        const step = () => {
            const nodeByKey = new Map(simNodes.map((node) => [node.key, node]));

            for (let i = 0; i < simNodes.length; i++) {
                const a = simNodes[i];
                for (let j = i + 1; j < simNodes.length; j++) {
                    const b = simNodes[j];
                    const dx = a.x - b.x;
                    const dy = a.y - b.y;
                    const distSq = Math.max(dx * dx + dy * dy, 1);
                    const force = 2400 / distSq;
                    const fx = (dx / Math.sqrt(distSq)) * force;
                    const fy = (dy / Math.sqrt(distSq)) * force;
                    a.vx += fx;
                    a.vy += fy;
                    b.vx -= fx;
                    b.vy -= fy;
                }
            }

            for (const edge of graph.edges) {
                const source = nodeByKey.get(edge.source);
                const target = nodeByKey.get(edge.target);
                if (!source || !target) continue;
                const dx = target.x - source.x;
                const dy = target.y - source.y;
                const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
                const desired = 150 - edge.weight * 90;
                const spring = 0.003 + edge.weight * 0.03;
                const force = (dist - desired) * spring;
                const fx = (dx / dist) * force;
                const fy = (dy / dist) * force;
                source.vx += fx;
                source.vy += fy;
                target.vx -= fx;
                target.vy -= fy;
            }

            let totalSpeed = 0;
            for (const node of simNodes) {
                const cx = WIDTH / 2 - node.x;
                const cy = HEIGHT / 2 - node.y;
                node.vx += cx * 0.0008;
                node.vy += cy * 0.0008;
                node.vx *= 0.86;
                node.vy *= 0.86;
                node.x = Math.max(24, Math.min(WIDTH - 24, node.x + node.vx));
                node.y = Math.max(24, Math.min(HEIGHT - 24, node.y + node.vy));
                totalSpeed += Math.abs(node.vx) + Math.abs(node.vy);
            }

            pushRenderedNodes();
            frame += 1;
            if (frame < Math.max(180, edgeCount * 8) || totalSpeed > 0.4) {
                raf = requestAnimationFrame(step);
            }
        };

        raf = requestAnimationFrame(step);
        return () => cancelAnimationFrame(raf);
    });

    const nodeLookup = $derived(new Map<string, GraphNode>(renderedNodes.map((node: GraphNode) => [node.key, node])));
    const rankedCounts = $derived([...renderedNodes].map((node: GraphNode) => node.count).sort((a, b) => b - a));
    const labelThreshold = $derived(rankedCounts[Math.min(5, Math.max(rankedCounts.length - 1, 0))] ?? 0);

    function edgePath(edge: GraphEdge): string {
        const source = nodeLookup.get(edge.source);
        const target = nodeLookup.get(edge.target);
        if (!source || !target) return "";
        return `M ${source.x} ${source.y} L ${target.x} ${target.y}`;
    }

    function edgeOpacity(weight: number): number {
        return 0.12 + Math.min(weight / 0.3, 1) * 0.68;
    }

    function edgeWidth(weight: number): number {
        return 1 + Math.min(weight / 0.3, 1) * 4;
    }

    function shouldShowLabel(node: GraphNode): boolean {
        return (
            renderedNodes.length <= 12 ||
            node.key === selectedNodeKey ||
            node.key === hoveredNodeKey ||
            node.count >= labelThreshold
        );
    }

    function nodeFill(node: GraphNode): string {
        if (node.key === selectedNodeKey) return "var(--accent-positive)";
        if (node.key === hoveredNodeKey) return "var(--accent-primary)";
        return "rgba(59, 130, 246, 0.85)";
    }

    function nodeStroke(node: GraphNode): string {
        if (node.key === selectedNodeKey || node.key === hoveredNodeKey) return "rgba(255, 255, 255, 0.95)";
        return "rgba(15, 23, 42, 0.2)";
    }

    function graphPointFromEvent(event: MouseEvent): { x: number; y: number } | null {
        if (!graphContainer) return null;
        const rect = graphContainer.getBoundingClientRect();
        return {
            x: (event.clientX - rect.left + graphContainer.scrollLeft - zoom.translateX) / zoom.scale,
            y: (event.clientY - rect.top + graphContainer.scrollTop - zoom.translateY) / zoom.scale,
        };
    }

    let draggedNodeKey: string | null = $state(null);

    function handleGraphMouseDown(event: MouseEvent) {
        if (event.button === 1 || (event.button === 0 && event.shiftKey)) {
            zoom.startPan(event);
        }
    }

    function handleNodeMouseDown(event: MouseEvent, key: string) {
        if (event.button !== 0 || event.shiftKey) return;
        event.preventDefault();
        event.stopPropagation();
        draggedNodeKey = key;
        selectedNodeKey = key;
    }

    function handleGraphMouseMove(event: MouseEvent) {
        if (draggedNodeKey) {
            const point = graphPointFromEvent(event);
            if (!point) return;
            const node = simNodes.find((candidate: GraphNode) => candidate.key === draggedNodeKey);
            if (!node) return;
            node.x = Math.max(24, Math.min(WIDTH - 24, point.x));
            node.y = Math.max(24, Math.min(HEIGHT - 24, point.y));
            node.vx = 0;
            node.vy = 0;
            pushRenderedNodes();
            return;
        }
        zoom.updatePan(event);
    }

    function handleGraphMouseUp() {
        draggedNodeKey = null;
        zoom.endPan();
    }

    const highlightedEdges = $derived.by(() => {
        const focus = selectedNodeKey ?? hoveredNodeKey;
        if (!focus) return new Set<string>();
        return new Set(
            visibleGraph.edges
                .filter((edge) => edge.source === focus || edge.target === focus)
                .map((edge) => `${edge.source}|${edge.target}`),
        );
    });

    const selectedSummary = $derived.by(() => {
        const focus = selectedNodeKey ?? hoveredNodeKey;
        if (!focus) return null;
        const node = renderedNodes.find((candidate: GraphNode) => candidate.key === focus);
        if (!node) return null;
        const neighbors = visibleGraph.edges
            .filter((edge) => edge.source === focus || edge.target === focus)
            .map((edge) => ({
                key: edge.source === focus ? edge.target : edge.source,
                weight: edge.weight,
                countAB: edge.countAB,
            }))
            .sort((a, b) => b.weight - a.weight)
            .slice(0, 5);
        return { node, neighbors };
    });
</script>

<div class="force-graph-section">
    <div class="section-header">
        <div>
            <h3 class="section-title">Force-Directed Co-Firing Graph</h3>
            <div class="section-description">
                Obsidian-style network view of cluster members. Edge strength is clustering-snapshot Jaccard; raise the
                threshold to surface the core scaffold.
            </div>
        </div>
        <div class="graph-controls">
            <label class="slider-control">
                <span>Min Jaccard</span>
                <input type="range" min="0" max="0.5" step="0.01" bind:value={minJaccard} />
                <span class="mono">{minJaccard.toFixed(2)}</span>
            </label>
            <label class="checkbox-control">
                <input type="checkbox" bind:checked={showIsolated} />
                <span>Show isolated</span>
            </label>
            <div class="graph-stats mono">{visibleGraph.nodes.length} nodes · {visibleGraph.edges.length} edges</div>
        </div>
    </div>

    {#if dataState.status === "loading"}
        <div class="status">Loading co-firing graph...</div>
    {:else if dataState.status === "error"}
        <div class="status error">{dataState.error}</div>
    {:else if keys.length < 2}
        <div class="status">Need at least 2 components.</div>
    {:else if visibleGraph.nodes.length === 0}
        <div class="status">No nodes remain at this threshold.</div>
    {:else}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
            class="graph-shell"
            bind:this={graphContainer}
            class:panning={zoom.isPanning}
            onmousedown={handleGraphMouseDown}
            onmousemove={handleGraphMouseMove}
            onmouseup={handleGraphMouseUp}
            onmouseleave={handleGraphMouseUp}
        >
            <ZoomControls
                scale={zoom.scale}
                onZoomIn={zoom.zoomIn}
                onZoomOut={zoom.zoomOut}
                onReset={zoom.reset}
                hint="Shift+scroll zoom"
            />

            <svg
                class="graph-canvas"
                width={WIDTH}
                height={HEIGHT}
                style="transform: translate({zoom.translateX}px, {zoom.translateY}px) scale({zoom.scale})"
            >
                {#each visibleGraph.edges as edge (`${edge.source}|${edge.target}`)}
                    {@const highlighted = highlightedEdges.has(`${edge.source}|${edge.target}`)}
                    <path
                        d={edgePath(edge)}
                        fill="none"
                        stroke="rgba(59, 130, 246, 0.9)"
                        stroke-width={edgeWidth(edge.weight)}
                        opacity={highlighted ? Math.min(1, edgeOpacity(edge.weight) + 0.2) : edgeOpacity(edge.weight)}
                    >
                        <title
                            >{edge.source} ↔ {edge.target} · Jaccard {edge.weight.toFixed(3)} · co-fire {edge.countAB}</title
                        >
                    </path>
                {/each}

                {#each renderedNodes as node (node.key)}
                    <g transform={`translate(${node.x}, ${node.y})`}>
                        <circle
                            r={node.radius}
                            fill={nodeFill(node)}
                            stroke={nodeStroke(node)}
                            stroke-width={node.key === selectedNodeKey ? 2.5 : 1.5}
                            class="graph-node"
                            onmouseenter={() => (hoveredNodeKey = node.key)}
                            onmouseleave={() => (hoveredNodeKey = null)}
                            onmousedown={(event: MouseEvent) => handleNodeMouseDown(event, node.key)}
                            onclick={() => (selectedNodeKey = selectedNodeKey === node.key ? null : node.key)}
                        >
                            <title>{node.key} · {node.count.toLocaleString()} activations</title>
                        </circle>
                        {#if shouldShowLabel(node)}
                            <text x={node.radius + 6} y="4" class="node-label">{nodeLabels.get(node.key) ?? node.key}</text>
                        {/if}
                    </g>
                {/each}
            </svg>
        </div>

        {#if selectedSummary}
            <div class="selection-panel">
                <div class="selection-title">{nodeLabels.get(selectedSummary.node.key) ?? selectedSummary.node.key}</div>
                <div class="selection-meta mono">
                    {selectedSummary.node.key} · {selectedSummary.node.count.toLocaleString()} activations
                </div>
                <div class="neighbor-list">
                    {#if selectedSummary.neighbors.length === 0}
                        <span class="neighbor-empty">No visible neighbors at this threshold.</span>
                    {:else}
                        {#each selectedSummary.neighbors as neighbor (neighbor.key)}
                            <span class="neighbor-pill">
                                {nodeLabels.get(neighbor.key) ?? neighbor.key} · J {neighbor.weight.toFixed(2)} · {neighbor.countAB}
                            </span>
                        {/each}
                    {/if}
                </div>
            </div>
        {/if}
    {/if}
</div>

<style>
    .force-graph-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
    }

    .section-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: var(--space-4);
        flex-wrap: wrap;
    }

    .section-title {
        margin: 0;
        font-size: var(--text-base);
        font-weight: 600;
    }

    .section-description {
        margin-top: var(--space-1);
        max-width: 720px;
        color: var(--text-muted);
        font-size: var(--text-xs);
        line-height: 1.4;
    }

    .graph-controls {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        flex-wrap: wrap;
    }

    .slider-control,
    .checkbox-control {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        font-size: var(--text-sm);
        color: var(--text-secondary);
    }

    .mono,
    .graph-stats {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .graph-shell {
        position: relative;
        overflow: auto;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        background:
            radial-gradient(circle at top, rgba(59, 130, 246, 0.08), transparent 35%),
            linear-gradient(180deg, rgba(15, 23, 42, 0.04), transparent 20%), var(--bg-base);
        cursor: default;
    }

    .graph-shell.panning {
        cursor: grabbing;
    }

    .graph-canvas {
        display: block;
        transform-origin: 0 0;
    }

    .graph-node {
        cursor: grab;
        transition:
            fill var(--transition-fast),
            stroke var(--transition-fast);
    }

    .graph-node:active {
        cursor: grabbing;
    }

    .node-label {
        font-size: 11px;
        fill: var(--text-secondary);
        pointer-events: none;
    }

    .selection-panel {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        padding: var(--space-3);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        background: var(--bg-elevated);
    }

    .selection-title {
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
    }

    .selection-meta {
        color: var(--text-muted);
    }

    .neighbor-list {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-2);
    }

    .neighbor-pill {
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
        background: var(--bg-inset);
        color: var(--text-secondary);
        font-size: var(--text-xs);
        white-space: nowrap;
    }

    .neighbor-empty,
    .status {
        font-size: var(--text-sm);
        color: var(--text-muted);
    }

    .status {
        padding: var(--space-4);
    }

    .status.error {
        color: var(--status-error);
    }
</style>
