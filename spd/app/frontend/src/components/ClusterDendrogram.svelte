<script lang="ts">
    import { getContext } from "svelte";
    import { SvelteMap, SvelteSet } from "svelte/reactivity";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import type { MergeIterationsResponse, MergePairIteration } from "../lib/api/clusters";
    import { fetchMergeIterations, fetchClusterPairwiseCorrelations } from "../lib/api/clusters";

    type ComponentMember = { layer: string; cIdx: number };

    type Props = {
        members: ComponentMember[];
        clusteringRunId: string;
        iteration: number;
        onLeafOrder?: (orderedMembers: ComponentMember[]) => void;
    };

    let { members, clusteringRunId, iteration, onLeafOrder }: Props = $props();

    const runState = getContext<RunContext>(RUN_KEY);

    type DataState =
        | { status: "loading" }
        | { status: "loaded"; response: MergeIterationsResponse }
        | { status: "error"; error: string };

    let data = $state<DataState>({ status: "loading" });

    const keys = $derived(members.map((m) => `${m.layer}:${m.cIdx}`));

    function getLabel(key: string): string {
        const loadable = runState.getInterpretation(key);
        if (loadable.status === "loaded" && loadable.data.status === "generated") {
            return loadable.data.data.label;
        }
        return key;
    }

    let firingCounts = new SvelteMap<string, number>();

    $effect(() => {
        const currentKeys = keys;
        if (currentKeys.length < 2) {
            data = { status: "loaded", response: { pairs: [], total_iterations: 0 } };
            return;
        }
        data = { status: "loading" };
        fetchMergeIterations(currentKeys, clusteringRunId, iteration)
            .then((response) => (data = { status: "loaded", response }))
            .catch((e) => (data = { status: "error", error: String(e) }));

        fetchClusterPairwiseCorrelations(currentKeys)
            .then((response) => {
                firingCounts.clear();
                for (const p of response.pairs) {
                    firingCounts.set(p.key_a, p.count_a);
                    firingCounts.set(p.key_b, p.count_b);
                }
            })
            .catch(() => {});
    });

    // Dendrogram tree types
    type LeafNode = { type: "leaf"; key: string; y: number };
    type InternalNode = { type: "internal"; left: TreeNode; right: TreeNode; mergeIter: number; y: number };
    type TreeNode = LeafNode | InternalNode;

    // Build dendrogram from pairwise merge iterations via single-linkage
    function buildDendrogram(
        componentKeys: string[],
        pairs: MergePairIteration[],
    ): TreeNode | null {
        if (componentKeys.length === 0) return null;
        if (componentKeys.length === 1) return { type: "leaf", key: componentKeys[0], y: 0 };

        // Sort pairs by merge iteration (earliest first)
        const sorted = pairs
            .filter((p) => p.merge_iteration >= 0)
            .sort((a, b) => a.merge_iteration - b.merge_iteration);

        // Union-find
        const parent = new SvelteMap<string, string>();
        const nodes = new SvelteMap<string, TreeNode>();
        let leafY = 0;

        for (const key of componentKeys) {
            parent.set(key, key);
            nodes.set(key, { type: "leaf", key, y: leafY });
            leafY++;
        }

        function find(x: string): string {
            let root = x;
            while (parent.get(root) !== root) root = parent.get(root)!;
            // Path compression
            let curr = x;
            while (curr !== root) {
                const next = parent.get(curr)!;
                parent.set(curr, root);
                curr = next;
            }
            return root;
        }

        for (const pair of sorted) {
            const rootA = find(pair.key_a);
            const rootB = find(pair.key_b);
            if (rootA === rootB) continue;

            const nodeA = nodes.get(rootA)!;
            const nodeB = nodes.get(rootB)!;
            const merged: InternalNode = {
                type: "internal",
                left: nodeA,
                right: nodeB,
                mergeIter: pair.merge_iteration,
                y: (nodeA.y + nodeB.y) / 2,
            };

            // New root key
            const mergedKey = `${rootA}|${rootB}`;
            parent.set(rootA, mergedKey);
            parent.set(rootB, mergedKey);
            parent.set(mergedKey, mergedKey);
            nodes.set(mergedKey, merged);
        }

        // Find remaining roots and merge any disconnected components
        const roots = new SvelteSet<string>();
        for (const key of componentKeys) {
            roots.add(find(key));
        }
        const rootNodes = [...roots].map((r) => nodes.get(r)!);
        if (rootNodes.length === 1) return rootNodes[0];

        // If disconnected, chain them at max iteration
        let result = rootNodes[0];
        for (let i = 1; i < rootNodes.length; i++) {
            result = {
                type: "internal",
                left: result,
                right: rootNodes[i],
                mergeIter: -1,
                y: (result.y + rootNodes[i].y) / 2,
            };
        }
        return result;
    }

    // Assign leaf y-positions by in-order traversal (so tree doesn't cross)
    function assignLeafPositions(node: TreeNode, startY: number): number {
        if (node.type === "leaf") {
            node.y = startY;
            return startY + 1;
        }
        const nextY = assignLeafPositions(node.left, startY);
        const finalY = assignLeafPositions(node.right, nextY);
        node.y = (node.left.y + node.right.y) / 2;
        return finalY;
    }

    // Collect all leaves in order
    function collectLeaves(node: TreeNode): LeafNode[] {
        if (node.type === "leaf") return [node];
        return [...collectLeaves(node.left), ...collectLeaves(node.right)];
    }

    function collectInternalNodes(node: TreeNode): InternalNode[] {
        if (node.type === "leaf") return [];
        return [node, ...collectInternalNodes(node.left), ...collectInternalNodes(node.right)];
    }

    function countLeaves(node: TreeNode): number {
        if (node.type === "leaf") return 1;
        return countLeaves(node.left) + countLeaves(node.right);
    }

    // Find max merge iteration for scaling
    function maxIter(node: TreeNode): number {
        if (node.type === "leaf") return 0;
        return Math.max(node.mergeIter, maxIter(node.left), maxIter(node.right));
    }

    // Generate SVG paths for the dendrogram
    type SvgLine = { x1: number; y1: number; x2: number; y2: number; mergeIter: number };

    function generateLines(
        node: TreeNode,
        xScale: (iter: number) => number,
        yScale: (pos: number) => number,
        rightX: number,
    ): SvgLine[] {
        if (node.type === "leaf") return [];

        const nodeX = node.mergeIter >= 0 ? xScale(node.mergeIter) : 0;
        const leftChild = node.left;
        const rightChild = node.right;

        const leftX = leftChild.type === "leaf" ? rightX : xScale(leftChild.mergeIter);
        const rightChildX = rightChild.type === "leaf" ? rightX : xScale(rightChild.mergeIter);

        const lines: SvgLine[] = [
            // Vertical line connecting children
            { x1: nodeX, y1: yScale(leftChild.y), x2: nodeX, y2: yScale(rightChild.y), mergeIter: node.mergeIter },
            // Horizontal to left child
            { x1: nodeX, y1: yScale(leftChild.y), x2: leftX, y2: yScale(leftChild.y), mergeIter: node.mergeIter },
            // Horizontal to right child
            { x1: nodeX, y1: yScale(rightChild.y), x2: rightChildX, y2: yScale(rightChild.y), mergeIter: node.mergeIter },
        ];

        return [...lines, ...generateLines(leftChild, xScale, yScale, rightX), ...generateLines(rightChild, xScale, yScale, rightX)];
    }

    // Computed dendrogram
    const tree = $derived.by(() => {
        if (data.status !== "loaded" || keys.length < 2) return null;
        const root = buildDendrogram(keys, data.response.pairs);
        if (!root) return null;
        assignLeafPositions(root, 0);
        return root;
    });

    const leaves = $derived(tree ? collectLeaves(tree) : []);
    const internalNodes = $derived(tree ? collectInternalNodes(tree) : []);
    const maxFiringCount = $derived(Math.max(1, ...leaves.map((l) => firingCounts.get(l.key) ?? 0)));

    // Emit leaf order to parent
    $effect(() => {
        if (!onLeafOrder || leaves.length === 0) return;
        const keyToMember = new SvelteMap<string, ComponentMember>();
        for (const m of members) keyToMember.set(`${m.layer}:${m.cIdx}`, m);
        const ordered = leaves.map((l) => keyToMember.get(l.key)!).filter(Boolean);
        onLeafOrder(ordered);
    });

    // SVG dimensions
    const LABEL_WIDTH = 250;
    const TREE_WIDTH = 300;
    const ROW_HEIGHT = 24;
    const PADDING = 16;

    const svgWidth = $derived(LABEL_WIDTH + TREE_WIDTH + PADDING * 2);
    const svgHeight = $derived(leaves.length * ROW_HEIGHT + PADDING * 2);

    const maxMergeIter = $derived(tree ? maxIter(tree) : 1);

    // X scale: merge iteration -> x position (early merges on the right, late on the left)
    // This makes the tree grow from right (leaves) to left (root)
    const xScale = $derived((iter: number) => {
        const treeRight = PADDING + TREE_WIDTH;
        const treeLeft = PADDING;
        if (maxMergeIter === 0) return treeLeft;
        return treeRight - (iter / maxMergeIter) * (treeRight - treeLeft);
    });

    const yScale = $derived((pos: number) => PADDING + pos * ROW_HEIGHT + ROW_HEIGHT / 2);

    const lines = $derived.by(() => {
        if (!tree || tree.type === "leaf") return [];
        const rightX = PADDING + TREE_WIDTH;
        return generateLines(tree, xScale, yScale, rightX);
    });
</script>

<div class="dendrogram-section">
    <h3 class="section-title">Merge Tree</h3>
    <div class="section-description">
        How components were grouped by the clustering algorithm. Branches that join on the right were merged early (strong co-activation); branches that join on the left were merged late (weaker signal).
    </div>

    {#if data.status === "loading"}
        <div class="status">Loading...</div>
    {:else if data.status === "error"}
        <div class="status error">{data.error}</div>
    {:else if keys.length < 2}
        <div class="status">Need at least 2 components.</div>
    {:else if tree}
        <div class="dendrogram-scroll">
            <svg width={svgWidth} height={svgHeight} class="dendrogram-svg">
                <!-- Tree lines -->
                {#each lines as line, i (i)}
                    <line
                        x1={line.x1}
                        y1={line.y1}
                        x2={line.x2}
                        y2={line.y2}
                        class="tree-line"
                    />
                {/each}

                <!-- Merge node circles -->
                {#each internalNodes as node, i (i)}
                    {@const cx = node.mergeIter >= 0 ? xScale(node.mergeIter) : PADDING}
                    {@const cy = yScale(node.y)}
                    {@const leftCount = countLeaves(node.left)}
                    {@const rightCount = countLeaves(node.right)}
                    <circle
                        {cx} {cy} r="4"
                        class="merge-node"
                    >
                        <title>Iteration {node.mergeIter}: merge {leftCount} + {rightCount} components</title>
                    </circle>
                {/each}

                <!-- Leaf labels + firing density -->
                {#each leaves as leaf (leaf.key)}
                    {@const firing = firingCounts.get(leaf.key) ?? 0}
                    {@const barWidth = (firing / maxFiringCount) * 40}
                    <rect
                        x={PADDING + TREE_WIDTH + 4}
                        y={yScale(leaf.y) - 4}
                        width={barWidth}
                        height={8}
                        class="firing-bar"
                    >
                        <title>{leaf.key}: {firing} activations</title>
                    </rect>
                    <text
                        x={PADDING + TREE_WIDTH + 48}
                        y={yScale(leaf.y)}
                        class="leaf-label"
                        dominant-baseline="central"
                    >
                        <title>{leaf.key}: {firing} activations</title>
                        {getLabel(leaf.key)}
                    </text>
                {/each}

                <!-- Iteration axis labels -->
                <text x={PADDING + TREE_WIDTH} y={svgHeight - 2} class="axis-label" text-anchor="end">0 (early)</text>
                <text x={PADDING} y={svgHeight - 2} class="axis-label" text-anchor="start">{maxMergeIter} (late)</text>
            </svg>
        </div>
    {/if}
</div>

<style>
    .dendrogram-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .section-title {
        font-size: var(--text-base);
        font-weight: 600;
        margin: 0;
    }

    .section-description {
        font-size: var(--text-xs);
        color: var(--text-muted);
        line-height: 1.4;
        max-width: 600px;
    }

    .status {
        font-size: var(--text-sm);
        color: var(--text-muted);
        padding: var(--space-2);
    }

    .status.error {
        color: var(--status-error);
    }

    .dendrogram-scroll {
        overflow: auto;
    }

    .dendrogram-svg {
        display: block;
    }

    .tree-line {
        stroke: var(--accent-primary);
        stroke-width: 1.5;
        fill: none;
    }

    .leaf-label {
        font-size: 11px;
        font-family: var(--font-sans);
        fill: var(--text-secondary);
    }

    .axis-label {
        font-size: 10px;
        font-family: var(--font-mono);
        fill: var(--text-muted);
    }

    .merge-node {
        fill: var(--accent-primary);
        opacity: 0.6;
        cursor: default;
    }

    .merge-node:hover {
        opacity: 1;
        r: 6;
    }

    .firing-bar {
        fill: var(--accent-primary);
        opacity: 0.4;
    }
</style>
