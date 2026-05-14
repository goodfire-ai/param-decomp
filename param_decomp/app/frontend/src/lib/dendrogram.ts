/**
 * Dendrogram tree construction from merge iteration data.
 * Used by ClusterDendrogram (full visualization) and ClusterMiniMatrix (leaf ordering).
 */

import type { MergePairIteration } from "./api/clusters";

export type LeafNode = { type: "leaf"; key: string; y: number };
export type InternalNode = {
    type: "internal";
    left: TreeNode;
    right: TreeNode;
    mergeIter: number;
    y: number;
};
export type TreeNode = LeafNode | InternalNode;

/** Build dendrogram from pairwise merge iterations via single-linkage. */
export function buildDendrogram(componentKeys: string[], pairs: MergePairIteration[]): TreeNode | null {
    if (componentKeys.length === 0) return null;
    if (componentKeys.length === 1) return { type: "leaf", key: componentKeys[0], y: 0 };

    const sorted = pairs
        .filter((p) => p.merge_iteration >= 0)
        .sort((a, b) => a.merge_iteration - b.merge_iteration);

    const parent = new Map<string, string>();
    const nodes = new Map<string, TreeNode>();
    let leafY = 0;

    for (const key of componentKeys) {
        parent.set(key, key);
        nodes.set(key, { type: "leaf", key, y: leafY });
        leafY++;
    }

    function find(x: string): string {
        let root = x;
        while (parent.get(root) !== root) root = parent.get(root)!;
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

        const mergedKey = `${rootA}|${rootB}`;
        parent.set(rootA, mergedKey);
        parent.set(rootB, mergedKey);
        parent.set(mergedKey, mergedKey);
        nodes.set(mergedKey, merged);
    }

    const roots = new Set<string>();
    for (const key of componentKeys) {
        roots.add(find(key));
    }
    const rootNodes = [...roots].map((r) => nodes.get(r)!);
    if (rootNodes.length === 1) return rootNodes[0];

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

/** Assign leaf y-positions by in-order traversal (so tree doesn't cross). */
export function assignLeafPositions(node: TreeNode, startY: number): number {
    if (node.type === "leaf") {
        node.y = startY;
        return startY + 1;
    }
    const nextY = assignLeafPositions(node.left, startY);
    const finalY = assignLeafPositions(node.right, nextY);
    node.y = (node.left.y + node.right.y) / 2;
    return finalY;
}

/** Collect all leaves in in-order traversal order. */
export function collectLeaves(node: TreeNode): LeafNode[] {
    if (node.type === "leaf") return [node];
    return [...collectLeaves(node.left), ...collectLeaves(node.right)];
}

/** Get the dendrogram leaf order as keys. Returns input unchanged if tree can't be built. */
export function dendrogramLeafOrder(componentKeys: string[], pairs: MergePairIteration[]): string[] {
    const root = buildDendrogram(componentKeys, pairs);
    if (!root) return componentKeys;
    assignLeafPositions(root, 0);
    return collectLeaves(root).map((l) => l.key);
}

export function collectInternalNodes(node: TreeNode): InternalNode[] {
    if (node.type === "leaf") return [];
    return [node, ...collectInternalNodes(node.left), ...collectInternalNodes(node.right)];
}

export function maxIter(node: TreeNode): number {
    if (node.type === "leaf") return 0;
    return Math.max(node.mergeIter, maxIter(node.left), maxIter(node.right));
}

export type SvgLine = { x1: number; y1: number; x2: number; y2: number; mergeIter: number };

export function generateLines(
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
        { x1: nodeX, y1: yScale(leftChild.y), x2: nodeX, y2: yScale(rightChild.y), mergeIter: node.mergeIter },
        { x1: nodeX, y1: yScale(leftChild.y), x2: leftX, y2: yScale(leftChild.y), mergeIter: node.mergeIter },
        {
            x1: nodeX,
            y1: yScale(rightChild.y),
            x2: rightChildX,
            y2: yScale(rightChild.y),
            mergeIter: node.mergeIter,
        },
    ];

    return [...lines, ...generateLines(leftChild, xScale, yScale, rightX), ...generateLines(rightChild, xScale, yScale, rightX)];
}
