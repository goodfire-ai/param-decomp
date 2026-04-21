<script lang="ts">
    import { getContext } from "svelte";
    import type { OutputProbability, PinnedNode, EdgeData } from "../../lib/promptAttributionsTypes";
    import ComponentNodeCard from "./ComponentNodeCard.svelte";
    import OutputNodeCard from "./OutputNodeCard.svelte";
    import { RUN_KEY, type RunContext } from "../../lib/useRun.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    type Props = {
        stagedNodes: PinnedNode[];
        outputProbs: Record<string, OutputProbability>;
        nodeCiVals: Record<string, number>;
        nodeSubcompActs: Record<string, number>;
        tokens: string[];
        edgesBySource: Map<string, EdgeData[]>;
        edgesByTarget: Map<string, EdgeData[]>;
        onStagedNodesChange: (nodes: PinnedNode[]) => void;
    };

    let {
        stagedNodes,
        outputProbs,
        nodeCiVals,
        nodeSubcompActs,
        tokens,
        edgesBySource,
        edgesByTarget,
        onStagedNodesChange,
    }: Props = $props();

    function clearAll() {
        onStagedNodesChange([]);
    }

    function unstageNode(node: PinnedNode) {
        onStagedNodesChange(stagedNodes.filter((n) => n !== node));
    }

    function toggleComponentPinned(layer: string, cIdx: number, seqIdx: number) {
        const alreadyPinned = stagedNodes.some((n) => n.layer === layer && n.cIdx === cIdx && n.seqIdx === seqIdx);
        if (alreadyPinned) {
            onStagedNodesChange(stagedNodes.filter((n) => n.layer !== layer || n.cIdx !== cIdx || n.seqIdx !== seqIdx));
        } else {
            onStagedNodesChange([...stagedNodes, { layer, cIdx, seqIdx }]);
        }
    }

    function getTokenAtPosition(seqIdx: number): string {
        if (seqIdx < 0 || seqIdx >= tokens.length) {
            throw new Error(`StagedNodesPanel: seqIdx ${seqIdx} out of bounds for tokens length ${tokens.length}`);
        }
        return tokens[seqIdx];
    }
</script>

{#if stagedNodes.length > 0}
    <div class="staged-container">
        <div class="staged-container-header">
            <span>Pinned Components ({stagedNodes.length})</span>
            <button onclick={clearAll}>Clear all</button>
        </div>

        <!-- Key in {#each} ensures ComponentNodeCard remounts when node identity changes -->
        <div class="staged-items">
            {#each stagedNodes as node, idx (`${node.layer}:${node.seqIdx}:${node.cIdx}-${idx}`)}
                {@const token = getTokenAtPosition(node.seqIdx)}
                {@const isOutput = node.layer === "output"}
                {@const isWte = node.layer === "embed"}
                {@const isComponent = !isWte && !isOutput}
                {@const nodeKey = `${node.layer}:${node.seqIdx}:${node.cIdx}`}
                {@const ciVal = isComponent ? (nodeCiVals[nodeKey] ?? null) : null}
                {@const subcompAct = isComponent ? (nodeSubcompActs[nodeKey] ?? null) : null}
                {@const clusterId = isComponent ? runState.getClusterId(node.layer, node.cIdx) : undefined}
                <div class="staged-item">
                    <div class="staged-header">
                        <div class="node-info">
                            <strong>{node.layer}:{node.seqIdx}:{node.cIdx}</strong>
                            <span class="token-preview">"{token}"</span>
                            {#if clusterId !== undefined}
                                <span class="cluster-id">Cluster: {clusterId ?? "null"}</span>
                            {/if}
                        </div>
                        <button class="unstage-btn" onclick={() => unstageNode(node)}>✕</button>
                    </div>

                    {#if isWte}
                        <div class="token-display">"{token}"</div>
                        <p class="wte-info">Input embedding at position {node.seqIdx}</p>
                    {:else if isOutput}
                        <OutputNodeCard cIdx={node.cIdx} {outputProbs} seqIdx={node.seqIdx} />
                    {:else}
                        <ComponentNodeCard
                            layer={node.layer}
                            cIdx={node.cIdx}
                            seqIdx={node.seqIdx}
                            {ciVal}
                            {subcompAct}
                            {token}
                            {edgesBySource}
                            {edgesByTarget}
                            {tokens}
                            {outputProbs}
                            onPinComponent={toggleComponentPinned}
                        />
                    {/if}
                </div>
            {/each}
        </div>
    </div>
{/if}

<style>
    .staged-container {
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        padding: var(--space-3);
    }

    .staged-container-header {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        font-weight: 600;
        color: var(--text-secondary);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: var(--space-2);
    }

    .staged-container-header button {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        color: var(--text-secondary);
    }

    .staged-container-header button:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    .staged-header {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin-bottom: var(--space-2);
    }

    .unstage-btn {
        background: var(--bg-inset);
        color: var(--text-primary);
        border: 1px solid var(--border-strong);
    }

    .unstage-btn:hover {
        background: var(--status-negative);
        color: var(--bg-base);
        border-color: var(--status-negative);
    }

    .token-display {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
        margin-bottom: var(--space-2);
    }

    .staged-items {
        display: flex;
        flex-direction: row;
        gap: var(--space-3);
        overflow-x: auto;
    }

    .staged-item {
        flex-shrink: 0;
        width: fit-content;
        max-width: 800px;
        border: 1px solid var(--border-default);
        padding: var(--space-3);
        background: var(--bg-elevated);
    }

    .wte-info {
        margin: var(--space-2) 0 0 0;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }
</style>
