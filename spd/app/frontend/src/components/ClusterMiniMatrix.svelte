<script lang="ts">
    import { onMount } from "svelte";
    import {
        fetchClusteringCoactivation,
        fetchMergeIterations,
        type ClusterPairwiseResponse,
        type MergeIterationsResponse,
    } from "../lib/api/clusters";
    import {
        buildDendrogram,
        assignLeafPositions,
        collectLeaves,
        generateLines,
        dendrogramLeafOrder,
    } from "../lib/dendrogram";

    type ComponentMember = { layer: string; cIdx: number };

    type Props = {
        members: ComponentMember[];
        clusteringRunId: string;
        iteration: number;
    };

    let { members, clusteringRunId, iteration }: Props = $props();

    const MAX_PREVIEW_MEMBERS = 24;
    const CANVAS_SIZE = 336;

    type DataState =
        | { status: "idle" }
        | { status: "loading" }
        | { status: "loaded"; response: ClusterPairwiseResponse }
        | { status: "error"; error: string };

    let dataState: DataState = $state({ status: "idle" });
    let canvasEl: HTMLCanvasElement | null = $state(null);
    let rootEl: HTMLDivElement | null = $state(null);
    let isVisible = $state(false);

    const previewMembers = $derived.by(() =>
        [...members]
            .sort((a, b) => {
                if (a.layer !== b.layer) return a.layer < b.layer ? -1 : 1;
                return a.cIdx - b.cIdx;
            })
            .slice(0, MAX_PREVIEW_MEMBERS),
    );
    const previewKeys = $derived(previewMembers.map((member) => `${member.layer}:${member.cIdx}`));

    function pairKey(a: string, b: string): string {
        return a < b ? `${a}|${b}` : `${b}|${a}`;
    }

    let leafOrder = $state<string[] | null>(null);
    let mergeData = $state<MergeIterationsResponse | null>(null);

    $effect(() => {
        const keys = previewKeys;
        const runId = clusteringRunId;
        const iter = iteration;
        if (!isVisible || keys.length < 2 || dataState.status !== "idle") return;

        dataState = { status: "loading" };
        Promise.all([
            fetchClusteringCoactivation(keys, runId),
            fetchMergeIterations(keys, runId, iter),
        ])
            .then(([coactResponse, mergeResponse]) => {
                mergeData = mergeResponse;
                leafOrder = dendrogramLeafOrder(keys, mergeResponse.pairs);
                dataState = { status: "loaded", response: coactResponse };
            })
            .catch((error) => {
                dataState = { status: "error", error: String(error) };
            });
    });

    const orderedKeys = $derived(leafOrder ?? previewKeys);

    const DENDRO_WIDTH = 576;
    const DENDRO_PADDING = 12;

    const miniDendro = $derived.by(() => {
        if (!mergeData || orderedKeys.length < 2) return null;
        const root = buildDendrogram(orderedKeys, mergeData.pairs);
        if (!root) return null;
        assignLeafPositions(root, 0);
        const leaves = collectLeaves(root);
        const nLeaves = leaves.length;
        const totalIters = mergeData.total_iterations;
        const height = CANVAS_SIZE;
        const treeRight = DENDRO_WIDTH - DENDRO_PADDING;
        const treeLeft = DENDRO_PADDING;
        const xScale = (iter: number) => {
            if (totalIters === 0) return treeLeft;
            return treeRight - (iter / totalIters) * (treeRight - treeLeft);
        };
        const yScale = (pos: number) => DENDRO_PADDING + (pos / Math.max(nLeaves - 1, 1)) * (height - DENDRO_PADDING * 2);
        const lines = generateLines(root, xScale, yScale, treeRight);
        const totalLabel = totalIters >= 1000 ? `${(totalIters / 1000).toFixed(totalIters % 1000 === 0 ? 0 : 1)}k` : String(totalIters);
        return { lines, height, totalLabel };
    });

    $effect(() => {
        if (!canvasEl || dataState.status !== "loaded") return;

        const keys = orderedKeys;
        const size = keys.length;
        const ctx = canvasEl.getContext("2d");
        if (!ctx) return;

        canvasEl.width = size;
        canvasEl.height = size;

        const lookup = new Map<string, number>();
        for (const pair of dataState.response.pairs) {
            lookup.set(pairKey(pair.key_a, pair.key_b), pair.jaccard);
        }

        const image = ctx.createImageData(size, size);
        for (let row = 0; row < size; row++) {
            for (let col = 0; col < size; col++) {
                const offset = (row * size + col) * 4;
                let r = 242;
                let g = 244;
                let b = 247;
                const value = row === col ? 1 : (lookup.get(pairKey(keys[row], keys[col])) ?? 0);
                const alpha = Math.max(0, Math.min(value, 1));
                r = Math.round(242 - alpha * 183);
                g = Math.round(244 - alpha * 114);
                b = Math.round(247 - alpha * 1);
                image.data[offset] = r;
                image.data[offset + 1] = g;
                image.data[offset + 2] = b;
                image.data[offset + 3] = 255;
            }
        }

        ctx.putImageData(image, 0, 0);
    });

    const stats = $derived.by(() => {
        if (dataState.status !== "loaded" || dataState.response.pairs.length === 0) {
            return { mean: 0, max: 0 };
        }
        let sum = 0;
        let max = 0;
        for (const pair of dataState.response.pairs) {
            sum += pair.jaccard;
            max = Math.max(max, pair.jaccard);
        }
        return { mean: sum / dataState.response.pairs.length, max };
    });

    const titleText = $derived.by(() => {
        const shown = previewKeys.length;
        const suffix = shown < members.length ? ` first ${shown}/${members.length} members` : ` ${shown} members`;
        if (dataState.status === "loaded") {
            return `Clustering Jaccard mini-matrix for${suffix}. Mean ${stats.mean.toFixed(3)}, max ${stats.max.toFixed(3)}.`;
        }
        if (dataState.status === "error") {
            return `Failed to load mini-matrix for${suffix}.`;
        }
        return `Clustering Jaccard mini-matrix for${suffix}.`;
    });

    onMount(() => {
        if (!rootEl) return;
        const observer = new IntersectionObserver(
            (entries) => {
                if (entries.some((entry) => entry.isIntersecting)) {
                    isVisible = true;
                    observer.disconnect();
                }
            },
            { rootMargin: "120px" },
        );
        observer.observe(rootEl);
        return () => observer.disconnect();
    });
</script>

<div class="mini-cluster-preview" bind:this={rootEl} title={titleText}>
    <div class="mini-visuals">
        {#if previewKeys.length < 2}
            <div class="mini-empty">1x</div>
        {:else if dataState.status === "error"}
            <div class="mini-empty">!</div>
        {:else if dataState.status !== "loaded"}
            <div class="mini-loading"></div>
        {:else}
            {#if miniDendro}
                <div class="dendro-col">
                    <svg class="mini-dendro" width={DENDRO_WIDTH} height={miniDendro.height}>
                        {#each miniDendro.lines as line, i (i)}
                            <line x1={line.x1} y1={line.y1} x2={line.x2} y2={line.y2} class="dendro-line" />
                        {/each}
                    </svg>
                    <div class="dendro-axis" style={`width: ${DENDRO_WIDTH}px; padding: 0 ${DENDRO_PADDING}px;`}>
                        <span>{miniDendro.totalLabel}</span>
                        <span>0</span>
                    </div>
                </div>
            {/if}
            <canvas
                bind:this={canvasEl}
                class="mini-canvas"
                width={CANVAS_SIZE}
                height={CANVAS_SIZE}
                style={`width: ${CANVAS_SIZE}px; height: ${CANVAS_SIZE}px;`}
            ></canvas>
        {/if}
    </div>
    <div class="mini-meta">
        <span>J</span>
        <span>{dataState.status === "loaded" ? stats.mean.toFixed(2) : "..."}</span>
    </div>
</div>

<style>
    .mini-cluster-preview {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        flex-shrink: 0;
    }

    .mini-visuals {
        display: flex;
        align-items: stretch;
        gap: 1px;
    }

    .mini-canvas,
    .mini-loading,
    .mini-empty {
        width: 336px;
        height: 336px;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--bg-base);
        image-rendering: pixelated;
    }

    .dendro-col {
        display: flex;
        flex-direction: column;
        flex-shrink: 0;
    }

    .mini-dendro {
        flex-shrink: 0;
    }

    .dendro-axis {
        display: flex;
        justify-content: space-between;
        font-size: 9px;
        font-family: var(--font-mono);
        color: var(--text-muted);
        border-top: 1px solid var(--border-default);
        padding-top: 2px;
        box-sizing: border-box;
    }

    .dendro-line {
        stroke: var(--accent-primary);
        stroke-width: 1.5;
        fill: none;
        opacity: 0.6;
    }

    .mini-loading {
        background:
            linear-gradient(135deg, var(--bg-inset) 25%, transparent 25%) -8px 0/16px 16px,
            linear-gradient(225deg, var(--bg-inset) 25%, transparent 25%) -8px 0/16px 16px,
            linear-gradient(315deg, var(--bg-inset) 25%, transparent 25%) 0 0/16px 16px,
            linear-gradient(45deg, var(--bg-inset) 25%, var(--bg-base) 25%) 0 0/16px 16px;
    }

    .mini-empty {
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--text-muted);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
    }

    .mini-meta {
        display: flex;
        gap: 4px;
        color: var(--text-muted);
        font-size: 10px;
        font-family: var(--font-mono);
    }
</style>
