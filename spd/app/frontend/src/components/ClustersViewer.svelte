<script lang="ts">
    import { getContext } from "svelte";
    import { RUN_KEY, type RunContext, type ClusterMappingData } from "../lib/useRun.svelte";
    import ClusterComponentCard from "./ClusterComponentCard.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    type Props = {
        clusterMappingData: ClusterMappingData;
    };

    let { clusterMappingData }: Props = $props();

    type ComponentMember = { layer: string; cIdx: number };

    /** "unclustered" is a sentinel for the singletons group */
    let selectedClusterId = $state<number | "unclustered" | null>(null);

    /** Invert the mapping: cluster ID -> list of component members */
    const clusterGroups = $derived.by(() => {
        const groups: Record<number, ComponentMember[]> = {};
        const singletons: ComponentMember[] = [];

        for (const [key, clusterId] of Object.entries(clusterMappingData)) {
            const lastColon = key.lastIndexOf(":");
            const layer = key.substring(0, lastColon);
            const cIdx = parseInt(key.substring(lastColon + 1));
            const member: ComponentMember = { layer, cIdx };

            if (clusterId === null) {
                singletons.push(member);
            } else {
                if (!groups[clusterId]) {
                    groups[clusterId] = [];
                }
                groups[clusterId].push(member);
            }
        }

        const sorted = Object.entries(groups)
            .map(([id, members]) => [Number(id), members] as [number, ComponentMember[]])
            .sort((a, b) => b[1].length - a[1].length);
        return { sorted, singletons };
    });

    const selectedMembers = $derived.by((): ComponentMember[] => {
        if (selectedClusterId === null) return [];
        if (selectedClusterId === "unclustered") return clusterGroups.singletons;
        const group = clusterGroups.sorted.find(([id]) => id === selectedClusterId);
        return group ? group[1] : [];
    });

    function getPreviewLabels(members: ComponentMember[]): string[] {
        const labels: string[] = [];
        for (const member of members) {
            if (labels.length >= 3) break;
            const key = `${member.layer}:${member.cIdx}`;
            const interp = runState.getInterpretation(key);
            if (interp.status === "loaded" && interp.data.status === "generated") {
                labels.push(interp.data.data.label);
            }
        }
        return labels;
    }
</script>

<div class="clusters-viewer">
    {#if selectedClusterId === null}
        <div class="cluster-list">
            <h2 class="list-header">Clusters ({clusterGroups.sorted.length})</h2>
            {#each clusterGroups.sorted as [clusterId, members] (clusterId)}
                {@const previewLabels = getPreviewLabels(members)}
                <button class="cluster-row" onclick={() => (selectedClusterId = clusterId)}>
                    <div class="cluster-row-main">
                        <span class="cluster-id">Cluster {clusterId}</span>
                        <span class="cluster-count">{members.length} components</span>
                    </div>
                    {#if previewLabels.length > 0}
                        <div class="preview-labels">
                            {#each previewLabels as label, i (i)}
                                <span class="preview-pill">{label}</span>
                            {/each}
                        </div>
                    {/if}
                </button>
            {/each}
            {#if clusterGroups.singletons.length > 0}
                <button class="cluster-row singletons-row" onclick={() => (selectedClusterId = "unclustered")}>
                    <div class="cluster-row-main">
                        <span class="cluster-id">Unclustered</span>
                        <span class="cluster-count">{clusterGroups.singletons.length} components</span>
                    </div>
                </button>
            {/if}
        </div>
    {:else}
        <div class="cluster-detail">
            <div class="detail-header">
                <button class="back-button" onclick={() => (selectedClusterId = null)}>&lt; Back</button>
                <h2 class="detail-title">
                    {selectedClusterId === "unclustered" ? "Unclustered" : `Cluster ${selectedClusterId}`}
                </h2>
                <span class="detail-count">{selectedMembers.length} components</span>
            </div>
            <div class="cluster-cards">
                {#each selectedMembers as member (`${member.layer}:${member.cIdx}`)}
                    <div class="cluster-card-item">
                        <ClusterComponentCard layer={member.layer} cIdx={member.cIdx} />
                    </div>
                {/each}
            </div>
        </div>
    {/if}
</div>

<style>
    .clusters-viewer {
        font-family: var(--font-sans);
        color: var(--text-primary);
    }

    /* Cluster list */
    .list-header {
        font-size: var(--text-lg);
        font-weight: 600;
        margin: 0 0 var(--space-4) 0;
    }

    .cluster-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .cluster-row {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        padding: var(--space-3) var(--space-4);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        cursor: pointer;
        text-align: left;
        font: inherit;
        color: inherit;
        transition:
            background var(--transition-normal),
            border-color var(--transition-normal);
    }

    .cluster-row:hover {
        background: var(--bg-elevated);
        border-color: var(--border-strong);
    }

    .cluster-row-main {
        display: flex;
        align-items: center;
        gap: var(--space-3);
    }

    .cluster-id {
        font-weight: 600;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .cluster-count {
        font-size: var(--text-sm);
        color: var(--text-muted);
    }

    .preview-labels {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-1);
    }

    .preview-pill {
        font-size: var(--text-xs);
        padding: var(--space-1) var(--space-2);
        background: var(--bg-inset);
        border-radius: var(--radius-sm);
        color: var(--text-secondary);
        max-width: 300px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .singletons-row {
        border-style: dashed;
    }

    /* Cluster detail */
    .cluster-detail {
        display: flex;
        flex-direction: column;
        gap: var(--space-4);
    }

    .detail-header {
        display: flex;
        align-items: center;
        gap: var(--space-3);
    }

    .back-button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        cursor: pointer;
        font: inherit;
        font-size: var(--text-sm);
        color: var(--text-secondary);
    }

    .back-button:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
        border-color: var(--border-strong);
    }

    .detail-title {
        font-size: var(--text-lg);
        font-weight: 600;
        margin: 0;
    }

    .detail-count {
        font-size: var(--text-sm);
        color: var(--text-muted);
    }

    .cluster-cards {
        display: flex;
        flex-direction: row;
        gap: var(--space-3);
        overflow-x: auto;
    }

    .cluster-card-item {
        flex-shrink: 0;
        width: fit-content;
        max-width: 800px;
        border: 1px solid var(--border-default);
        padding: var(--space-3);
        background: var(--bg-elevated);
    }
</style>
