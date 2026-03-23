<script lang="ts">
    import { getContext } from "svelte";
    import { RUN_KEY, type RunContext, type ClusterMappingData } from "../lib/useRun.svelte";
    import ClusterComponentCard from "./ClusterComponentCard.svelte";
    import ClusterCorrelationMatrix from "./ClusterCorrelationMatrix.svelte";
    import ClusterDendrogram from "./ClusterDendrogram.svelte";
    import ClusterFullMatrix from "./ClusterFullMatrix.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    type Props = {
        clusterMappingData: ClusterMappingData;
        clusteringRunId: string;
        iteration: number;
    };

    let { clusterMappingData, clusteringRunId, iteration }: Props = $props();

    type ComponentMember = { layer: string; cIdx: number };

    type SubTab = "matrix" | "list";
    let activeSubTab = $state<SubTab>("matrix");

    /** "unclustered" is a sentinel for the singletons group */
    let selectedClusterId = $state<number | "unclustered" | null>(null);
    let orderedMembers = $state<ComponentMember[]>([]);

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
                (groups[clusterId] ??= []).push(member);
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

    const layerGroups = $derived.by((): { layer: string; members: ComponentMember[] }[] => {
        const groups: Record<string, ComponentMember[]> = {};
        for (const m of selectedMembers) {
            (groups[m.layer] ??= []).push(m);
        }
        return Object.entries(groups)
            .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
            .map(([layer, members]) => ({
                layer,
                members: members.sort((a, b) => a.cIdx - b.cIdx),
            }));
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
        <div class="subtab-bar">
            <button class="subtab-btn" class:active={activeSubTab === "matrix"} onclick={() => (activeSubTab = "matrix")}>Matrix</button>
            <button class="subtab-btn" class:active={activeSubTab === "list"} onclick={() => (activeSubTab = "list")}>List ({clusterGroups.sorted.length})</button>
        </div>
        {#if activeSubTab === "matrix"}
            <div class="matrix-pane">
                <ClusterFullMatrix {clusterMappingData} onSelectCluster={(id) => { selectedClusterId = id; }} />
            </div>
        {:else}
        <div class="cluster-list">
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
        {/if}
    {:else}
        <div class="cluster-detail">
            <div class="detail-header">
                <div class="detail-header-left">
                    <button class="back-button" onclick={() => (selectedClusterId = null)}>&lt; Back</button>
                    <h2 class="detail-title">
                        {selectedClusterId === "unclustered" ? "Unclustered" : `Cluster ${selectedClusterId}`}
                    </h2>
                    <span class="detail-count">{selectedMembers.length} components</span>
                </div>
                {#if layerGroups.length > 0}
                    <div class="layer-breakdown">
                        {#each layerGroups as group (group.layer)}
                            <div class="layer-group">
                                <span class="layer-group-label">{group.layer}</span>
                                <div class="layer-group-pills">
                                    {#each group.members as member (`${group.layer}:${member.cIdx}`)}
                                        {@const key = `${member.layer}:${member.cIdx}`}
                                        {@const interp = runState.getInterpretation(key)}
                                        <span
                                            class="component-idx-pill"
                                            title={interp.status === "loaded" && interp.data.status === "generated"
                                                ? `${key}: ${interp.data.data.label}`
                                                : key}
                                        >
                                            {member.cIdx}
                                        </span>
                                    {/each}
                                </div>
                            </div>
                        {/each}
                    </div>
                {/if}
            </div>
            {#if selectedMembers.length >= 2}
                <div class="grid-and-dendrogram">
                    <ClusterCorrelationMatrix members={orderedMembers.length > 0 ? orderedMembers : selectedMembers} {clusteringRunId} />
                    <ClusterDendrogram members={selectedMembers} {clusteringRunId} {iteration} onLeafOrder={(order) => (orderedMembers = order)} />
                </div>
            {/if}
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
        height: 100%;
        display: flex;
        flex-direction: column;
        min-height: 0;
    }

    .subtab-bar {
        display: flex;
        gap: var(--space-1);
        flex-shrink: 0;
        margin-bottom: var(--space-3);
    }

    .subtab-btn {
        padding: var(--space-2) var(--space-4);
        font: inherit;
        font-size: var(--text-sm);
        font-weight: 500;
        background: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        cursor: pointer;
        color: var(--text-muted);
        transition:
            color var(--transition-normal),
            border-color var(--transition-normal);
    }

    .subtab-btn:hover {
        color: var(--text-secondary);
    }

    .subtab-btn.active {
        color: var(--text-primary);
        border-bottom-color: var(--accent-primary);
    }

    .matrix-pane {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }

    .grid-and-dendrogram {
        display: flex;
        gap: var(--space-4);
        align-items: flex-start;
    }

    /* Cluster list */
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
        justify-content: space-between;
        gap: var(--space-3);
    }

    .detail-header-left {
        display: flex;
        align-items: center;
        gap: var(--space-3);
    }

    .layer-breakdown {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .layer-group {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .layer-group-label {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-secondary);
        white-space: nowrap;
    }

    .layer-group-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 2px;
    }

    .component-idx-pill {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        padding: 1px var(--space-1);
        background: var(--bg-inset);
        border-radius: var(--radius-sm);
        color: var(--accent-primary);
        font-weight: 600;
        cursor: default;
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
