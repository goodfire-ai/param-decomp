<script lang="ts">
    import { getContext } from "svelte";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import ClustersViewer from "./ClustersViewer.svelte";
    import StatusText from "./ui/StatusText.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    const clusterMapping = $derived(runState.clusterMapping);
</script>

<div class="tab-wrapper">
    {#if clusterMapping}
        <ClustersViewer clusterMappingData={clusterMapping.data} />
    {:else}
        <StatusText
            >No clusters loaded. Use the cluster path input in the header bar to load a cluster mapping.</StatusText
        >
    {/if}
</div>

<style>
    .tab-wrapper {
        height: 100%;
        padding: var(--space-6);
    }
</style>
