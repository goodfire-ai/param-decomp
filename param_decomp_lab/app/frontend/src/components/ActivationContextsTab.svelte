<script lang="ts">
    import { getContext, onMount } from "svelte";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import ActivationContextsViewer from "./ActivationContextsViewer.svelte";
    import StatusText from "./ui/StatusText.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    onMount(() => {
        runState.loadActivationContextsSummary();
    });

    const summary = $derived(runState.activationContextsSummary);
</script>

<div class="tab-wrapper">
    {#if summary.status === "uninitialized" || summary.status === "loading"}
        <div class="loading">Loading activation contexts summary...</div>
    {:else if summary.status === "error"}
        <StatusText>Error loading summary: {String(summary.error)}</StatusText>
    {:else if summary.data === null}
        <StatusText>No harvest data available. Run postprocessing first.</StatusText>
    {:else}
        <ActivationContextsViewer activationContextsSummary={summary.data} />
    {/if}
</div>

<style>
    .tab-wrapper {
        height: 100%;
        padding: var(--space-6);
    }

    .loading {
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
    }
</style>
