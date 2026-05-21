<script lang="ts">
    import { getContext, onMount } from "svelte";
    import { SvelteMap, SvelteSet } from "svelte/reactivity";
    import type { Loadable } from "../lib";
    import {
        getSubruns,
        getSubrunInterpretations,
        type CompareInterpretationHeadline,
        type SubrunSummary,
    } from "../lib/api";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import AutointerpComparer from "./AutointerpComparer.svelte";
    import SubrunSelector from "./SubrunSelector.svelte";
    import StatusText from "./ui/StatusText.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    let subrunsState = $state<Loadable<SubrunSummary[]>>({ status: "uninitialized" });
    let selectedIds = new SvelteSet<string>();

    // Headline cache: subrunId → Record<componentKey, headline>
    let headlinesCache = new SvelteMap<string, Record<string, CompareInterpretationHeadline>>();
    let headlinesLoading = new SvelteSet<string>();

    onMount(() => {
        subrunsState = { status: "loading" };
        getSubruns()
            .then((data) => {
                subrunsState = { status: "loaded", data };
            })
            .catch((error) => {
                subrunsState = { status: "error", error };
            });

        // Load activation contexts summary if not already loaded
        runState.loadActivationContextsSummary();
    });

    // When selection changes, fetch headlines for newly selected subruns
    $effect(() => {
        for (const id of selectedIds) {
            if (headlinesCache.has(id) || headlinesLoading.has(id)) continue;
            headlinesLoading.add(id);
            getSubrunInterpretations(id)
                .then((data) => {
                    headlinesCache.set(id, data);
                })
                .catch((error) => {
                    console.error(`Failed to load headlines for ${id}:`, error);
                })
                .finally(() => {
                    headlinesLoading.delete(id);
                });
        }
    });

    const summary = $derived(runState.activationContextsSummary);

    const selectedSubruns = $derived.by(() => {
        if (subrunsState.status !== "loaded") return [];
        return subrunsState.data.filter((s) => selectedIds.has(s.subrun_id));
    });
</script>

<div class="tab-wrapper">
    {#if subrunsState.status === "uninitialized" || subrunsState.status === "loading"}
        <div class="loading">Loading subruns...</div>
    {:else if subrunsState.status === "error"}
        <StatusText>Error loading subruns: {String(subrunsState.error)}</StatusText>
    {:else if subrunsState.data.length === 0}
        <StatusText>No completed autointerp subruns found.</StatusText>
    {:else}
        <SubrunSelector subruns={subrunsState.data} {selectedIds} />

        {#if selectedSubruns.length > 0}
            {#if summary.status === "loaded" && summary.data !== null}
                <AutointerpComparer activationContextsSummary={summary.data} {selectedSubruns} {headlinesCache} />
            {:else if summary.status === "loading" || summary.status === "uninitialized"}
                <div class="loading">Loading component data...</div>
            {:else if summary.status === "error"}
                <StatusText>Error loading component data: {String(summary.error)}</StatusText>
            {:else}
                <StatusText>No harvest data available. Run postprocessing first.</StatusText>
            {/if}
        {:else}
            <div class="empty-hint">Select one or more subruns to compare interpretations.</div>
        {/if}
    {/if}
</div>

<style>
    .tab-wrapper {
        height: 100%;
        padding: var(--space-6);
        display: flex;
        flex-direction: column;
        gap: var(--space-4);
    }

    .loading {
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
    }

    .empty-hint {
        text-align: center;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
        padding: var(--space-6);
    }
</style>
