<script lang="ts">
    import * as api from "./lib/api";
    import { useRun, RUN_KEY } from "./lib/useRun.svelte";
    import type { Loadable } from "./lib";
    import { onMount, setContext } from "svelte";
    import RunSelector from "./components/RunSelector.svelte";
    import RunView from "./components/RunView.svelte";

    // Initialize run state and provide via context for all child components
    const runState = useRun();
    setContext(RUN_KEY, runState);

    let backendUser = $state<Loadable<string>>({ status: "uninitialized" });

    // The circuit builder works without a W&B run — let users skip the selector.
    let skipToTabs = $state(false);
    let showWhichView = $derived(
        runState.run.status === "loaded" || skipToTabs ? "run-view" : "run-selector",
    );

    async function handleLoadRun(wandbPath: string, contextLength: number) {
        await runState.loadRun(wandbPath, contextLength);
    }

    onMount(() => {
        runState.syncStatus();
        api.whoami().then((user) => (backendUser = { status: "loaded", data: user }));
    });
</script>

{#if showWhichView === "run-selector"}
    <RunSelector
        onSelect={handleLoadRun}
        isLoading={runState.run.status === "loading"}
        username={backendUser.status === "loaded" ? backendUser.data : null}
    />
    <button class="circuit-builder-skip" onclick={() => (skipToTabs = true)}>
        Circuit Builder → <span class="sub">(no W&B run needed)</span>
    </button>
{:else}
    <RunView />
{/if}

<style>
    .circuit-builder-skip {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 10;
        padding: 0.5rem 1rem;
        border: 1px solid var(--border-default, #bbb);
        border-radius: 6px;
        background: var(--bg-surface, #fff);
        cursor: pointer;
        font-size: 0.9rem;
    }
    .circuit-builder-skip:hover {
        background: var(--bg-hover, #f0f4ff);
    }
    .circuit-builder-skip .sub {
        color: var(--text-muted, #888);
        font-size: 0.75rem;
    }
</style>
