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

    let showWhichView = $derived(runState.run.status === "loaded" ? "run-view" : "run-selector");

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
{:else}
    <RunView />
{/if}
