<script lang="ts">
    import { getContext } from "svelte";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import ActivationContextsTab from "./ActivationContextsTab.svelte";
    import ClusterPathInput from "./ClusterPathInput.svelte";
    import ClustersTab from "./ClustersTab.svelte";
    import DatasetExplorerTab from "./DatasetExplorerTab.svelte";
    import InvestigationsTab from "./InvestigationsTab.svelte";
    import DataSourcesTab from "./DataSourcesTab.svelte";
    import PromptAttributionsTab from "./PromptAttributionsTab.svelte";
    import DisplaySettingsDropdown from "./ui/DisplaySettingsDropdown.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    const datasetSearchEnabled = $derived(
        runState.run?.status === "loaded" && runState.run.data.dataset_search_enabled,
    );

    let activeTab = $state<
        | "prompts"
        | "components"
        | "dataset-search"
        | "model-graph"
        | "data-sources"
        | "investigations"
        | "clusters"
        | null
    >(null);

    $effect(() => {
        if (runState.prompts.status === "loaded" && activeTab === null) {
            activeTab = "prompts";
        }
    });

    $effect(() => {
        if (activeTab === "clusters" && !runState.clusterMapping) {
            activeTab = "prompts";
        }
    });
</script>

<div class="app-layout">
    <header class="top-bar">
        {#if runState.run.status === "loaded"}
            {@const wandbParts = runState.run.data.wandb_path.split("/")}
            <button type="button" class="change-run-button" onclick={() => runState.clearRun()}>&lt;</button>
            <span class="run-path"
                >{runState.run.data.wandb_path}
                <a
                    class="wandb-link"
                    href="https://wandb.ai/{wandbParts[0]}/{wandbParts[1]}/runs/{wandbParts[2]}"
                    target="_blank"
                    rel="noopener">(wandb)</a
                >
            </span>
        {/if}

        <nav class="nav-group">
            <button
                type="button"
                class="tab-button"
                class:active={activeTab === "investigations"}
                onclick={() => (activeTab = "investigations")}
            >
                Investigations
            </button>
            {#if runState.run.status === "loaded" && runState.run.data}
                <button
                    type="button"
                    class="tab-button"
                    class:active={activeTab === "prompts"}
                    onclick={() => (activeTab = "prompts")}
                >
                    Prompts
                </button>
                <button
                    type="button"
                    class="tab-button"
                    class:active={activeTab === "components"}
                    onclick={() => (activeTab = "components")}
                >
                    Components
                </button>
                {#if datasetSearchEnabled}
                    <button
                        type="button"
                        class="tab-button"
                        class:active={activeTab === "dataset-search"}
                        onclick={() => (activeTab = "dataset-search")}
                    >
                        Dataset Explorer
                    </button>
                {/if}
                <button
                    type="button"
                    class="tab-button"
                    class:active={activeTab === "data-sources"}
                    onclick={() => (activeTab = "data-sources")}
                >
                    Data Sources
                </button>
                {#if runState.clusterMapping}
                    <button
                        type="button"
                        class="tab-button"
                        class:active={activeTab === "clusters"}
                        onclick={() => (activeTab = "clusters")}
                    >
                        Clusters
                    </button>
                {/if}
            {/if}
        </nav>

        <div class="top-bar-spacer"></div>
        {#if runState.run.status === "loaded"}
            <div class="cluster-path-input-container">
                <ClusterPathInput />
            </div>
        {/if}
        <DisplaySettingsDropdown />
    </header>

    <main class="main-content">
        {#if runState.run.status === "error"}
            <div class="warning-banner">
                {runState.run.error}
            </div>
        {/if}
        <!-- Investigations tab - always available, doesn't require loaded run -->
        <div class="tab-content" class:hidden={activeTab !== "investigations"}>
            <InvestigationsTab />
        </div>
        {#if runState.prompts.status === "loaded"}
            <!-- Use hidden class instead of conditional rendering to preserve state -->
            <div class="tab-content" class:hidden={activeTab !== "prompts"}>
                <PromptAttributionsTab prompts={runState.prompts.data} />
            </div>
            <div class="tab-content" class:hidden={activeTab !== "components"}>
                <ActivationContextsTab />
            </div>
            {#if datasetSearchEnabled}
                <div class="tab-content" class:hidden={activeTab !== "dataset-search"}>
                    <DatasetExplorerTab />
                </div>
            {/if}
            <div class="tab-content" class:hidden={activeTab !== "data-sources"}>
                <DataSourcesTab />
            </div>
            {#if runState.clusterMapping}
                <div class="tab-content" class:hidden={activeTab !== "clusters"}>
                    <ClustersTab />
                </div>
            {/if}
        {:else if runState.run.status === "loading" || runState.prompts.status === "loading"}
            <div class="empty-state">
                <p>Loading run...</p>
            </div>
        {:else}
            <div class="empty-state">
                <p>Enter a W&B Path above to get started</p>
            </div>
        {/if}
    </main>
</div>

<style>
    .app-layout {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        background: var(--bg-base);
    }

    .top-bar {
        display: flex;
        align-items: stretch;
        background: var(--bg-surface);
        border-bottom: 1px solid var(--border-default);
        flex-shrink: 0;
        min-height: 44px;
    }

    .run-path {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding: 0 var(--space-3);
        border-right: 1px solid var(--border-default);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        color: var(--text-primary);
    }

    .wandb-link {
        color: var(--text-muted);
        text-decoration: none;
        font-size: var(--text-xs);
    }

    .wandb-link:hover {
        color: var(--accent-primary);
    }

    .change-run-button {
        display: flex;
        align-items: center;
        padding: 0 var(--space-3);
        margin: 0;
        background: none;
        border: none;
        border-right: 1px solid var(--border-default);
        border-radius: 0;
        font: inherit;
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-muted);
        cursor: pointer;
    }

    .change-run-button:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
    }

    /* Navigation tabs */
    .nav-group {
        display: flex;
    }

    .tab-button {
        padding: var(--space-2) var(--space-4);
        margin: 0;
        background: none;
        border: none;
        border-right: 1px solid var(--border-default);
        border-radius: 0;
        font: inherit;
        font-weight: 500;
        font-size: var(--text-sm);
        color: var(--text-muted);
        cursor: pointer;
        transition:
            color var(--transition-normal),
            background var(--transition-normal);
    }

    .tab-button:hover {
        color: var(--text-primary);
        background: var(--bg-inset);
    }

    .tab-button.active {
        color: var(--text-primary);
        background: var(--bg-inset);
    }

    .top-bar-spacer {
        flex: 1;
    }

    .cluster-path-input-container {
        flex: 1;
        padding: 0 var(--space-3);
        display: flex;
        align-items: center;
        justify-content: flex-end;
    }

    .main-content {
        flex: 1;
        min-width: 0;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }

    .tab-content {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }

    .tab-content.hidden {
        display: none;
    }

    .warning-banner {
        background: var(--bg-surface);
        color: var(--accent-primary);
        padding: var(--space-2) var(--space-3);
        border: 1px solid var(--accent-primary-dim);
        border-radius: var(--radius-md);
        margin: var(--space-3);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        flex-shrink: 0;
    }

    .empty-state {
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 1;
        color: var(--text-muted);
        font-family: var(--font-sans);
    }
</style>
