<script lang="ts">
    import DatasetSearchPanel from "./DatasetSearchPanel.svelte";
    import DatasetRandomPanel from "./DatasetRandomPanel.svelte";

    type InnerTab = "search" | "random";
    let activeInnerTab = $state<InnerTab>("search");
</script>

<div class="explorer-container">
    <div class="inner-tabs">
        <button
            type="button"
            class="inner-tab"
            class:active={activeInnerTab === "search"}
            onclick={() => (activeInnerTab = "search")}
        >
            Search
        </button>
        <button
            type="button"
            class="inner-tab"
            class:active={activeInnerTab === "random"}
            onclick={() => (activeInnerTab = "random")}
        >
            Random Samples
        </button>
    </div>

    <div class="tab-panels">
        <div class="tab-panel" class:hidden={activeInnerTab !== "search"}>
            <DatasetSearchPanel />
        </div>
        <div class="tab-panel" class:hidden={activeInnerTab !== "random"}>
            <DatasetRandomPanel />
        </div>
    </div>
</div>

<style>
    .explorer-container {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        padding: var(--space-6);
        gap: var(--space-4);
    }

    .inner-tabs {
        display: flex;
        gap: var(--space-1);
        border-bottom: 1px solid var(--border-default);
        padding-bottom: var(--space-2);
    }

    .inner-tab {
        padding: var(--space-2) var(--space-4);
        background: none;
        border: none;
        border-bottom: 2px solid transparent;
        font: inherit;
        font-weight: 500;
        font-size: var(--text-sm);
        color: var(--text-muted);
        cursor: pointer;
        transition:
            color var(--transition-normal),
            border-color var(--transition-normal);
    }

    .inner-tab:hover {
        color: var(--text-primary);
    }

    .inner-tab.active {
        color: var(--text-primary);
        border-bottom-color: var(--accent-primary);
    }

    .tab-panels {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }

    .tab-panel {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }

    .tab-panel.hidden {
        display: none;
    }
</style>
