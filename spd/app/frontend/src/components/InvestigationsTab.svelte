<script lang="ts">
    import * as api from "../lib/api";
    import type {
        InvestigationSummary,
        InvestigationDetail,
        GraphArtifact,
        LaunchResponse,
    } from "../lib/api/investigations";
    import type { Loadable } from "../lib";
    import ResearchLogViewer from "./investigations/ResearchLogViewer.svelte";

    // State
    let investigations = $state<Loadable<InvestigationSummary[]>>({ status: "uninitialized" });
    let selected = $state<Loadable<InvestigationDetail> | null>(null);
    let activeTab = $state<"research" | "events">("research");
    let loadedArtifacts = $state<Record<string, GraphArtifact>>({});
    let artifactsLoading = $state(false);

    // Launch state
    let launchPrompt = $state("");
    let launchState = $state<Loadable<LaunchResponse>>({ status: "uninitialized" });

    async function handleLaunch() {
        if (!launchPrompt.trim()) return;
        launchState = { status: "loading" };
        try {
            const result = await api.launchInvestigation(launchPrompt.trim());
            launchState = { status: "loaded", data: result };
            launchPrompt = "";
            await loadInvestigations();
            selectInvestigation(result.inv_id);
        } catch (e) {
            launchState = { status: "error", error: e };
        }
    }

    // Load investigations on mount
    $effect(() => {
        loadInvestigations();
    });

    async function loadInvestigations() {
        investigations = { status: "loading" };
        try {
            const data = await api.listInvestigations();
            investigations = { status: "loaded", data };
        } catch (e) {
            investigations = { status: "error", error: String(e) };
        }
    }

    let selectedInvId = $state<string | null>(null);

    async function selectInvestigation(invId: string) {
        selectedInvId = invId;
        selected = { status: "loading" };
        loadedArtifacts = {};
        artifactsLoading = false;
        await fetchInvestigation(invId);
    }

    async function fetchInvestigation(invId: string) {
        try {
            const data = await api.getInvestigation(invId);
            selected = { status: "loaded", data };

            // Load all artifacts for this investigation
            if (data.artifact_ids.length > 0) {
                artifactsLoading = true;
                const artifacts: Record<string, GraphArtifact> = {};
                await Promise.all(
                    data.artifact_ids.map(async (artifactId) => {
                        try {
                            const artifact = await api.getArtifact(invId, artifactId);
                            artifacts[artifactId] = artifact;
                        } catch (e) {
                            console.error(`Failed to load artifact ${artifactId}:`, e);
                        }
                    }),
                );
                loadedArtifacts = artifacts;
                artifactsLoading = false;
            }
        } catch (e) {
            selected = { status: "error", error: String(e) };
            artifactsLoading = false;
        }
    }

    function refreshSelected() {
        if (selectedInvId) fetchInvestigation(selectedInvId);
    }

    function goBack() {
        selected = null;
        selectedInvId = null;
        loadedArtifacts = {};
        artifactsLoading = false;
    }

    function formatDate(isoString: string): string {
        const date = new Date(isoString);
        return date.toLocaleString();
    }

    function formatId(id: string): string {
        return id.replace("inv-", "");
    }

    function getEventTypeColor(eventType: string): string {
        switch (eventType) {
            case "tool_start":
            case "tool_call":
                return "var(--accent-blue)";
            case "tool_complete":
                return "var(--accent-green)";
            case "optimization_progress":
                return "var(--accent-yellow)";
            case "error":
                return "var(--accent-red)";
            default:
                return "var(--text-muted)";
        }
    }
</script>

<div class="investigations-container">
    {#if selected?.status === "loaded"}
        <!-- Investigation Detail View -->
        <div class="detail-header">
            <button class="back-button" onclick={goBack}>← Back</button>
            <h2>{selected.data.title || formatId(selected.data.id)}</h2>
            <button class="refresh-button" onclick={refreshSelected}>↻ Refresh</button>
            {#if selected.data.status}
                <span
                    class="status-pill"
                    class:completed={selected.data.status === "completed"}
                    class:in-progress={selected.data.status === "in_progress"}
                >
                    {selected.data.status}
                </span>
            {/if}
        </div>

        {#if selected.data.summary}
            <p class="investigation-summary-detail">{selected.data.summary}</p>
        {/if}

        <!-- TODO: Add ability to open corresponding graph for this investigation -->
        <p class="investigation-info">
            {formatId(selected.data.id)} · Started {formatDate(selected.data.created_at)}
            {#if selected.data.wandb_path}
                · <span class="wandb-path">{selected.data.wandb_path}</span>
            {/if}
        </p>

        <div class="tabs">
            <button class="tab" class:active={activeTab === "research"} onclick={() => (activeTab = "research")}>
                Research Log
            </button>
            <button class="tab" class:active={activeTab === "events"} onclick={() => (activeTab = "events")}>
                Events ({selected.data.events.length})
            </button>
        </div>

        <div class="tab-content">
            {#if activeTab === "research"}
                {#if selected.data.research_log}
                    <ResearchLogViewer
                        markdown={selected.data.research_log}
                        artifacts={loadedArtifacts}
                        {artifactsLoading}
                    />
                {:else}
                    <p class="empty-message">No research log available</p>
                {/if}
            {:else}
                <div class="events-list">
                    {#each selected.data.events as event, i (i)}
                        <div class="event-entry">
                            <span class="event-type" style="color: {getEventTypeColor(event.event_type)}">
                                {event.event_type}
                            </span>
                            <span class="event-time">{formatDate(event.timestamp)}</span>
                            <span class="event-message">{event.message}</span>
                            {#if event.details && Object.keys(event.details).length > 0}
                                <details class="event-details">
                                    <summary>Details</summary>
                                    <pre>{JSON.stringify(event.details, null, 2)}</pre>
                                </details>
                            {/if}
                        </div>
                    {:else}
                        <p class="empty-message">No events recorded</p>
                    {/each}
                </div>
            {/if}
        </div>
    {:else if selected?.status === "loading"}
        <div class="loading">Loading investigation...</div>
    {:else}
        <!-- Investigations List View -->
        <div class="list-header">
            <h2>Investigations</h2>
            <button class="refresh-button" onclick={loadInvestigations}>↻ Refresh</button>
        </div>

        <form
            class="launch-form"
            onsubmit={(e) => {
                e.preventDefault();
                handleLaunch();
            }}
        >
            <input
                class="launch-input"
                type="text"
                placeholder="Ask a research question..."
                bind:value={launchPrompt}
                disabled={launchState.status === "loading"}
            />
            <button
                class="launch-button"
                type="submit"
                disabled={launchState.status === "loading" || !launchPrompt.trim()}
            >
                {launchState.status === "loading" ? "Launching..." : "Investigate"}
            </button>
        </form>
        {#if launchState.status === "error"}
            <div class="launch-error">{launchState.error}</div>
        {/if}

        {#if investigations.status === "loading"}
            <div class="loading">Loading investigations...</div>
        {:else if investigations.status === "error"}
            <div class="error">{investigations.error}</div>
        {:else if investigations.status === "loaded"}
            <div class="investigations-list">
                {#each investigations.data as inv (inv.id)}
                    <button class="investigation-card" onclick={() => selectInvestigation(inv.id)}>
                        <div class="card-header">
                            <span class="investigation-id">{formatId(inv.id)}</span>
                            <span class="investigation-date">{formatDate(inv.created_at)}</span>
                        </div>
                        {#if inv.title}
                            <span class="investigation-title">{inv.title}</span>
                        {/if}
                        {#if inv.prompt}
                            <span class="investigation-prompt">{inv.prompt}</span>
                        {/if}
                        {#if inv.wandb_path}
                            <span class="investigation-wandb">{inv.wandb_path}</span>
                        {/if}
                        <div class="card-status">
                            {#if inv.status}
                                <span
                                    class="status-badge"
                                    class:success={inv.status === "completed"}
                                    class:warning={inv.status === "in_progress"}
                                >
                                    {inv.status === "completed" ? "✓" : inv.status === "in_progress" ? "⏳" : "?"}
                                    {inv.status}
                                </span>
                            {/if}
                            {#if inv.has_research_log}
                                <span class="status-badge">📝 Log</span>
                            {/if}
                            {#if inv.event_count > 0}
                                <span class="status-badge">{inv.event_count} events</span>
                            {/if}
                            {#if inv.has_explanations}
                                <span class="status-badge success">✓ Explanations</span>
                            {/if}
                        </div>
                        {#if inv.summary}
                            <span class="investigation-summary">{inv.summary}</span>
                        {:else if inv.last_event_message}
                            <span class="last-event">{inv.last_event_message}</span>
                        {/if}
                    </button>
                {:else}
                    <p class="empty-message">
                        No investigations found. Run <code>spd-investigate</code> to create one.
                    </p>
                {/each}
            </div>
        {/if}
    {/if}
</div>

<style>
    .investigations-container {
        padding: var(--space-4);
        height: 100%;
        overflow-y: auto;
    }

    .list-header,
    .detail-header {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        margin-bottom: var(--space-4);
    }

    .list-header h2,
    .detail-header h2 {
        margin: 0;
        font-size: var(--text-xl);
        color: var(--text-primary);
    }

    .back-button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        color: var(--text-secondary);
        cursor: pointer;
        font-size: var(--text-sm);
    }

    .back-button:hover {
        background: var(--bg-elevated);
        color: var(--text-primary);
    }

    .refresh-button {
        padding: var(--space-1) var(--space-2);
        background: none;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        color: var(--text-secondary);
        cursor: pointer;
        font-size: var(--text-sm);
    }

    .refresh-button:hover {
        background: var(--bg-surface);
    }

    .launch-form {
        display: flex;
        gap: var(--space-2);
        margin-bottom: var(--space-4);
    }

    .launch-input {
        flex: 1;
        padding: var(--space-2) var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        color: var(--text-primary);
        font-size: var(--text-sm);
    }

    .launch-input:focus {
        outline: none;
        border-color: var(--accent-primary);
    }

    .launch-input:disabled {
        opacity: 0.5;
    }

    .launch-button {
        padding: var(--space-2) var(--space-3);
        background: var(--accent-primary);
        border: none;
        border-radius: var(--radius-md);
        color: var(--bg-base);
        font-size: var(--text-sm);
        font-weight: 600;
        cursor: pointer;
        white-space: nowrap;
    }

    .launch-button:hover:not(:disabled) {
        opacity: 0.9;
    }

    .launch-button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .launch-error {
        color: var(--accent-red);
        font-size: var(--text-sm);
        margin-bottom: var(--space-3);
    }

    .wandb-path,
    .investigation-wandb {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--accent-primary);
    }

    .investigation-info {
        color: var(--text-muted);
        font-size: var(--text-sm);
        margin-bottom: var(--space-4);
    }

    .investigations-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .investigation-card {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: var(--space-1);
        padding: var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        cursor: pointer;
        text-align: left;
        width: 100%;
        transition:
            border-color var(--transition-normal),
            background var(--transition-normal);
    }

    .investigation-card:hover {
        border-color: var(--accent-primary);
        background: var(--bg-elevated);
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
    }

    .investigation-id {
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
        font-family: var(--font-mono);
    }

    .investigation-date {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .card-status {
        display: flex;
        gap: var(--space-1);
        flex-wrap: wrap;
    }

    .status-badge {
        padding: var(--space-0) var(--space-1);
        background: var(--bg-inset);
        border-radius: var(--radius-sm);
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .status-badge.success {
        background: var(--accent-green-dim);
        color: var(--accent-green);
    }

    .status-badge.warning {
        background: var(--accent-yellow-dim);
        color: var(--accent-yellow);
    }

    .investigation-prompt {
        font-size: var(--text-xs);
        color: var(--text-secondary);
        font-style: italic;
        line-height: 1.4;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-box-orient: vertical;
    }

    .investigation-title {
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
    }

    .investigation-summary {
        font-size: var(--text-xs);
        color: var(--text-secondary);
        line-height: 1.4;
    }

    .investigation-summary-detail {
        font-size: var(--text-sm);
        color: var(--text-secondary);
        margin-bottom: var(--space-3);
        line-height: 1.5;
    }

    .status-pill {
        padding: var(--space-0) var(--space-2);
        border-radius: var(--radius-full);
        font-size: var(--text-xs);
        font-weight: 500;
        background: var(--bg-inset);
        color: var(--text-muted);
    }

    .status-pill.completed {
        background: var(--accent-green-dim);
        color: var(--accent-green);
    }

    .status-pill.in-progress {
        background: var(--accent-yellow-dim);
        color: var(--accent-yellow);
    }

    .last-event {
        font-size: var(--text-xs);
        color: var(--text-muted);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 100%;
    }

    .tabs {
        display: flex;
        gap: var(--space-1);
        margin-bottom: var(--space-3);
        border-bottom: 1px solid var(--border-default);
    }

    .tab {
        padding: var(--space-2) var(--space-3);
        background: none;
        border: none;
        border-bottom: 2px solid transparent;
        color: var(--text-muted);
        cursor: pointer;
        font-size: var(--text-sm);
        transition: color var(--transition-normal);
    }

    .tab:hover {
        color: var(--text-primary);
    }

    .tab.active {
        color: var(--text-primary);
        border-bottom-color: var(--accent-primary);
    }

    .tab-content {
        flex: 1;
        overflow-y: auto;
    }

    .events-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
        max-height: 70vh;
        overflow-y: auto;
    }

    .event-entry {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
        padding: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
    }

    .event-type {
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        font-weight: 600;
    }

    .event-time {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .event-message {
        font-size: var(--text-sm);
        color: var(--text-primary);
    }

    .event-details {
        margin-top: var(--space-1);
    }

    .event-details summary {
        font-size: var(--text-xs);
        color: var(--text-muted);
        cursor: pointer;
    }

    .event-details pre {
        margin: var(--space-1) 0 0 0;
        padding: var(--space-2);
        background: var(--bg-inset);
        border-radius: var(--radius-sm);
        font-size: var(--text-xs);
        overflow-x: auto;
    }

    .loading {
        color: var(--text-muted);
        padding: var(--space-4);
        text-align: center;
    }

    .error {
        color: var(--accent-red);
        padding: var(--space-4);
        text-align: center;
    }

    .empty-message {
        color: var(--text-muted);
        font-size: var(--text-sm);
    }

    .empty-message code {
        background: var(--bg-inset);
        padding: var(--space-0) var(--space-1);
        border-radius: var(--radius-sm);
        font-family: var(--font-mono);
    }
</style>
