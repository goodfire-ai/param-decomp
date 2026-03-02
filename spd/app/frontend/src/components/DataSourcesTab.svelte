<script lang="ts">
    import { getContext, onMount } from "svelte";
    import type { Loadable } from "../lib";
    import { fetchDataSources, type DataSourcesResponse } from "../lib/api";
    import { fetchPretrainInfoForLoadedRun, type PretrainInfoResponse } from "../lib/api/pretrainInfo";
    import { RUN_KEY, type RunContext } from "../lib/useRun.svelte";
    import TopologyDiagram from "./TopologyDiagram.svelte";

    const runState = getContext<RunContext>(RUN_KEY);

    let data = $state<Loadable<DataSourcesResponse>>({ status: "uninitialized" });
    let pretrainData = $state<Loadable<PretrainInfoResponse>>({ status: "uninitialized" });

    onMount(async () => {
        data = { status: "loading" };
        pretrainData = { status: "loading" };
        try {
            const result = await fetchDataSources();
            data = { status: "loaded", data: result };
        } catch (e) {
            data = { status: "error", error: e };
        }
        try {
            const result = await fetchPretrainInfoForLoadedRun();
            pretrainData = { status: "loaded", data: result };
        } catch (e) {
            pretrainData = { status: "error", error: e };
        }
    });

    function formatConfigValue(value: unknown): string {
        if (value === null || value === undefined) return "\u2014";
        if (typeof value === "object") return JSON.stringify(value);
        return String(value);
    }

    function formatPretrainConfigYaml(config: Record<string, unknown>): string {
        const lines: string[] = [];
        for (const [key, value] of Object.entries(config)) {
            if (typeof value === "object" && value !== null && !Array.isArray(value)) {
                lines.push(`${key}:`);
                for (const [subKey, subValue] of Object.entries(value as Record<string, unknown>)) {
                    lines.push(`  ${subKey}: ${formatConfigValue(subValue)}`);
                }
            } else {
                lines.push(`${key}: ${formatConfigValue(value)}`);
            }
        }
        return lines.join("\n");
    }
</script>

<div class="ds-page">
    <!-- Top row: Run Config + Target Model side by side -->
    <div class="top-row">
        {#if runState.run.status === "loaded" && runState.run.data.config_yaml}
            <section class="card top-card">
                <h3 class="card-title">Run Config</h3>
                <pre class="config-yaml">{runState.run.data.config_yaml}</pre>
            </section>
        {/if}

        <section class="card top-card">
            <h3 class="card-title">Target Model</h3>
            {#if pretrainData.status === "loading"}
                <p class="status-text">Loading...</p>
            {:else if pretrainData.status === "loaded"}
                {@const pt = pretrainData.data}
                <div class="info-grid">
                    <span class="label">Architecture</span>
                    <span class="value mono">{pt.summary}</span>
                    {#if pt.pretrain_wandb_path}
                        <span class="label">Pretrain run</span>
                        <span class="value mono">{pt.pretrain_wandb_path}</span>
                    {/if}
                </div>
                {#if pt.topology}
                    <div class="topology-section">
                        <TopologyDiagram topology={pt.topology} />
                    </div>
                {/if}
                {#if pt.pretrain_config}
                    <details class="config-details">
                        <summary class="config-summary">Pretraining config</summary>
                        <pre class="config-yaml">{formatPretrainConfigYaml(pt.pretrain_config)}</pre>
                    </details>
                {/if}
            {:else if pretrainData.status === "error"}
                <p class="status-text error">Failed to load target model info</p>
            {/if}
        </section>
    </div>

    <!-- Pipeline data sources grid -->
    <div class="pipeline-grid">
        <!-- Harvest -->
        <section class="card pipeline-card" class:unavailable={data.status === "loaded" && !data.data.harvest}>
            <div class="card-header">
                <span class="status-dot" class:active={data.status === "loaded" && !!data.data.harvest}></span>
                <h3 class="card-title">Harvest</h3>
            </div>
            {#if data.status === "loading"}
                <p class="status-text">Loading...</p>
            {:else if data.status === "loaded" && data.data.harvest}
                {@const harvest = data.data.harvest}
                <div class="info-grid">
                    <span class="label">Subrun</span>
                    <span class="value mono">{harvest.subrun_id}</span>
                    <span class="label">Components</span>
                    <span class="value">{harvest.n_components.toLocaleString()}</span>
                    <span class="label">Intruder eval</span>
                    <span class="value">{harvest.has_intruder_scores ? "yes" : "no"}</span>
                    {#each Object.entries(harvest.config) as [key, value] (key)}
                        <span class="label">{key}</span>
                        <span class="value mono">{formatConfigValue(value)}</span>
                    {/each}
                </div>
            {:else if data.status === "loaded"}
                <p class="status-text">Not available</p>
            {/if}
        </section>

        <!-- Autointerp -->
        <section class="card pipeline-card" class:unavailable={data.status === "loaded" && !data.data.autointerp}>
            <div class="card-header">
                <span class="status-dot" class:active={data.status === "loaded" && !!data.data.autointerp}></span>
                <h3 class="card-title">Autointerp</h3>
            </div>
            {#if data.status === "loading"}
                <p class="status-text">Loading...</p>
            {:else if data.status === "loaded" && data.data.autointerp}
                {@const autointerp = data.data.autointerp}
                <div class="info-grid">
                    <span class="label">Subrun</span>
                    <span class="value mono">{autointerp.subrun_id}</span>
                    <span class="label">Interpretations</span>
                    <span class="value">{autointerp.n_interpretations.toLocaleString()}</span>
                    <span class="label">Eval scores</span>
                    <span class="value">
                        {#if autointerp.eval_scores.length > 0}
                            {autointerp.eval_scores.join(", ")}
                        {:else}
                            <span class="muted">none</span>
                        {/if}
                    </span>
                    {#each Object.entries(autointerp.config) as [key, value] (key)}
                        <span class="label">{key}</span>
                        <span class="value mono">{formatConfigValue(value)}</span>
                    {/each}
                </div>
            {:else if data.status === "loaded"}
                <p class="status-text">Not available</p>
            {/if}
        </section>

        <!-- Dataset Attributions -->
        <section class="card pipeline-card" class:unavailable={data.status === "loaded" && !data.data.attributions}>
            <div class="card-header">
                <span class="status-dot" class:active={data.status === "loaded" && !!data.data.attributions}></span>
                <h3 class="card-title">Dataset Attributions</h3>
            </div>
            {#if data.status === "loading"}
                <p class="status-text">Loading...</p>
            {:else if data.status === "loaded" && data.data.attributions}
                {@const attributions = data.data.attributions}
                <div class="info-grid">
                    <span class="label">Subrun</span>
                    <span class="value mono">{attributions.subrun_id}</span>
                    <span class="label">Tokens</span>
                    <span class="value">{attributions.n_tokens_processed.toLocaleString()}</span>
                    <span class="label">CI threshold</span>
                    <span class="value mono">{attributions.ci_threshold}</span>
                </div>
            {:else if data.status === "loaded"}
                <p class="status-text">Not available</p>
            {/if}
        </section>

        <!-- Graph Interp -->
        <section class="card pipeline-card" class:unavailable={data.status === "loaded" && !data.data.graph_interp}>
            <div class="card-header">
                <span class="status-dot" class:active={data.status === "loaded" && !!data.data.graph_interp}></span>
                <h3 class="card-title">Graph Interp</h3>
            </div>
            {#if data.status === "loading"}
                <p class="status-text">Loading...</p>
            {:else if data.status === "loaded" && data.data.graph_interp}
                {@const graph_interp = data.data.graph_interp}
                <div class="info-grid">
                    <span class="label">Subrun</span>
                    <span class="value mono">{graph_interp.subrun_id}</span>
                    {#each Object.entries(graph_interp.label_counts) as [key, value] (key)}
                        <span class="label">{key} labels</span>
                        <span class="value">{value.toLocaleString()}</span>
                    {/each}
                    {#if graph_interp.config}
                        {#each Object.entries(graph_interp.config) as [key, value] (key)}
                            <span class="label">{key}</span>
                            <span class="value mono">{formatConfigValue(value)}</span>
                        {/each}
                    {/if}
                </div>
            {:else if data.status === "loaded"}
                <p class="status-text">Not available</p>
            {/if}
        </section>
    </div>
</div>

<style>
    .ds-page {
        padding: var(--space-6);
        display: flex;
        flex-direction: column;
        gap: var(--space-4);
        overflow-y: auto;
        max-height: 100%;
    }

    .top-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-4);
    }

    .pipeline-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-4);
    }

    @media (max-width: 900px) {
        .top-row,
        .pipeline-grid {
            grid-template-columns: 1fr;
        }
    }

    .card {
        background: var(--bg-inset);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: var(--space-4);
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
    }

    .card.unavailable {
        opacity: 0.45;
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .card-title {
        font-family: var(--font-sans);
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: 0.01em;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--border-default);
        flex-shrink: 0;
    }

    .status-dot.active {
        background: var(--status-positive);
    }

    .status-text {
        color: var(--text-muted);
        font-family: var(--font-sans);
        font-size: var(--text-xs);
        margin: 0;
    }

    .status-text.error {
        color: var(--status-negative);
    }

    .info-grid {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 3px var(--space-3);
        font-size: var(--text-xs);
    }

    .label {
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .value {
        color: var(--text-primary);
        font-family: var(--font-sans);
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .value.mono {
        font-family: var(--font-mono);
    }

    .muted {
        color: var(--text-muted);
    }

    .topology-section {
        border-top: 1px solid var(--border-subtle);
        padding-top: var(--space-3);
    }

    .config-details {
        margin-top: var(--space-1);
    }

    .config-summary {
        font-family: var(--font-sans);
        font-size: var(--text-xs);
        color: var(--text-muted);
        cursor: pointer;
        padding: var(--space-1) 0;
    }

    .config-summary:hover {
        color: var(--text-primary);
    }

    .config-yaml {
        max-height: 50vh;
        overflow: auto;
        margin: 0;
        font-size: 11px;
        font-family: var(--font-mono);
        color: var(--text-secondary);
        white-space: pre-wrap;
        word-wrap: break-word;
        line-height: 1.5;
    }
</style>
