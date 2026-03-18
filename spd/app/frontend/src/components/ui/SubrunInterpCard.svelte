<script lang="ts">
    import type { Loadable } from "../../lib";
    import type { CompareInterpretationDetail, CompareInterpretationHeadline } from "../../lib/api";

    interface Props {
        subrunId: string;
        strategy: string;
        llmModel: string;
        headline: CompareInterpretationHeadline | null;
        detail: Loadable<CompareInterpretationDetail | null>;
    }

    let { subrunId, strategy, llmModel, headline, detail }: Props = $props();

    let showPrompt = $state(false);

    function scoreClass(score: number): string {
        if (score >= 0.7) return "score-high";
        if (score >= 0.5) return "score-medium";
        return "score-low";
    }

    const shortModel = $derived(llmModel.split("/").pop() ?? llmModel);
</script>

<div class="subrun-card">
    <div class="card-header">
        <span class="strategy-tag">{strategy}</span>
        <span class="model-tag">{shortModel}</span>
        <span class="subrun-id">{subrunId}</span>
    </div>

    {#if headline === null}
        <div class="no-interp">No interpretation for this component</div>
    {:else}
        <div class="card-body">
            <div class="label-row">
                <span class="label">{headline.label}</span>
                {#if headline.detection_score !== null}
                    <span class="score-pill {scoreClass(headline.detection_score)}"
                        >Det {Math.round(headline.detection_score * 100)}%</span
                    >
                {/if}
                {#if headline.fuzzing_score !== null}
                    <span class="score-pill {scoreClass(headline.fuzzing_score)}"
                        >Fuz {Math.round(headline.fuzzing_score * 100)}%</span
                    >
                {/if}
            </div>

            {#if detail.status === "loaded" && detail.data}
                <div class="reasoning">{detail.data.reasoning}</div>
                <button class="prompt-toggle" onclick={() => (showPrompt = !showPrompt)}>
                    {showPrompt ? "Hide" : "Show"} Prompt
                </button>
                {#if showPrompt}
                    <div class="prompt-display">
                        <pre>{detail.data.prompt}</pre>
                    </div>
                {/if}
            {:else if detail.status === "loading"}
                <div class="reasoning loading-text">Loading...</div>
            {:else if detail.status === "error"}
                <div class="reasoning error-text">Error loading detail</div>
            {/if}
        </div>
    {/if}
</div>

<style>
    .subrun-card {
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        background: var(--bg-surface);
        overflow: hidden;
    }

    .card-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border-bottom: 1px solid var(--border-default);
    }

    .strategy-tag {
        font-size: var(--text-xs);
        font-weight: 600;
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
        background: color-mix(in srgb, var(--accent-primary) 20%, transparent);
        color: var(--accent-primary);
    }

    .model-tag {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .subrun-id {
        margin-left: auto;
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .card-body {
        padding: var(--space-3);
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .no-interp {
        padding: var(--space-3);
        font-size: var(--text-sm);
        color: var(--text-muted);
        font-style: italic;
    }

    .label-row {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        flex-wrap: wrap;
    }

    .label {
        font-weight: 500;
        font-size: var(--text-sm);
        color: var(--text-primary);
    }

    .score-pill {
        font-size: var(--text-xs);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
        font-weight: 600;
        white-space: nowrap;
    }

    .score-high {
        background: color-mix(in srgb, var(--status-positive-bright) 20%, transparent);
        color: var(--status-positive-bright);
    }

    .score-medium {
        background: color-mix(in srgb, var(--status-warning) 20%, transparent);
        color: var(--status-warning);
    }

    .score-low {
        background: color-mix(in srgb, var(--text-muted) 20%, transparent);
        color: var(--text-muted);
    }

    .reasoning {
        font-size: var(--text-xs);
        color: var(--text-secondary);
        line-height: 1.4;
    }

    .loading-text {
        color: var(--text-muted);
        font-style: italic;
    }

    .error-text {
        color: var(--status-negative);
    }

    .prompt-toggle {
        align-self: flex-start;
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-xs);
        background: var(--bg-elevated);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        cursor: pointer;
        font-weight: 500;
    }

    .prompt-toggle:hover {
        background: var(--bg-surface);
        border-color: var(--border-strong);
    }

    .prompt-display {
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        padding: var(--space-3);
        max-height: 400px;
        overflow: auto;
    }

    .prompt-display pre {
        margin: 0;
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        white-space: pre-wrap;
        word-break: break-word;
        color: var(--text-secondary);
    }
</style>
