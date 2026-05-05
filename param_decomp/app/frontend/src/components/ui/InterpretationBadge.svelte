<script lang="ts">
    import type { Loadable } from "../../lib";
    import type { InterpretationDetail } from "../../lib/api";
    import type { InterpretationBackendState } from "../../lib/useRun.svelte";
    import { displaySettings } from "../../lib/displaySettings.svelte";

    interface Props {
        interpretation: Loadable<InterpretationBackendState>;
        interpretationDetail: Loadable<InterpretationDetail | null>;
        onGenerate: () => void;
    }

    let { interpretation, interpretationDetail, onGenerate }: Props = $props();

    let showPrompt = $state(false);

    function scoreClass(score: number): string {
        if (score >= 0.7) return "score-high";
        if (score >= 0.5) return "score-medium";
        return "score-low";
    }
</script>

<div class="interpretation-container">
    <div
        class="interpretation-badge"
        class:loading={(interpretation.status === "loaded" && interpretation.data.status === "generating") ||
            interpretation.status === "loading"}
    >
        {#if interpretation.status === "loading"}
            <span class="interpretation-label loading-text">Loading interpretations...</span>
        {:else if interpretation.status === "loaded"}
            {@const interpretationData = interpretation.data}
            {#if interpretationData.status === "none"}
                <button class="generate-btn" onclick={onGenerate}>Generate Interpretation</button>
            {:else if interpretationData.status === "generating"}
                <span class="interpretation-label loading-text">Generating interpretation...</span>
            {:else if interpretationData.status === "generated"}
                <div class="interpretation-content">
                    <div class="interpretation-header">
                        <span class="interpretation-label">{interpretationData.data.label}</span>
                        {#if interpretationData.data.detection_score !== null}
                            <span class="score-pill {scoreClass(interpretationData.data.detection_score)}"
                                >Det {Math.round(interpretationData.data.detection_score * 100)}%</span
                            >
                        {/if}
                        {#if interpretationData.data.fuzzing_score !== null}
                            <span class="score-pill {scoreClass(interpretationData.data.fuzzing_score)}"
                                >Fuz {Math.round(interpretationData.data.fuzzing_score * 100)}%</span
                            >
                        {/if}
                    </div>
                    {#if interpretationDetail.status === "loaded" && interpretationDetail.data?.reasoning}
                        <span class="interpretation-reasoning">{interpretationDetail.data.reasoning}</span>
                    {:else if interpretationDetail.status === "loading"}
                        <span class="interpretation-reasoning loading-text">Loading reasoning...</span>
                        <br /><br /><br />
                        <!-- breaks are a hacky way to reduce the observed height change once the
                        reasoning loads -->
                    {/if}
                </div>
                {#if displaySettings.showAutoInterpPromptButton}
                    <button class="prompt-toggle" onclick={() => (showPrompt = !showPrompt)}>
                        {showPrompt ? "Hide" : "View"} Autointerp Prompt
                    </button>
                {/if}
                <!-- Error state for generating -->
            {:else if interpretationData.status === "generation-error"}
                <span class="interpretation-label error-text">{String(interpretationData.error)}</span>
                <button class="retry-btn" onclick={onGenerate}>Retry</button>
            {/if}
            <!-- Error state for fetching -->
        {:else if interpretation.status === "error"}
            <span class="interpretation-label error-text">{String(interpretation.error)}</span>
        {/if}
    </div>

    {#if displaySettings.showAutoInterpPromptButton && showPrompt}
        <div class="prompt-display">
            {#if interpretationDetail.status === "loading"}
                <span class="loading-text">Loading prompt...</span>
            {:else if interpretationDetail.status === "error"}
                <span class="error-text">Error loading prompt: {String(interpretationDetail.error)}</span>
            {:else if interpretationDetail.status === "loaded" && interpretationDetail.data}
                <pre>{interpretationDetail.data.prompt}</pre>
            {:else}
                <span class="loading-text">Loading prompt...</span>
            {/if}
        </div>
    {/if}
</div>

<style>
    .interpretation-container {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .interpretation-badge {
        display: flex;
        align-items: flex-start;
        gap: var(--space-2);
        padding: var(--space-2) var(--space-3);
        background: var(--bg-inset);
        border-radius: var(--radius-md);
        border-left: 3px solid var(--accent-primary);
    }

    .interpretation-content {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
        flex: 1;
        min-width: 0;
    }

    .interpretation-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .interpretation-reasoning {
        font-size: var(--text-xs);
        color: var(--text-secondary);
        line-height: 1.4;
    }

    .interpretation-badge.loading {
        opacity: 0.7;
        border-left-color: var(--text-muted);
    }

    .interpretation-label {
        font-weight: 500;
        color: var(--text-primary);
        font-size: var(--text-sm);
    }

    .interpretation-label.loading-text {
        color: var(--text-muted);
        font-style: italic;
    }

    .interpretation-label.error-text {
        color: var(--status-negative);
    }

    .interpretation-label.muted {
        color: var(--text-muted);
        font-style: italic;
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

    .generate-btn,
    .retry-btn {
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-xs);
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: var(--radius-sm);
        cursor: pointer;
        font-weight: 500;
    }

    .generate-btn:hover,
    .retry-btn:hover {
        background: var(--accent-primary-dim);
    }

    .retry-btn {
        background: var(--bg-elevated);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
    }

    .retry-btn:hover {
        background: var(--bg-surface);
        border-color: var(--border-strong);
    }

    .prompt-toggle {
        margin-left: auto;
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
        background: var(--bg-surface);
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
