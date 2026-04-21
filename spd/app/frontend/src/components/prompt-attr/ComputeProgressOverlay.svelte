<script lang="ts">
    import type { CISnapshot } from "../../lib/promptAttributionsTypes";
    import type { LoadingState } from "./types";
    import { bgBaseRgb } from "../../lib/colors";
    import OptimizationGrid from "./OptimizationGrid.svelte";

    type Props = {
        state: LoadingState;
        ciSnapshot?: CISnapshot | null;
    };

    let { state, ciSnapshot = null }: Props = $props();

    const overlayBg = `rgba(${bgBaseRgb.r}, ${bgBaseRgb.g}, ${bgBaseRgb.b}, 0.95)`;
</script>

<div class="loading-overlay" style="background: {overlayBg};">
    <div class="content">
        {#if ciSnapshot}
            <OptimizationGrid snapshot={ciSnapshot} />
        {/if}
        <div class="stages">
            {#each state.stages as stage, i (i)}
                {@const isCurrent = i === state.currentStage}
                {@const isComplete = i < state.currentStage}
                <div class="stage" class:current={isCurrent} class:complete={isComplete}>
                    <div class="stage-header">
                        <span class="stage-number">{i + 1}</span>
                        <span class="stage-name">{stage.name}</span>
                        {#if isComplete}
                            <span class="stage-check">✓</span>
                        {/if}
                    </div>
                    {#if isCurrent}
                        <div class="progress-bar">
                            {#if stage.progress !== null}
                                <div class="progress-fill" style="width: {stage.progress * 100}%"></div>
                            {:else}
                                <div class="progress-fill indeterminate"></div>
                            {/if}
                        </div>
                    {:else if isComplete}
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 100%"></div>
                        </div>
                    {:else}
                        <div class="progress-bar empty"></div>
                    {/if}
                </div>
            {/each}
        </div>
    </div>
</div>

<style>
    .loading-overlay {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 100;
    }

    .content {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: var(--space-6);
    }

    .stages {
        display: flex;
        flex-direction: column;
        gap: var(--space-4);
        width: 280px;
    }

    .stage {
        opacity: 0.4;
    }

    .stage.current,
    .stage.complete {
        opacity: 1;
    }

    .stage-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        margin-bottom: var(--space-1);
    }

    .stage-number {
        width: 20px;
        height: 20px;
        background: var(--border-default);
        color: var(--text-muted);
        font-size: var(--text-xs);
        font-weight: 600;
        font-family: var(--font-mono);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .stage.current .stage-number {
        background: var(--accent-primary);
        color: var(--bg-base);
    }

    .stage.complete .stage-number {
        background: var(--status-positive);
        color: var(--text-primary);
    }

    .stage-name {
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-muted);
        font-weight: 500;
        letter-spacing: 0.05em;
    }

    .stage.current .stage-name {
        color: var(--text-primary);
    }

    .stage-check {
        color: var(--status-positive-bright);
        font-size: var(--text-sm);
        margin-left: auto;
    }

    .progress-bar {
        height: 4px;
        background: var(--border-default);
        overflow: hidden;
        margin-left: 28px;
    }

    .progress-bar.empty {
        background: var(--border-subtle);
    }

    .progress-fill {
        height: 100%;
        background: var(--accent-primary);
        transition: width var(--transition-normal);
    }

    .stage.complete .progress-fill {
        background: var(--status-positive);
    }

    .progress-fill.indeterminate {
        width: 30%;
        animation: indeterminate 1.2s ease-in-out infinite;
    }

    @keyframes indeterminate {
        0% {
            transform: translateX(-100%);
        }
        50% {
            transform: translateX(233%);
        }
        100% {
            transform: translateX(-100%);
        }
    }
</style>
