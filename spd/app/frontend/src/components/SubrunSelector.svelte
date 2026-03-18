<script lang="ts">
    import type { SubrunSummary } from "../lib/api";
    import { SvelteSet } from "svelte/reactivity";

    interface Props {
        subruns: SubrunSummary[];
        selectedIds: SvelteSet<string>;
    }

    let { subruns, selectedIds }: Props = $props();

    function toggle(id: string) {
        if (selectedIds.has(id)) {
            selectedIds.delete(id);
        } else {
            selectedIds.add(id);
        }
    }

    function formatScore(score: number | null): string {
        if (score === null) return "-";
        return `${Math.round(score * 100)}%`;
    }

    function shortModel(model: string): string {
        return model.split("/").pop() ?? model;
    }
</script>

<div class="selector">
    <div class="selector-label">Subruns</div>
    <div class="chips">
        {#each subruns as subrun (subrun.subrun_id)}
            <button
                type="button"
                class="chip"
                class:selected={selectedIds.has(subrun.subrun_id)}
                onclick={() => toggle(subrun.subrun_id)}
            >
                <span class="chip-strategy">{subrun.strategy}</span>
                <span class="chip-model">{shortModel(subrun.llm_model)}</span>
                <span class="chip-meta">
                    {subrun.n_completed} interps
                    {#if subrun.mean_detection_score !== null}
                        &middot; Det {formatScore(subrun.mean_detection_score)}
                    {/if}
                    {#if subrun.mean_fuzzing_score !== null}
                        &middot; Fuz {formatScore(subrun.mean_fuzzing_score)}
                    {/if}
                </span>
                <span class="chip-time">{subrun.timestamp}</span>
            </button>
        {/each}
    </div>
</div>

<style>
    .selector {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .selector-label {
        font-size: var(--text-sm);
        font-weight: 500;
        font-family: var(--font-sans);
        color: var(--text-secondary);
    }

    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-2);
    }

    .chip {
        display: flex;
        flex-direction: column;
        gap: 2px;
        padding: var(--space-2) var(--space-3);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        background: var(--bg-surface);
        cursor: pointer;
        text-align: left;
        transition:
            border-color var(--transition-normal),
            background var(--transition-normal);
    }

    .chip:hover {
        border-color: var(--border-strong);
        background: var(--bg-elevated);
    }

    .chip.selected {
        border-color: var(--accent-primary);
        background: color-mix(in srgb, var(--accent-primary) 10%, var(--bg-surface));
    }

    .chip-strategy {
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
    }

    .chip-model {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .chip-meta {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .chip-time {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }
</style>
