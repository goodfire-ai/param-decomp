<script lang="ts">
    import type { SubrunSummary } from "../lib/api";
    import { SvelteSet } from "svelte/reactivity";

    interface Props {
        subruns: SubrunSummary[];
        selectedIds: SvelteSet<string>;
    }

    let { subruns, selectedIds }: Props = $props();
    let collapsed = $state(false);

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

    const nSelected = $derived(selectedIds.size);
</script>

<div class="selector">
    <button class="selector-header" onclick={() => (collapsed = !collapsed)}>
        <span class="selector-label">
            Subruns ({subruns.length})
            {#if nSelected > 0}
                <span class="selected-count">&middot; {nSelected} selected</span>
            {/if}
        </span>
        <span class="collapse-icon">{collapsed ? "▸" : "▾"}</span>
    </button>

    {#if !collapsed}
        <table class="subrun-table">
            <thead>
                <tr>
                    <th class="col-select"></th>
                    <th class="col-note">Note</th>
                    <th class="col-strategy">Strategy</th>
                    <th class="col-model">Model</th>
                    <th class="col-count">Interps</th>
                    <th class="col-score">Det</th>
                    <th class="col-score">Fuz</th>
                    <th class="col-harvest">Harvest</th>
                    <th class="col-time">Time</th>
                </tr>
            </thead>
            <tbody>
                {#each subruns as subrun (subrun.subrun_id)}
                    <tr
                        class:selected={selectedIds.has(subrun.subrun_id)}
                        onclick={() => toggle(subrun.subrun_id)}
                    >
                        <td class="col-select">
                            <span class="checkbox" class:checked={selectedIds.has(subrun.subrun_id)}
                            ></span>
                        </td>
                        <td class="col-note">
                            {#if subrun.note}
                                <span class="note-text">{subrun.note}</span>
                            {:else}
                                <span class="empty">-</span>
                            {/if}
                        </td>
                        <td class="col-strategy">{subrun.strategy}</td>
                        <td class="col-model">{shortModel(subrun.llm_model)}</td>
                        <td class="col-count">{subrun.n_completed}</td>
                        <td class="col-score">{formatScore(subrun.mean_detection_score)}</td>
                        <td class="col-score">{formatScore(subrun.mean_fuzzing_score)}</td>
                        <td class="col-harvest">
                            {#if subrun.harvest_mismatch}
                                <span class="harvest-warn" title={subrun.harvest_subrun_id ?? ""}>
                                    ⚠ {subrun.harvest_subrun_id}
                                </span>
                            {:else if subrun.harvest_subrun_id}
                                <span class="harvest-ok">{subrun.harvest_subrun_id}</span>
                            {:else}
                                <span class="empty">?</span>
                            {/if}
                        </td>
                        <td class="col-time">{subrun.timestamp}</td>
                    </tr>
                {/each}
            </tbody>
        </table>
    {/if}
</div>

<style>
    .selector {
        display: flex;
        flex-direction: column;
    }

    .selector-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: var(--space-2) var(--space-1);
        background: none;
        border: none;
        cursor: pointer;
        color: var(--text-secondary);
    }

    .selector-header:hover {
        color: var(--text-primary);
    }

    .selector-label {
        font-size: var(--text-sm);
        font-weight: 500;
        font-family: var(--font-sans);
    }

    .selected-count {
        color: var(--accent-primary);
    }

    .collapse-icon {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .subrun-table {
        width: 100%;
        border-collapse: collapse;
        font-size: var(--text-xs);
    }

    thead th {
        text-align: left;
        padding: var(--space-1) var(--space-2);
        font-weight: 600;
        color: var(--text-muted);
        border-bottom: 1px solid var(--border-default);
        white-space: nowrap;
        font-family: var(--font-sans);
    }

    tbody tr {
        cursor: pointer;
        transition: background var(--transition-normal);
    }

    tbody tr:hover {
        background: var(--bg-elevated);
    }

    tbody tr.selected {
        background: color-mix(in srgb, var(--accent-primary) 8%, var(--bg-surface));
    }

    tbody td {
        padding: var(--space-1) var(--space-2);
        border-bottom: 1px solid var(--border-subtle, var(--border-default));
        vertical-align: top;
    }

    .col-select {
        width: 28px;
        text-align: center;
    }

    .checkbox {
        display: inline-block;
        width: 14px;
        height: 14px;
        border: 1.5px solid var(--border-strong);
        border-radius: 3px;
        vertical-align: middle;
    }

    .checkbox.checked {
        background: var(--accent-primary);
        border-color: var(--accent-primary);
    }

    .col-note {
        max-width: 300px;
    }

    .note-text {
        color: var(--accent-primary);
        font-weight: 500;
    }

    .col-strategy {
        font-weight: 600;
        color: var(--text-primary);
    }

    .col-model {
        font-family: var(--font-mono);
        color: var(--text-secondary);
    }

    .col-count {
        text-align: right;
        color: var(--text-secondary);
    }

    .col-score {
        text-align: right;
        color: var(--text-secondary);
        white-space: nowrap;
    }

    .col-time {
        font-family: var(--font-mono);
        color: var(--text-muted);
        white-space: nowrap;
    }

    .col-harvest {
        font-family: var(--font-mono);
        white-space: nowrap;
    }

    .harvest-warn {
        color: var(--status-warning);
        font-weight: 600;
    }

    .harvest-ok {
        color: var(--text-muted);
    }

    .empty {
        color: var(--text-muted);
    }
</style>
