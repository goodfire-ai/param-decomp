<script lang="ts">
    import {
        type NodeColorMode,
        type EdgeVariant,
        displaySettings,
        CORRELATION_STAT_LABELS,
        CORRELATION_STAT_DESCRIPTIONS,
        NODE_COLOR_MODE_LABELS,
        EDGE_VARIANT_LABELS,
    } from "../../lib/displaySettings.svelte";
    import GearIcon from "./icons/GearIcon.svelte";

    let showDropdown = $state(false);

    const colorModes: NodeColorMode[] = ["ci", "subcomp_act"];
    const edgeVariants: EdgeVariant[] = ["signed", "abs_target"];
</script>

<div
    class="settings-wrapper"
    role="group"
    onmouseenter={() => (showDropdown = true)}
    onmouseleave={() => (showDropdown = false)}
>
    <button type="button" class="settings-button" title="Display Settings">
        <GearIcon />
    </button>
    {#if showDropdown}
        <div class="settings-dropdown">
            <div class="settings-section">
                <h4>Correlation Stats</h4>
                <p class="settings-hint">Select which correlation metrics to display</p>
                <div class="checkbox-list">
                    <label class="checkbox-item">
                        <input
                            type="checkbox"
                            checked={displaySettings.showPmi}
                            onchange={() => (displaySettings.showPmi = !displaySettings.showPmi)}
                        />
                        <span class="stat-label">{CORRELATION_STAT_LABELS["pmi"]}</span>
                        <span class="stat-desc">{CORRELATION_STAT_DESCRIPTIONS["pmi"]}</span>
                    </label>
                    <label class="checkbox-item">
                        <input
                            type="checkbox"
                            checked={displaySettings.showPrecision}
                            onchange={() => (displaySettings.showPrecision = !displaySettings.showPrecision)}
                        />
                        <span class="stat-label">{CORRELATION_STAT_LABELS["precision"]}</span>
                        <span class="stat-desc">{CORRELATION_STAT_DESCRIPTIONS["precision"]}</span>
                    </label>
                    <label class="checkbox-item">
                        <input
                            type="checkbox"
                            checked={displaySettings.showRecall}
                            onchange={() => (displaySettings.showRecall = !displaySettings.showRecall)}
                        />
                        <span class="stat-label">{CORRELATION_STAT_LABELS["recall"]}</span>
                        <span class="stat-desc">{CORRELATION_STAT_DESCRIPTIONS["recall"]}</span>
                    </label>
                    <label class="checkbox-item">
                        <input
                            type="checkbox"
                            checked={displaySettings.showJaccard}
                            onchange={() => (displaySettings.showJaccard = !displaySettings.showJaccard)}
                        />
                        <span class="stat-label">{CORRELATION_STAT_LABELS["jaccard"]}</span>
                        <span class="stat-desc">{CORRELATION_STAT_DESCRIPTIONS["jaccard"]}</span>
                    </label>
                </div>
            </div>
            <div class="settings-section">
                <h4>Visualizations</h4>
                <div class="checkbox-list">
                    <label class="checkbox-item single-row">
                        <input
                            type="checkbox"
                            checked={displaySettings.showSetOverlapVis}
                            onchange={() => (displaySettings.showSetOverlapVis = !displaySettings.showSetOverlapVis)}
                        />
                        <span class="stat-label">Set overlap bars</span>
                    </label>
                    <label class="checkbox-item single-row">
                        <input
                            type="checkbox"
                            checked={displaySettings.showEdgeAttributions}
                            onchange={() =>
                                (displaySettings.showEdgeAttributions = !displaySettings.showEdgeAttributions)}
                        />
                        <span class="stat-label">Edge attributions</span>
                    </label>
                    <label class="checkbox-item single-row">
                        <input
                            type="checkbox"
                            checked={displaySettings.centerOnPeak}
                            onchange={() => (displaySettings.centerOnPeak = !displaySettings.centerOnPeak)}
                        />
                        <span class="stat-label">Center on peak</span>
                    </label>
                    <label class="checkbox-item single-row">
                        <input
                            type="checkbox"
                            checked={displaySettings.curvedEdges}
                            onchange={() => (displaySettings.curvedEdges = !displaySettings.curvedEdges)}
                        />
                        <span class="stat-label">Curved edges</span>
                    </label>
                    <label class="checkbox-item single-row">
                        <input
                            type="checkbox"
                            checked={displaySettings.showAutoInterpPromptButton}
                            onchange={() =>
                                (displaySettings.showAutoInterpPromptButton =
                                    !displaySettings.showAutoInterpPromptButton)}
                        />
                        <span class="stat-label">Autointerp prompt button</span>
                    </label>
                </div>
            </div>
            <div class="settings-section">
                <h4>Node Color</h4>
                <p class="settings-hint">Color intensity based on</p>
                <div class="radio-list">
                    {#each colorModes as mode (mode)}
                        <label class="radio-item">
                            <input
                                type="radio"
                                name="node-color-mode"
                                checked={displaySettings.nodeColorMode === mode}
                                onchange={() => (displaySettings.nodeColorMode = mode)}
                            />
                            <span class="stat-label">{NODE_COLOR_MODE_LABELS[mode]}</span>
                        </label>
                    {/each}
                </div>
            </div>
            <div class="settings-section">
                <h4>Edge Variant</h4>
                <p class="settings-hint">Attribution target: value or |value|</p>
                <div class="radio-list">
                    {#each edgeVariants as variant (variant)}
                        <label class="radio-item">
                            <input
                                type="radio"
                                name="edge-variant"
                                checked={displaySettings.edgeVariant === variant}
                                onchange={() => (displaySettings.edgeVariant = variant)}
                            />
                            <span class="stat-label">{EDGE_VARIANT_LABELS[variant]}</span>
                        </label>
                    {/each}
                </div>
            </div>
            <div class="settings-section">
                <h4>Component Filtering</h4>
                <p class="settings-hint">Filter components in Components tab by mean CI</p>
                <div class="cutoff-control">
                    <label for="mean-ci-cutoff">Mean CI Cutoff:</label>
                    <input
                        id="mean-ci-cutoff"
                        type="number"
                        step="any"
                        value={displaySettings.meanCiCutoff}
                        oninput={(e) => {
                            const val = parseFloat((e.target as HTMLInputElement).value);
                            if (!isNaN(val) && val >= 0) {
                                displaySettings.meanCiCutoff = val;
                            }
                        }}
                    />
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .settings-wrapper {
        position: relative;
        display: flex;
        align-items: center;
        padding: 0 var(--space-3);
        border-left: 1px solid var(--border-default);
    }

    .settings-button {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: var(--space-1);
        background: transparent;
        border: none;
        border-radius: var(--radius-sm);
        cursor: pointer;
        color: var(--text-muted);
        transition:
            color 0.15s,
            background 0.15s;
    }

    .settings-button:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
    }

    .settings-dropdown {
        position: absolute;
        top: 100%;
        right: 0;
        padding-top: var(--space-2);
        z-index: 1000;
        min-width: 280px;
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .settings-section {
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
        padding: var(--space-3);
    }

    .settings-section h4 {
        margin: 0 0 var(--space-1) 0;
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
    }

    .settings-hint {
        margin: 0 0 var(--space-2) 0;
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .checkbox-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .checkbox-item {
        display: grid;
        grid-template-columns: auto 1fr;
        grid-template-rows: auto auto;
        gap: 0 var(--space-2);
        cursor: pointer;
        padding: var(--space-1);
        border-radius: var(--radius-sm);
    }

    .checkbox-item:hover {
        background: var(--bg-inset);
    }

    .checkbox-item input {
        grid-row: span 2;
        margin: 0;
        cursor: pointer;
        accent-color: var(--accent-primary);
    }

    .checkbox-item.single-row {
        grid-template-rows: auto;
    }

    .checkbox-item.single-row input {
        grid-row: span 1;
    }

    .stat-label {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-primary);
    }

    .stat-desc {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-mono);
    }

    .radio-list {
        display: flex;
        gap: var(--space-3);
    }

    .radio-item {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        cursor: pointer;
        padding: var(--space-1);
        border-radius: var(--radius-sm);
    }

    .radio-item:hover {
        background: var(--bg-inset);
    }

    .radio-item input {
        margin: 0;
        cursor: pointer;
        accent-color: var(--accent-primary);
    }

    .cutoff-control {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .cutoff-control label {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-primary);
    }

    .cutoff-control input[type="number"] {
        width: 100px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        background: var(--bg-surface);
        color: var(--text-primary);
    }

    .cutoff-control input[type="number"]:focus {
        outline: none;
        border-color: var(--accent-primary);
    }
</style>
