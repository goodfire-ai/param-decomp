<script lang="ts">
    import type { OptimizeConfigDraft, MaskType, LossType, LossConfigDraft } from "./types";
    import TokenDropdown from "./TokenDropdown.svelte";
    import TokenSpan from "../ui/TokenSpan.svelte";
    import { getNextTokenProbBgColor } from "../../lib/colors";
    import { getProbAtPosition, formatProb } from "../../lib/tokenUtils";

    type Props = {
        config: OptimizeConfigDraft;
        tokens: string[];
        nextTokenProbs: (number | null)[];
        onChange: (newConfig: OptimizeConfigDraft) => void;
        cardId: number;
    };

    let { config, tokens, nextTokenProbs, onChange, cardId }: Props = $props();
    let showAdvanced = $state(false);

    // Slider value 0-100 controls impMinCoeff on log scale from 1e-5 to 10
    const sliderValue = $derived.by(() => {
        const value = (100 * Math.log10(config.impMinCoeff / 1e-5)) / 6;
        return Math.round(Math.max(0, Math.min(100, value)));
    });

    function handleSliderChange(value: number) {
        const impMinCoeff = 1e-5 * Math.pow(1e6, value / 100);
        const rounded = parseFloat(impMinCoeff.toPrecision(2));
        onChange({ ...config, impMinCoeff: rounded });
    }

    function handleLossTypeChange(newType: LossType) {
        const position = config.loss.position;
        const coeff = config.loss.coeff;
        let newLoss: LossConfigDraft;
        if (newType === "kl") {
            newLoss = { type: "kl", coeff, position };
        } else if (newType === "logit") {
            newLoss = { type: "logit", coeff, position, labelTokenId: null, labelTokenText: "" };
        } else {
            newLoss = { type: "ce", coeff, position, labelTokenId: null, labelTokenText: "" };
        }
        onChange({ ...config, loss: newLoss });
    }

    function handlePositionClick(pos: number) {
        onChange({ ...config, loss: { ...config.loss, position: pos } });
    }
</script>

<div class="opt-settings">
    <!-- Loss type selection -->
    <div class="loss-type-options">
        <label class="loss-type-option" class:selected={config.loss.type === "kl"}>
            <input
                type="radio"
                name="loss-type-{cardId}"
                checked={config.loss.type === "kl"}
                onchange={() => handleLossTypeChange("kl")}
            />
            <span class="option-name">KL Divergence</span>
        </label>
        <label class="loss-type-option" class:selected={config.loss.type === "ce"}>
            <input
                type="radio"
                name="loss-type-{cardId}"
                checked={config.loss.type === "ce"}
                onchange={() => handleLossTypeChange("ce")}
            />
            <span class="option-name">Cross-Entropy</span>
        </label>
        <label class="loss-type-option" class:selected={config.loss.type === "logit"}>
            <input
                type="radio"
                name="loss-type-{cardId}"
                checked={config.loss.type === "logit"}
                onchange={() => handleLossTypeChange("logit")}
            />
            <span class="option-name">Logit</span>
        </label>
    </div>

    <!-- Token position selector strip -->
    <div class="position-section">
        <span class="section-label">Position</span>
        <div class="token-strip">
            {#each tokens as tok, i (i)}
                {@const prob = getProbAtPosition(nextTokenProbs, i)}
                <button
                    type="button"
                    class="strip-token"
                    class:selected={config.loss.position === i}
                    onclick={() => handlePositionClick(i)}
                    title="pos {i}{prob !== null ? ` | P: ${formatProb(prob)}` : ''}"
                    ><TokenSpan token={tok} backgroundColor={getNextTokenProbBgColor(prob)} /></button
                >
            {/each}
        </div>
        <div class="position-info">
            <span class="pos-label">pos {config.loss.position}</span>
            {#if config.loss.type === "ce" || config.loss.type === "logit"}
                <span class="predict-label">{config.loss.type === "logit" ? "maximize" : "predict"}</span>
                <TokenDropdown
                    value={config.loss.labelTokenText}
                    selectedTokenId={config.loss.labelTokenId}
                    promptId={cardId}
                    position={config.loss.position}
                    onSelect={(tokenId, tokenString) => {
                        if (config.loss.type !== "ce" && config.loss.type !== "logit")
                            throw new Error("inconsistent state: Token dropdown rendered but loss type has no label");

                        if (tokenId !== null) {
                            onChange({
                                ...config,
                                loss: { ...config.loss, labelTokenId: tokenId, labelTokenText: tokenString },
                            });
                        }
                    }}
                    placeholder="token..."
                />
            {/if}
        </div>
    </div>

    <!-- Sparsity slider -->
    <div class="slider-section">
        <div class="slider-header">
            <span class="section-label">Sparsity</span>
            <input
                type="text"
                class="imp-min-input"
                value={config.impMinCoeff.toPrecision(2)}
                onchange={(e) => {
                    const val = parseFloat(e.currentTarget.value);
                    if (!isNaN(val) && val > 0) {
                        onChange({ ...config, impMinCoeff: val });
                    }
                }}
            />
        </div>
        <input
            type="range"
            class="sparsity-slider"
            min={0}
            max={100}
            value={sliderValue}
            oninput={(e) => handleSliderChange(parseInt(e.currentTarget.value))}
        />
        <div class="slider-labels">
            <span class="slider-label">1e-5</span>
            <span class="slider-label">10</span>
        </div>
    </div>

    <!-- Advanced toggle -->
    <button class="advanced-toggle" onclick={() => (showAdvanced = !showAdvanced)}>
        {showAdvanced ? "▼" : "▶"} Advanced
    </button>

    {#if showAdvanced}
        <div class="advanced-section">
            <div class="settings-grid">
                <label>
                    <span class="label-text">steps</span>
                    <input
                        type="number"
                        value={config.steps}
                        oninput={(e) => {
                            if (e.currentTarget.value === "") return;
                            onChange({ ...config, steps: parseInt(e.currentTarget.value) });
                        }}
                        min={10}
                        max={5000}
                        step={100}
                    />
                </label>
                <label>
                    <span class="label-text">pnorm</span>
                    <input
                        type="number"
                        value={config.pnorm}
                        oninput={(e) => {
                            if (e.currentTarget.value === "") return;
                            onChange({ ...config, pnorm: parseFloat(e.currentTarget.value) });
                        }}
                        min={0.1}
                        max={2}
                        step={0.1}
                    />
                </label>
                <label>
                    <span class="label-text">beta</span>
                    <input
                        type="number"
                        value={config.beta}
                        oninput={(e) => {
                            if (e.currentTarget.value === "") return;
                            onChange({ ...config, beta: parseFloat(e.currentTarget.value) });
                        }}
                        min={0}
                        max={10}
                        step={0.1}
                    />
                </label>
                <label>
                    <span class="label-text">mask_type</span>
                    <select
                        value={config.maskType}
                        onchange={(e) => onChange({ ...config, maskType: e.currentTarget.value as MaskType })}
                    >
                        <option value="stochastic">stochastic</option>
                        <option value="ci">ci</option>
                    </select>
                </label>
                <label>
                    <span class="label-text">loss_coeff</span>
                    <input
                        type="number"
                        value={config.loss.coeff}
                        oninput={(e) => {
                            if (e.currentTarget.value === "") return;
                            const coeff = parseFloat(e.currentTarget.value);
                            onChange({ ...config, loss: { ...config.loss, coeff } });
                        }}
                        min={0}
                        step={0.1}
                    />
                </label>
                <label>
                    <span class="label-text">adv_n_steps</span>
                    <input
                        type="number"
                        value={config.advPgdNSteps ?? ""}
                        oninput={(e) => {
                            const val = e.currentTarget.value;
                            onChange({ ...config, advPgdNSteps: val === "" ? null : parseInt(val) });
                        }}
                        min={1}
                        max={50}
                        step={1}
                        placeholder="off"
                    />
                </label>
                <label>
                    <span class="label-text">adv_step_size</span>
                    <input
                        type="number"
                        value={config.advPgdStepSize ?? ""}
                        oninput={(e) => {
                            const val = e.currentTarget.value;
                            onChange({ ...config, advPgdStepSize: val === "" ? null : parseFloat(val) });
                        }}
                        min={0.001}
                        max={1}
                        step={0.01}
                        placeholder="off"
                    />
                </label>
            </div>
        </div>
    {/if}
</div>

<style>
    .opt-settings {
        display: flex;
        flex-direction: column;
        gap: var(--space-3);
        max-width: 500px;
    }

    .loss-type-options {
        display: flex;
        gap: var(--space-2);
    }

    .loss-type-option {
        flex: 1;
        display: flex;
        align-items: center;
        gap: var(--space-2);
        padding: var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        cursor: pointer;
        transition: border-color var(--transition-normal);
    }

    .loss-type-option:hover {
        border-color: var(--border-strong);
    }

    .loss-type-option.selected {
        border-color: var(--accent-primary);
        background: var(--bg-surface);
    }

    .loss-type-option input {
        cursor: pointer;
        accent-color: var(--accent-primary);
    }

    .option-name {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-primary);
    }

    .position-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .section-label {
        font-size: var(--text-xs);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
    }

    .token-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 2px;
        padding: var(--space-2);
        background: var(--bg-inset);
        border: 1px solid var(--border-default);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
    }

    .strip-token {
        padding: 2px 2px;
        border: 1px solid var(--border-subtle);
        border-radius: 2px;
        cursor: pointer;
        white-space: pre;
        font-family: inherit;
        font-size: inherit;
        color: var(--text-primary);
        background: transparent;
        position: relative;
        transition:
            border-color var(--transition-fast),
            box-shadow var(--transition-fast);
    }

    .strip-token:hover {
        border-color: var(--border-strong);
    }

    .strip-token.selected {
        border-color: var(--accent-primary);
        box-shadow: 0 0 0 1px var(--accent-primary);
        z-index: 1;
    }

    .strip-token::after {
        content: attr(title);
        position: absolute;
        bottom: calc(100% + 4px);
        left: 50%;
        transform: translateX(-50%);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        color: var(--text-primary);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-xs);
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        z-index: 100;
        border-radius: var(--radius-sm);
    }

    .strip-token:hover::after {
        opacity: 1;
    }

    .position-info {
        display: flex;
        align-items: center;
        gap: var(--space-2);
    }

    .pos-label {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        background: var(--bg-inset);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
    }

    .predict-label {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .slider-section {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .slider-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .imp-min-input {
        width: 80px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
    }

    .imp-min-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .slider-labels {
        display: flex;
        justify-content: space-between;
    }

    .slider-label {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .sparsity-slider {
        width: 100%;
        height: 20px;
        appearance: none;
        background: transparent;
        cursor: pointer;
    }

    .sparsity-slider::-webkit-slider-runnable-track {
        width: 100%;
        height: 6px;
        background: var(--border-default);
        border-radius: var(--radius-sm);
    }

    .sparsity-slider::-webkit-slider-thumb {
        appearance: none;
        width: 16px;
        height: 16px;
        background: var(--accent-primary);
        border-radius: 50%;
        cursor: pointer;
        margin-top: -5px;
    }

    .sparsity-slider::-moz-range-track {
        width: 100%;
        height: 6px;
        background: var(--border-default);
        border-radius: var(--radius-sm);
    }

    .sparsity-slider::-moz-range-thumb {
        width: 16px;
        height: 16px;
        background: var(--accent-primary);
        border-radius: 50%;
        cursor: pointer;
        border: none;
    }

    .advanced-toggle {
        background: none;
        border: none;
        padding: var(--space-1) 0;
        font-size: var(--text-xs);
        color: var(--text-muted);
        cursor: pointer;
        text-align: left;
    }

    .advanced-toggle:hover {
        color: var(--text-secondary);
    }

    .advanced-section {
        padding: var(--space-2);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
    }

    .settings-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
        gap: var(--space-2);
    }

    .settings-grid label {
        display: flex;
        flex-direction: column;
        gap: var(--space-1);
    }

    .label-text {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
    }

    .settings-grid input[type="number"],
    .settings-grid select {
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .settings-grid input[type="number"]:focus,
    .settings-grid select:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .settings-grid select {
        cursor: pointer;
    }
</style>
