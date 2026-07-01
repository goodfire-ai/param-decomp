<script lang="ts">
    import { fmtAct, fmtCount, fmtDensity } from "$lib/format";
    import SiteCurve from "$lib/ui/SiteCurve.svelte";
    import type { ActivationExample, ComponentLabel } from "$lib/types";

    let { data } = $props();

    // svelte-ignore state_referenced_locally — resynced from data on navigation below
    let label: ComponentLabel | null = $state(data.detail.label);
    $effect(() => {
        label = data.detail.label;
    });

    let labeling = $state(false);
    let colorBy: "act" | "ci" = $state("act");

    async function requestLabel() {
        labeling = true;
        const res = await fetch(
            `/api/runs/${data.run}/sites/${data.site}/components/${data.detail.idx}/label`,
            { method: "POST" },
        );
        if (!res.ok) throw new Error(`label request failed: ${res.status}`);
        label = await res.json();
        labeling = false;
    }


    const valueAt = (ex: ActivationExample, pos: number) =>
        Math.max(0, colorBy === "act" ? ex.acts[pos] : ex.cis[pos]);

    const vMax = $derived(
        Math.max(0, ...data.detail.examples.flatMap((ex) => (colorBy === "act" ? ex.acts : ex.cis))),
    );

    const ALPHA_MIN = 0.06;
    const ALPHA_MAX = 0.55;

    function tokenAlpha(ex: ActivationExample, pos: number): number {
        if (vMax <= 0) return ALPHA_MIN;
        const a = ALPHA_MIN + (valueAt(ex, pos) / vMax) * (ALPHA_MAX - ALPHA_MIN);
        return Math.min(Math.max(a, ALPHA_MIN), ALPHA_MAX);
    }

    function tokenStyle(ex: ActivationExample, pos: number): string {
        return `background: rgba(var(--hl), ${tokenAlpha(ex, pos).toFixed(3)});`;
    }

    const isHot = (ex: ActivationExample, pos: number) =>
        vMax > 0 && valueAt(ex, pos) >= 0.9 * vMax;

    function tip(ex: ActivationExample, pos: number): string {
        return `act ${ex.acts[pos].toFixed(3)}\nci  ${ex.cis[pos].toFixed(3)}`;
    }
</script>

<svelte:head><title>Scope · {data.site} · {data.detail.idx}</title></svelte:head>

<div class="narrow">
    <header class="identity">
        <div class="eyebrow-row">
            <span class="eyebrow">
                component {data.detail.idx} ·
                <a href="/r/{data.run}/s/{data.site}">{data.site}</a>
                · rank {data.detail.rank + 1} of {fmtCount(data.curve.total)}
            </span>
            <span class="stepper">
                {#if data.detail.prev_idx !== null}
                    <a
                        class="step num"
                        href="/r/{data.run}/s/{data.site}/c/{data.detail.prev_idx}"
                        title="previous by rank">←</a>
                {/if}
                {#if data.detail.next_idx !== null}
                    <a
                        class="step num"
                        href="/r/{data.run}/s/{data.site}/c/{data.detail.next_idx}"
                        title="next by rank">→</a>
                {/if}
            </span>
        </div>
        {#if label !== null}
            <h1 class="display">{label.text}</h1>
            <div class="subline-row">
                <span class="subline">
                    auto-interpreted label · {label.model} · {label.created_at}
                </span>
                <button class="quiet" disabled={labeling} onclick={requestLabel}>
                    {labeling ? "labelling…" : "relabel · $0.03"}
                </button>
            </div>
        {:else}
            <div class="unlabeled-row">
                <h1 class="display unlabeled">Unlabeled</h1>
                <button class="quiet" disabled={labeling} onclick={requestLabel}>
                    {labeling ? "labelling…" : "label · $0.03"}
                </button>
            </div>
        {/if}
    </header>

    <SiteCurve curve={data.curve} currentRank={data.detail.rank} run={data.run} site={data.site} />

    <div class="hairgrid stats">
        <div class="stat">
            <span class="eyebrow">density</span>
            <span class="value num">{fmtDensity(data.detail.density)}</span>
        </div>
        <div class="stat">
            <span class="eyebrow">max act</span>
            <span class="value num">{fmtAct(data.detail.max_act)}</span>
        </div>
        <div class="stat">
            <span class="eyebrow">mean ci</span>
            <span class="value num">{fmtDensity(data.detail.mean_ci)}</span>
        </div>
        <div class="stat">
            <span class="eyebrow">examples</span>
            <span class="value num">{data.detail.examples.length}</span>
        </div>
    </div>

    <div class="columns">
        <section>
            <h2 class="eyebrow">input token affinity (pmi)</h2>
            <ol class="pmi">
                {#each data.detail.input_pmi as [token, score], i (i)}
                    <li>
                        <span class="pmi-tok num">{token}</span>
                        <span class="pmi-score num">{score.toFixed(2)}</span>
                        <span class="pmi-bar" style="width: {Math.min(score / 10, 1) * 100}%"
                        ></span>
                    </li>
                {/each}
            </ol>
        </section>
        <section>
            <h2 class="eyebrow">output token affinity (pmi)</h2>
            <ol class="pmi">
                {#each data.detail.output_pmi as [token, score], i (i)}
                    <li>
                        <span class="pmi-tok num">{token}</span>
                        <span class="pmi-score num">{score.toFixed(2)}</span>
                        <span class="pmi-bar" style="width: {Math.min(score / 10, 1) * 100}%"
                        ></span>
                    </li>
                {/each}
            </ol>
        </section>
    </div>

    <section>
        <div class="examples-head">
            <h2 class="eyebrow">activation examples</h2>
            <span class="toggle eyebrow">
                colour by
                <button class:active={colorBy === "act"} onclick={() => (colorBy = "act")}>
                    activation
                </button>
                <button class:active={colorBy === "ci"} onclick={() => (colorBy = "ci")}>
                    causal importance
                </button>
            </span>
        </div>
        <ol class="examples">
            {#each data.detail.examples as ex, i (i)}
                <li>
                    <span class="ex-max num" class:top={i === 0}>{fmtAct(ex.max_act)}</span>
                    <span class="ex-text">
                        {#each ex.tokens as token, pos (pos)}<span
                                class="tok"
                                class:hot={isHot(ex, pos)}
                                style={tokenStyle(ex, pos)}
                                data-tip={tip(ex, pos)}>{token}</span
                            >{/each}
                    </span>
                </li>
            {/each}
        </ol>
        <div class="legend">
            <div class="ramp panel">
                <span class="swatch s1"></span>
                <span class="swatch s2"></span>
                <span class="swatch s3"></span>
                <span class="swatch s4"></span>
                <span class="swatch s5"></span>
            </div>
            <div class="legend-labels">
                <span class="eyebrow">0.0 — inactive</span>
                <span class="eyebrow">
                    {colorBy === "act" ? "activation strength" : "causal importance"}
                </span>
                <span class="eyebrow">{fmtAct(vMax)} — max</span>
            </div>
        </div>
    </section>
</div>

<style>
    .identity {
        margin-bottom: 32px;
    }
    .eyebrow-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }
    .eyebrow-row a {
        color: inherit;
    }
    .eyebrow-row a:hover {
        color: var(--fg);
    }
    .stepper {
        display: inline-flex;
        gap: 4px;
        margin-left: auto;
    }
    .step {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        border: 1px solid var(--line);
        border-radius: 4px;
        font-size: 13px;
        color: var(--dim);
        text-decoration: none;
        background: var(--panel);
    }
    .step:hover {
        color: var(--fg);
        border-color: var(--line-2);
    }
    h1.display {
        margin: 0;
    }
    .unlabeled {
        color: var(--faint);
    }
    .unlabeled-row {
        display: flex;
        align-items: baseline;
        gap: 16px;
    }
    .subline-row {
        display: flex;
        align-items: baseline;
        gap: 16px;
        margin-top: 8px;
    }
    .subline-row button {
        margin-left: auto;
        flex-shrink: 0;
    }
    .stats {
        grid-template-columns: repeat(4, 1fr);
        margin-bottom: 32px;
    }
    .stat {
        display: flex;
        flex-direction: column;
        gap: 2px;
        padding: 12px 16px;
    }
    .value {
        font-size: 18px;
        font-weight: 500;
    }
    .columns {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 32px;
        margin-bottom: 32px;
    }
    h2 {
        margin: 0 0 12px;
    }
    .pmi {
        list-style: none;
        margin: 0;
        padding: 0;
        font-size: 13px;
        column-count: 2;
        column-gap: 24px;
    }
    .pmi li {
        display: grid;
        grid-template-columns: 1fr 48px;
        grid-template-rows: auto 3px;
        row-gap: 2px;
        padding: 2px 0;
        break-inside: avoid;
    }
    .pmi-tok {
        white-space: pre;
        font-weight: 500;
    }
    .pmi-score {
        text-align: right;
        color: var(--dim);
    }
    .pmi-bar {
        grid-column: 1 / 3;
        height: 2px;
        background: var(--accent);
        opacity: 0.45;
    }
    .examples-head {
        display: flex;
        align-items: baseline;
        margin-bottom: 12px;
    }
    .examples-head h2 {
        margin: 0;
    }
    .toggle {
        margin-left: auto;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .toggle button {
        font-family: var(--mono);
        font-size: 10.5px;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        background: none;
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 2px 8px;
        color: var(--dim);
        cursor: pointer;
    }
    .toggle button:hover {
        border-color: var(--line-2);
    }
    .toggle button.active {
        color: var(--fg);
        border-color: var(--line-2);
    }
    .examples {
        list-style: none;
        margin: 0;
        padding: 0;
    }
    .examples li {
        display: flex;
        gap: 12px;
        align-items: baseline;
        padding: 5px 0;
        border-bottom: 1px solid var(--line);
        font-size: 13.5px;
        line-height: 1.75;
    }
    .ex-max {
        flex-shrink: 0;
        min-width: 48px;
        text-align: right;
        font-size: 11px;
        color: var(--dim);
    }
    .ex-max.top {
        color: var(--accent);
    }
    .ex-text {
        white-space: nowrap;
        overflow-x: auto;
        scrollbar-width: none;
    }
    .legend {
        margin-top: 16px;
        max-width: 360px;
    }
    .ramp {
        display: flex;
        overflow: hidden;
    }
    .swatch {
        flex: 1;
        height: 12px;
    }
    .swatch.s1 {
        background: rgba(var(--hl), 0.06);
    }
    .swatch.s2 {
        background: rgba(var(--hl), 0.16);
    }
    .swatch.s3 {
        background: rgba(var(--hl), 0.28);
    }
    .swatch.s4 {
        background: rgba(var(--hl), 0.4);
    }
    .swatch.s5 {
        background: rgba(var(--hl), 0.52);
    }
    .legend-labels {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-top: 4px;
    }
</style>
