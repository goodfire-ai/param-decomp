<script lang="ts">
    import { page } from "$app/state";
    import { fmtAct, fmtCi, fmtCount, fmtDensity, matrixKind } from "$lib/format";
    import type { ActivationExample, ComponentDetail } from "$lib/types";

    let {
        data,
    }: {
        data: {
            run: string;
            site: string;
            detail: ComponentDetail;
            examplePageSize: number;
        };
    } = $props();

    const mk = $derived(matrixKind(data.site));
    const label = $derived(data.detail.label);

    // Sidebar state (sort/page/q) survives component navigation; the example page `ep`
    // is per-component, so the component stepper and site-relative links drop it.
    const carry = $derived.by(() => {
        const p = new URLSearchParams(page.url.search);
        p.delete("ep");
        const s = p.toString();
        return s ? `?${s}` : "";
    });

    const examplePage = $derived(data.detail.example_page);
    const nExamplePages = $derived(Math.ceil(data.detail.n_examples / data.examplePageSize));

    function examplePageHref(ep: number): string {
        const p = new URLSearchParams(page.url.search);
        if (ep <= 0) p.delete("ep");
        else p.set("ep", String(ep));
        const s = p.toString();
        return s ? `?${s}` : "";
    }

    let colorBy: "ci" | "act" = $state("ci");

    const valueAt = (ex: ActivationExample, pos: number) =>
        Math.max(0, colorBy === "ci" ? ex.cis[pos] : ex.acts[pos]);

    const vMax = $derived(
        Math.max(
            0,
            ...data.detail.examples.flatMap((ex) => (colorBy === "ci" ? ex.cis : ex.acts)),
        ),
    );

    const ALPHA_MIN = 0.06;
    const ALPHA_MAX = 0.62;

    function tokenStyle(ex: ActivationExample, pos: number): string {
        const a =
            vMax <= 0
                ? ALPHA_MIN
                : ALPHA_MIN + (valueAt(ex, pos) / vMax) * (ALPHA_MAX - ALPHA_MIN);
        return `background: rgba(var(--hl), ${a.toFixed(3)});`;
    }

    function peakPos(ex: ActivationExample): number {
        let best = 0;
        for (let i = 1; i < ex.cis.length; i++) if (ex.cis[i] > ex.cis[best]) best = i;
        return best;
    }

    function tip(ex: ActivationExample, pos: number): string {
        return `act ${ex.acts[pos].toFixed(3)}\nci  ${ex.cis[pos].toFixed(3)}`;
    }
</script>

<svelte:head><title>Scope · {data.site} · {data.detail.idx}</title></svelte:head>

<div class="doc">
    <header class="identity">
        <div class="top-row">
            <span class="eyebrow">
                {#if mk}<span class="mtag {mk}">{mk}</span> {/if}component
                <span class="num">#{data.detail.idx}</span> · rank {data.detail.rank + 1}
            </span>
            <span class="stepper">
                {#if data.detail.prev_idx !== null}
                    <a class="step num" href="/r/{data.run}/s/{data.site}/c/{data.detail.prev_idx}{carry}" title="previous by mean CI">←</a>
                {/if}
                {#if data.detail.next_idx !== null}
                    <a class="step num" href="/r/{data.run}/s/{data.site}/c/{data.detail.next_idx}{carry}" title="next by mean CI">→</a>
                {/if}
            </span>
        </div>
        {#if label !== null}
            <h1 class="display">{label.text}</h1>
            <p class="subline">auto-interpreted · {label.model} · {label.created_at}</p>
        {:else}
            <h1 class="display unlabeled">Unlabeled</h1>
        {/if}
    </header>

    <div class="hairgrid stats">
        <div class="stat">
            <span class="eyebrow">mean ci</span>
            <span class="value num">{fmtCi(data.detail.mean_ci)}</span>
        </div>
        <div class="stat">
            <span class="eyebrow">density</span>
            <span class="value num">{fmtDensity(data.detail.density)}</span>
        </div>
        <div class="stat">
            <span class="eyebrow">max act</span>
            <span class="value num">{fmtAct(data.detail.max_act)}</span>
        </div>
        <div class="stat">
            <span class="eyebrow">examples</span>
            <span class="value num">{fmtCount(data.detail.n_examples)}</span>
        </div>
    </div>

    <div class="columns">
        <section>
            <h2 class="eyebrow">input tokens (pmi)</h2>
            <div class="chips">
                {#each data.detail.input_pmi as [token, score], i (i)}
                    <span class="chip"><span class="chip-tok num">{token}</span><span class="chip-score num">{score.toFixed(1)}</span></span>
                {/each}
            </div>
        </section>
        <section>
            <h2 class="eyebrow">output tokens (pmi)</h2>
            <div class="chips">
                {#each data.detail.output_pmi as [token, score], i (i)}
                    <span class="chip"><span class="chip-tok num">{token}</span><span class="chip-score num">{score.toFixed(1)}</span></span>
                {/each}
            </div>
        </section>
    </div>

    <section>
        <div class="examples-head">
            <h2 class="eyebrow">activating examples</h2>
            <span class="toggle eyebrow">
                shade by
                <button class:active={colorBy === "ci"} onclick={() => (colorBy = "ci")}>causal importance</button>
                <button class:active={colorBy === "act"} onclick={() => (colorBy = "act")}>activation</button>
            </span>
        </div>
        <ol class="examples" start={examplePage * data.examplePageSize + 1}>
            {#each data.detail.examples as ex, i (i)}
                {@const peak = peakPos(ex)}
                <li>
                    <span class="ex-max num" class:top={i === 0 && examplePage === 0}>{fmtAct(ex.max_act)}</span>
                    <span class="ex-text">
                        {#each ex.tokens as token, pos (pos)}<span
                                class="tok"
                                class:peak={pos === peak}
                                style={tokenStyle(ex, pos)}
                                data-tip={tip(ex, pos)}>{token}</span
                            >{/each}
                    </span>
                </li>
            {/each}
        </ol>
        {#if nExamplePages > 1}
            <nav class="pager eyebrow">
                {#if examplePage > 0}
                    <a class="page-step" href={examplePageHref(examplePage - 1)}>← newer</a>
                {:else}
                    <span class="page-step disabled">← newer</span>
                {/if}
                <span class="page-pos num">page {examplePage + 1} / {nExamplePages}</span>
                {#if examplePage + 1 < nExamplePages}
                    <a class="page-step" href={examplePageHref(examplePage + 1)}>weaker →</a>
                {:else}
                    <span class="page-step disabled">weaker →</span>
                {/if}
            </nav>
        {/if}
    </section>
</div>

<style>
    .doc {
        max-width: 880px;
        padding: 32px 40px 80px;
    }
    .identity {
        margin-bottom: 28px;
    }
    .top-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
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
        margin: 0 0 4px;
    }
    .unlabeled {
        color: var(--faint);
    }
    .stats {
        grid-template-columns: repeat(4, 1fr);
        margin-bottom: 28px;
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
    .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .chip {
        display: inline-flex;
        align-items: baseline;
        gap: 6px;
        padding: 2px 8px;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 4px;
        font-size: 12px;
    }
    .chip-tok {
        white-space: pre;
        font-weight: 500;
    }
    .chip-score {
        color: var(--dim);
        font-size: 11px;
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
        line-height: 1.85;
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
    .pager {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-top: 16px;
    }
    .page-step {
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 3px 10px;
        color: var(--dim);
        text-decoration: none;
        background: var(--panel);
    }
    .page-step:hover {
        color: var(--fg);
        border-color: var(--line-2);
    }
    .page-step.disabled {
        color: var(--faint);
        background: none;
        pointer-events: none;
    }
    .page-pos {
        color: var(--dim);
        margin-left: auto;
        margin-right: auto;
    }
</style>
