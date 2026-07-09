<script lang="ts">
    /**
     * Circuit Builder: assemble rank-1 LoRAs from PD subcomponents and compare
     * base vs edited model. Read vector = normalized V column of a subcomponent;
     * write vector = weighted sum of downstream subcomponents' j-vectors
     * (computed on the fly, averaged over n_prompts).
     */
    import {
        computeJVectors,
        deleteLora,
        getComponentDetail,
        getDownstream,
        getSites,
        getSubcomponents,
        listLoras,
        loadCircuitBuilder,
        putLora,
        runCompare,
        searchLabels,
        type CompareResult,
        type ComponentDetail,
        type JVectorInfo,
        type LoraSpec,
        type SearchHit,
        type SiteInfo,
        type SubcomponentInfo,
        type TokenLogit,
    } from "../lib/api/circuitBuilder";

    let status = $state<"unloaded" | "loading" | "ready">("unloaded");
    let source = $state<"mock" | "run">("mock");
    let runRef = $state("p-55ea3f9b");
    let loadedRunId = $state<string | null>(null);
    let error = $state<string | null>(null);
    let sites = $state<SiteInfo[]>([]);
    let loras = $state<LoraSpec[]>([]);

    // --- editor state ---
    let editing = $state<LoraSpec | null>(null);
    let readSubcomps = $state<SubcomponentInfo[]>([]);
    let readPage = $state(0);
    let downstream = $state<string[]>([]);
    let writeSite = $state<string>("");
    let writeSubcomps = $state<SubcomponentInfo[]>([]);
    let writePage = $state(0);
    let jInfo = $state<JVectorInfo[]>([]);
    let busy = $state(false);
    const PAGE = 25;

    // --- label search + component detail ---
    let readQuery = $state("");
    let readHits = $state<SearchHit[]>([]);
    let writeQuery = $state("");
    let writeHits = $state<SearchHit[]>([]);
    let detail = $state<ComponentDetail | null>(null);

    async function runSearch(which: "read" | "write") {
        await guard(async () => {
            if (which === "read") {
                readHits = readQuery.trim() ? await searchLabels(readQuery, 30) : [];
            } else {
                writeHits = writeQuery.trim()
                    ? await searchLabels(writeQuery, 30, editing?.read_site ?? null)
                    : [];
            }
        });
    }

    async function showDetail(site: string, idx: number) {
        await guard(async () => {
            detail = await getComponentDetail(site, idx);
        });
    }

    async function pickReadFromSearch(hit: SearchHit) {
        if (!editing) return;
        editing.read_site = hit.site;
        editing.read_idx = hit.idx;
        readHits = [];
        readQuery = "";
        await refreshReadSite();
        editing.read_idx = hit.idx;
        void showDetail(hit.site, hit.idx);
    }

    function pickWriteFromSearch(hit: SearchHit) {
        addWrite({ site: hit.site, idx: hit.idx } as SubcomponentInfo);
        void showDetail(hit.site, hit.idx);
    }

    // --- compare state ---
    let prompt = $state("The quick brown fox");
    let topK = $state(8);
    let maxNewTokens = $state(32);
    let temperature = $state(0.8);
    let compareResult = $state<CompareResult | null>(null);
    let expandedPosition = $state<number | null>(null);
    let tooltip = $state<{
        x: number;
        y: number;
        title: string;
        base: TokenLogit[] | null;
        edited: TokenLogit[] | null;
    } | null>(null);

    function showTip(e: MouseEvent, title: string, base: TokenLogit[] | null, edited: TokenLogit[] | null) {
        tooltip = { x: e.clientX + 14, y: e.clientY + 14, title, base, edited };
    }
    function hideTip() {
        tooltip = null;
    }

    async function guard(fn: () => Promise<void>) {
        error = null;
        busy = true;
        try {
            await fn();
        } catch (e) {
            error = e instanceof Error ? e.message : String(e);
        } finally {
            busy = false;
        }
    }

    async function load() {
        status = "loading";
        await guard(async () => {
            const res = await loadCircuitBuilder(source, source === "run" ? runRef : null);
            loadedRunId = res.run_id;
            sites = await getSites();
            loras = await listLoras();
            status = "ready";
        });
        if (error) status = "unloaded";
    }

    function newLora() {
        editing = {
            name: `circuit-${loras.length + 1}`,
            read_site: sites[0]?.site ?? "",
            read_idx: 0,
            writes: [],
            scale: 1.0,
            n_prompts: 16,
            enabled: true,
        };
        jInfo = [];
        void refreshReadSite();
    }

    async function refreshReadSite() {
        if (!editing) return;
        await guard(async () => {
            readPage = 0;
            readSubcomps = await getSubcomponents(editing!.read_site, 0, PAGE, 2);
            downstream = await getDownstream(editing!.read_site);
            writeSite = downstream[0] ?? "";
            if (writeSite) {
                writePage = 0;
                writeSubcomps = await getSubcomponents(writeSite, 0, PAGE, 2);
            } else {
                writeSubcomps = [];
            }
        });
    }

    async function pageSubcomps(which: "read" | "write", delta: number) {
        if (!editing) return;
        await guard(async () => {
            if (which === "read") {
                readPage = Math.max(0, readPage + delta);
                readSubcomps = await getSubcomponents(editing!.read_site, readPage * PAGE, PAGE, 2);
            } else {
                writePage = Math.max(0, writePage + delta);
                writeSubcomps = await getSubcomponents(writeSite, writePage * PAGE, PAGE, 2);
            }
        });
    }

    async function changeWriteSite(site: string) {
        writeSite = site;
        writePage = 0;
        await guard(async () => {
            writeSubcomps = await getSubcomponents(site, 0, PAGE, 2);
        });
    }

    function addWrite(sc: SubcomponentInfo) {
        if (!editing) return;
        if (editing.writes.some((w) => w.site === sc.site && w.idx === sc.idx)) return;
        editing.writes = [...editing.writes, { site: sc.site, idx: sc.idx, weight: null }];
    }

    function removeWrite(i: number) {
        if (!editing) return;
        editing.writes = editing.writes.filter((_, k) => k !== i);
    }

    async function computeJs() {
        if (!editing || editing.writes.length === 0) return;
        await guard(async () => {
            jInfo = await computeJVectors(
                editing!.read_site,
                editing!.writes.map((w) => ({ site: w.site, idx: w.idx })),
                editing!.n_prompts,
            );
        });
    }

    async function saveLora() {
        if (!editing) return;
        await guard(async () => {
            await putLora(editing!);
            loras = await listLoras();
            editing = null;
        });
    }

    async function removeLora(name: string) {
        await guard(async () => {
            await deleteLora(name);
            loras = await listLoras();
        });
    }

    async function toggleLora(spec: LoraSpec) {
        await guard(async () => {
            await putLora({ ...spec, enabled: !spec.enabled });
            loras = await listLoras();
        });
    }

    async function compare() {
        await guard(async () => {
            compareResult = await runCompare({
                prompt,
                top_k: topK,
                max_new_tokens: maxNewTokens,
                temperature,
                seed: 0,
            });
        });
    }

    const maxKl = $derived(
        compareResult ? Math.max(1e-9, ...compareResult.positions.map((p) => p.kl_base_to_edited)) : 1,
    );
</script>

<div class="cb-root">
    {#if error}
        <div class="error-banner">{error}</div>
    {/if}

    {#if status !== "ready"}
        <div class="empty-state">
            <p>
                Circuit builder edits model weights by hand: pick a subcomponent's read direction,
                point it at downstream subcomponents via j-vectors, and see what changes.
            </p>
            <div class="row" style="justify-content: center">
                <label><input type="radio" bind:group={source} value="mock" /> mock</label>
                <label><input type="radio" bind:group={source} value="run" /> saved run</label>
                {#if source === "run"}
                    <input bind:value={runRef} placeholder="p-55ea3f9b" />
                {/if}
            </div>
            <button disabled={status === "loading"} onclick={load}>
                {status === "loading" ? "Loading…" : source === "mock" ? "Load (mock data)" : `Load ${runRef}`}
            </button>
        </div>
    {:else}
        <div class="loaded-banner">run: <strong>{loadedRunId}</strong></div>
        <div class="columns">
            <!-- ===================== LoRA list ===================== -->
            <section class="panel">
                <h3>LoRA adapters</h3>
                {#each loras as lora (lora.name)}
                    <div class="lora-card" class:disabled={!lora.enabled}>
                        <div class="lora-head">
                            <strong>{lora.name}</strong>
                            <span>
                                <button onclick={() => toggleLora(lora)}>{lora.enabled ? "on" : "off"}</button>
                                <button onclick={() => (editing = structuredClone($state.snapshot(lora)))}>edit</button>
                                <button onclick={() => removeLora(lora.name)}>✕</button>
                            </span>
                        </div>
                        <div class="lora-detail">
                            read {lora.read_site}:{lora.read_idx} → {lora.writes.length} write
                            {lora.writes.length === 1 ? "term" : "terms"}, scale {lora.scale}
                        </div>
                    </div>
                {:else}
                    <p class="hint">No adapters yet.</p>
                {/each}
                <button class="primary" onclick={newLora}>+ New LoRA</button>
            </section>

            <!-- ===================== Editor ===================== -->
            <section class="panel wide">
                {#if editing}
                    <h3>Edit: {editing.name}</h3>
                    <div class="row">
                        <label>name <input bind:value={editing.name} /></label>
                        <label>scale <input type="number" step="0.1" bind:value={editing.scale} /></label>
                        <label>n_prompts <input type="number" min="1" bind:value={editing.n_prompts} /></label>
                    </div>

                    <h4>Read vector (normalized V column)</h4>
                    <div class="row">
                        <input
                            class="search"
                            placeholder="search all labels…"
                            bind:value={readQuery}
                            oninput={() => runSearch("read")}
                        />
                    </div>
                    {#if readHits.length > 0}
                        <div class="search-results">
                            {#each readHits as hit (hit.site + ":" + hit.idx)}
                                <button class="subcomp" onclick={() => pickReadFromSearch(hit)}>
                                    <span class="idx">{hit.site}:{hit.idx}</span>
                                    <span class="label" class:fallback={hit.label_source === "fallback"}>{hit.label}</span>
                                </button>
                            {/each}
                        </div>
                    {/if}
                    <div class="row">
                        <label>
                            site
                            <select bind:value={editing.read_site} onchange={refreshReadSite}>
                                {#each sites as s (s.site)}
                                    <option value={s.site}>{s.site} (C={s.C})</option>
                                {/each}
                            </select>
                        </label>
                        <span class="hint">selected: {editing.read_site}:{editing.read_idx}</span>
                    </div>
                    <div class="subcomp-list">
                        {#each readSubcomps as sc (sc.idx)}
                            <button
                                class="subcomp"
                                class:selected={editing.read_idx === sc.idx}
                                title={sc.examples.map((e) => e.tokens.join("")).join("\n")}
                                onclick={() => {
                                    editing!.read_idx = sc.idx;
                                    void showDetail(sc.site, sc.idx);
                                }}
                            >
                                <span class="idx">{sc.idx}</span>
                                <span class="label" class:fallback={sc.label_source === "fallback"}>{sc.label ?? "(unlabeled)"}</span>
                                <span class="norm">‖U‖·‖V‖={sc.u_norm_absorbed.toFixed(2)}</span>
                            </button>
                        {/each}
                        <div class="pager">
                            <button disabled={readPage === 0} onclick={() => pageSubcomps("read", -1)}>‹</button>
                            page {readPage + 1}
                            <button onclick={() => pageSubcomps("read", 1)}>›</button>
                        </div>
                    </div>

                    <h4>Write vector (Σ λᵢ · j-vectors of downstream subcomponents)</h4>
                    <div class="row">
                        <input
                            class="search"
                            placeholder="search downstream labels…"
                            bind:value={writeQuery}
                            oninput={() => runSearch("write")}
                        />
                    </div>
                    {#if writeHits.length > 0}
                        <div class="search-results">
                            {#each writeHits as hit (hit.site + ":" + hit.idx)}
                                <button class="subcomp" onclick={() => pickWriteFromSearch(hit)}>
                                    <span class="idx">{hit.site}:{hit.idx}</span>
                                    <span class="label" class:fallback={hit.label_source === "fallback"}>{hit.label}</span>
                                    <span class="norm">+</span>
                                </button>
                            {/each}
                        </div>
                    {/if}
                    <div class="row">
                        <label>
                            downstream site
                            <select value={writeSite} onchange={(e) => changeWriteSite(e.currentTarget.value)}>
                                {#each downstream as s (s)}
                                    <option value={s}>{s}</option>
                                {/each}
                            </select>
                        </label>
                    </div>
                    <div class="subcomp-list">
                        {#each writeSubcomps as sc (sc.idx)}
                            <button
                                class="subcomp"
                                title={sc.examples.map((e) => e.tokens.join("")).join("\n")}
                                onclick={() => {
                                    addWrite(sc);
                                    void showDetail(sc.site, sc.idx);
                                }}
                            >
                                <span class="idx">{sc.idx}</span>
                                <span class="label" class:fallback={sc.label_source === "fallback"}>{sc.label ?? "(unlabeled)"}</span>
                                <span class="norm">+</span>
                            </button>
                        {/each}
                        <div class="pager">
                            <button disabled={writePage === 0} onclick={() => pageSubcomps("write", -1)}>‹</button>
                            page {writePage + 1}
                            <button onclick={() => pageSubcomps("write", 1)}>›</button>
                        </div>
                    </div>

                    {#if editing.writes.length > 0}
                        <h4>Write terms</h4>
                        {#each editing.writes as term, i (term.site + ":" + term.idx)}
                            {@const j = jInfo.find((x) => x.site === term.site && x.idx === term.idx)}
                            <div class="write-term">
                                <span>{term.site}:{term.idx}</span>
                                <label>
                                    λ
                                    <input
                                        type="number"
                                        step="0.5"
                                        bind:value={term.weight}
                                        placeholder={j ? j.raw_norm.toExponential(2) : "‖j‖"}
                                        title="prefactor on the unit j-vector; empty = default ‖j‖ (raw derivative scale)"
                                    />
                                </label>
                                {#if j}<span class="norm">‖j‖={j.raw_norm.toExponential(2)}{term.weight === null ? " (default λ)" : ""}</span>{/if}
                                <button onclick={() => removeWrite(i)}>✕</button>
                            </div>
                        {/each}
                        <button disabled={busy} onclick={computeJs}>
                            {busy ? "computing…" : `Compute j-vectors (${editing.n_prompts} prompts)`}
                        </button>
                    {/if}

                    <div class="row">
                        <button class="primary" disabled={editing.writes.length === 0} onclick={saveLora}>
                            Save LoRA
                        </button>
                        <button onclick={() => (editing = null)}>Cancel</button>
                    </div>
                {:else}
                    <p class="hint">Select a LoRA to edit, or create a new one.</p>
                {/if}
            </section>

            <!-- ===================== Component detail ===================== -->
            {#if detail}
                <section class="panel detail-panel">
                    <h3>
                        {detail.site}:{detail.idx}
                        {#if detail.label_source}
                            <span class="source-tag">{detail.label_source}</span>
                        {/if}
                    </h3>
                    <p class="detail-label">{detail.label ?? "(no autointerp label)"}</p>
                    {#if detail.reasoning}
                        <h4>Explanation</h4>
                        <p class="detail-reasoning">{detail.reasoning}</p>
                    {/if}
                    <h4>Activating examples</h4>
                    {#each detail.examples as ex, i (i)}
                        <div class="example">
                            {#each ex.tokens as tok, t (t)}
                                <span class="ex-tok" class:peak={t === ex.active_position}>{tok}</span>
                            {/each}
                            <span class="norm">act={ex.activation}</span>
                        </div>
                    {:else}
                        <p class="hint">No harvested examples for this component.</p>
                    {/each}
                    <button onclick={() => (detail = null)}>close</button>
                </section>
            {/if}

            <!-- ===================== Compare ===================== -->
            <section class="panel wide">
                <h3>Base vs edited</h3>
                <textarea rows="3" bind:value={prompt}></textarea>
                <div class="row">
                    <label>top-k <input type="number" min="1" max="50" bind:value={topK} /></label>
                    <label>new tokens <input type="number" min="1" max="256" bind:value={maxNewTokens} /></label>
                    <label>temperature <input type="number" step="0.1" min="0" bind:value={temperature} /></label>
                    <button class="primary" disabled={busy} onclick={compare}>
                        {busy ? "running…" : "Run comparison"}
                    </button>
                </div>

                {#if compareResult}
                    <div class="gen-grid">
                        {#each [
                            { name: "base", gen: compareResult.base },
                            { name: "edited", gen: compareResult.edited },
                        ] as side (side.name)}
                            <div>
                                {#each [
                                    { kind: "greedy", tokens: side.gen.greedy_tokens },
                                    { kind: "sampled", tokens: side.gen.sampled_tokens },
                                ] as variant (variant.kind)}
                                    <h4>{side.name} · {variant.kind}</h4>
                                    <div class="gen-text">
                                        {#each variant.tokens as t, i (i)}
                                            <span
                                                class="gen-tok"
                                                role="note"
                                                onmouseenter={(e) =>
                                                    showTip(
                                                        e,
                                                        `${side.name} ${variant.kind} @ +${i}`,
                                                        side.name === "base" ? t.top : null,
                                                        side.name === "edited" ? t.top : null,
                                                    )}
                                                onmouseleave={hideTip}>{t.token}</span>
                                        {/each}
                                    </div>
                                {/each}
                            </div>
                        {/each}
                    </div>
                    <h4>Per-position KL(base‖edited) — mean {compareResult.mean_kl.toExponential(3)}</h4>
                    <div class="kl-strip">
                        {#each compareResult.positions as p (p.position)}
                            <button
                                class="kl-token"
                                class:selected={expandedPosition === p.position}
                                style={`--h:${Math.round((p.kl_base_to_edited / maxKl) * 100)}%`}
                                onmouseenter={(e) =>
                                    showTip(
                                        e,
                                        `after "${p.token}" · KL=${p.kl_base_to_edited.toExponential(2)}`,
                                        p.top_base,
                                        p.top_edited,
                                    )}
                                onmouseleave={hideTip}
                                onclick={() =>
                                    (expandedPosition = expandedPosition === p.position ? null : p.position)}
                            >
                                <span class="bar"></span>
                                <span class="tok">{p.token}</span>
                            </button>
                        {/each}
                    </div>
                    {#if expandedPosition !== null}
                        {@const p = compareResult.positions[expandedPosition]}
                        <div class="topk-grid">
                            <div>
                                <h4>base @ {p.position} ("{p.token}")</h4>
                                {#each p.top_base as t (t.token_id)}
                                    <div class="tok-row">
                                        <code>{t.token}</code><span>{(t.prob * 100).toFixed(1)}%</span>
                                    </div>
                                {/each}
                            </div>
                            <div>
                                <h4>edited @ {p.position}</h4>
                                {#each p.top_edited as t (t.token_id)}
                                    <div class="tok-row">
                                        <code>{t.token}</code><span>{(t.prob * 100).toFixed(1)}%</span>
                                    </div>
                                {/each}
                            </div>
                        </div>
                    {/if}
                {/if}
            </section>
        </div>
    {/if}
    {#if tooltip}
        <div class="tooltip" style={`left:${tooltip.x}px; top:${tooltip.y}px`}>
            <div class="tooltip-title">{tooltip.title}</div>
            <div class="tooltip-cols">
                {#if tooltip.base}
                    <div>
                        {#if tooltip.edited}<div class="tooltip-col-head">base</div>{/if}
                        {#each tooltip.base as t (t.token_id)}
                            <div class="tok-row"><code>{t.token}</code><span>{(t.prob * 100).toFixed(1)}%</span></div>
                        {/each}
                    </div>
                {/if}
                {#if tooltip.edited}
                    <div>
                        {#if tooltip.base}<div class="tooltip-col-head">edited</div>{/if}
                        {#each tooltip.edited as t (t.token_id)}
                            <div class="tok-row"><code>{t.token}</code><span>{(t.prob * 100).toFixed(1)}%</span></div>
                        {/each}
                    </div>
                {/if}
            </div>
        </div>
    {/if}
</div>

<style>
    .cb-root {
        padding: 1rem;
    }
    .error-banner {
        background: #fee;
        border: 1px solid #c00;
        color: #900;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    .columns {
        display: grid;
        grid-template-columns: 260px 1fr 1fr;
        gap: 1rem;
        align-items: start;
    }
    .panel {
        background: var(--bg-surface, #fff);
        border: 1px solid var(--border-default, #ddd);
        border-radius: 6px;
        padding: 0.75rem;
    }
    .row {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        flex-wrap: wrap;
        margin: 0.5rem 0;
    }
    .row label {
        display: flex;
        gap: 0.3rem;
        align-items: center;
        font-size: 0.85rem;
    }
    input[type="number"] {
        width: 5rem;
    }
    textarea {
        width: 100%;
        box-sizing: border-box;
    }
    .subcomp-list {
        max-height: 220px;
        overflow-y: auto;
        border: 1px solid var(--border-default, #eee);
        border-radius: 4px;
    }
    .subcomp {
        display: flex;
        gap: 0.5rem;
        width: 100%;
        text-align: left;
        border: none;
        background: none;
        padding: 0.25rem 0.5rem;
        cursor: pointer;
        font-size: 0.85rem;
    }
    .subcomp:hover {
        background: var(--bg-hover, #f5f5f5);
    }
    .subcomp.selected {
        background: var(--accent-soft, #e3f0ff);
    }
    .subcomp .idx {
        min-width: 3rem;
        color: var(--text-muted, #888);
    }
    .subcomp .label {
        flex: 1;
    }
    .subcomp .norm {
        color: var(--text-muted, #888);
    }
    .pager {
        display: flex;
        gap: 0.5rem;
        justify-content: center;
        padding: 0.25rem;
        font-size: 0.8rem;
    }
    .write-term {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        font-size: 0.85rem;
        padding: 0.15rem 0;
    }
    .lora-card {
        border: 1px solid var(--border-default, #ddd);
        border-radius: 4px;
        padding: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .lora-card.disabled {
        opacity: 0.5;
    }
    .lora-head {
        display: flex;
        justify-content: space-between;
    }
    .lora-detail {
        font-size: 0.75rem;
        color: var(--text-muted, #777);
    }
    .gen-grid,
    .topk-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    .gen-text {
        white-space: pre-wrap;
        background: var(--bg-base, #f8f8f8);
        padding: 0.5rem;
        border-radius: 4px;
        max-height: 8rem;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .gen-tok:hover {
        background: var(--accent-soft, #e3f0ff);
        outline: 1px solid var(--accent, #2563eb);
    }
    .tooltip {
        position: fixed;
        z-index: 1000;
        background: var(--bg-surface, #fff);
        border: 1px solid var(--border-default, #bbb);
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        padding: 0.5rem 0.75rem;
        pointer-events: none;
        max-width: 24rem;
    }
    .tooltip-title {
        font-size: 0.75rem;
        color: var(--text-muted, #777);
        margin-bottom: 0.25rem;
    }
    .tooltip-cols {
        display: flex;
        gap: 1rem;
    }
    .tooltip-col-head {
        font-size: 0.75rem;
        font-weight: bold;
    }
    .kl-strip {
        display: flex;
        flex-wrap: wrap;
        align-items: flex-end;
        gap: 1px;
    }
    .kl-token {
        display: flex;
        flex-direction: column;
        align-items: center;
        border: none;
        background: none;
        cursor: pointer;
        padding: 0;
    }
    .kl-token .bar {
        width: 100%;
        min-width: 0.8rem;
        height: 40px;
        background: linear-gradient(to top, var(--accent, #d33) var(--h), transparent var(--h));
    }
    .kl-token .tok {
        font-size: 0.7rem;
        font-family: monospace;
    }
    .kl-token.selected .tok {
        font-weight: bold;
    }
    .tok-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
    }
    .hint {
        color: var(--text-muted, #888);
        font-size: 0.85rem;
    }
    .search {
        width: 100%;
        box-sizing: border-box;
        padding: 0.3rem 0.5rem;
    }
    .search-results {
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid var(--accent, #2563eb);
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .label.fallback {
        opacity: 0.55;
        font-style: italic;
    }
    .detail-panel {
        grid-column: 1 / -1;
    }
    .source-tag {
        font-size: 0.7rem;
        border: 1px solid var(--border-default, #bbb);
        border-radius: 3px;
        padding: 0.05rem 0.3rem;
        color: var(--text-muted, #777);
        vertical-align: middle;
    }
    .detail-label {
        font-weight: 600;
    }
    .detail-reasoning {
        white-space: pre-wrap;
        font-size: 0.9rem;
    }
    .example {
        font-family: monospace;
        font-size: 0.8rem;
        padding: 0.2rem 0;
        border-bottom: 1px dashed var(--border-default, #eee);
    }
    .ex-tok.peak {
        background: var(--accent-soft, #ffe08a);
        font-weight: bold;
        border-radius: 2px;
    }
    .primary {
        background: var(--accent, #2563eb);
        color: white;
        border: none;
        padding: 0.35rem 0.8rem;
        border-radius: 4px;
        cursor: pointer;
    }
    .empty-state {
        max-width: 32rem;
        margin: 3rem auto;
        text-align: center;
    }
    .loaded-banner {
        font-size: 0.8rem;
        color: var(--text-muted, #777);
        margin-bottom: 0.5rem;
    }
</style>
