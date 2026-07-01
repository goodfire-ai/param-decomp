<script lang="ts">
    import { fmtCount } from "$lib/format";
    import type { Catalog, Run } from "$lib/types";

    let { data }: { data: { catalog: Catalog } } = $props();
    const catalog = $derived(data.catalog);

    type Decomp = {
        run_id: string;
        nComponents: number;
        nLabeled: number;
        nSites: number;
        firstSite: string | null;
    };

    function summarize(run: Run): Decomp {
        const present = run.sites.filter(
            (s) => s.n_components > 0 && s.subruns.some((r) => r.status === "present"),
        );
        return {
            run_id: run.run_id,
            nComponents: run.sites.reduce((a, s) => a + s.n_components, 0),
            nLabeled: run.sites.reduce((a, s) => a + s.n_labeled, 0),
            nSites: present.length,
            firstSite: present.length > 0 ? present[0].site : null,
        };
    }

    const decomps = $derived(catalog.runs.map(summarize));
</script>

<svelte:head><title>Scope · Decompositions</title></svelte:head>

<div class="scroll">
    <div class="doc">
        <h1>Decompositions</h1>
        <p class="preamble subline">
            Each decomposition splits a model's weight matrices into sparse components. Open one to
            browse its components.
        </p>

        <ol class="list">
            {#each decomps as d (d.run_id)}
                <li>
                    {#if d.firstSite !== null}
                        <a class="row" href="/r/{d.run_id}/s/{d.firstSite}">
                            <span class="run mono">{d.run_id}</span>
                            <span class="stats subline">
                                {fmtCount(d.nComponents)} components · {fmtCount(d.nLabeled)} labeled
                                · {d.nSites} sites
                            </span>
                            <span class="go" aria-hidden="true">→</span>
                        </a>
                    {:else}
                        <div class="row pending">
                            <span class="run mono">{d.run_id}</span>
                            <span class="stats subline">
                                <span class="sq live pulse"></span> harvesting — no components published
                                yet
                            </span>
                        </div>
                    {/if}
                </li>
            {/each}
        </ol>
    </div>
</div>

<style>
    .scroll {
        height: 100%;
        overflow: auto;
    }
    .doc {
        max-width: 760px;
        margin: 0 auto;
        padding: 48px 32px 96px;
    }
    h1 {
        font-size: 22px;
        margin: 0 0 8px;
    }
    .preamble {
        max-width: 560px;
        margin: 0 0 40px;
    }
    .list {
        list-style: none;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .row {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px 20px;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: var(--panel);
        text-decoration: none;
        color: var(--fg);
    }
    a.row:hover {
        border-color: var(--line-2);
        background: var(--panel-2);
    }
    .run {
        font-size: 15px;
        font-weight: 500;
    }
    .stats {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-left: auto;
    }
    a.row .go {
        color: var(--dim);
        font-size: 16px;
    }
    a.row:hover .go {
        color: var(--accent);
    }
    .pending {
        opacity: 0.75;
    }
    .pending .stats {
        color: var(--dim);
    }
    .sq {
        width: 7px;
        height: 7px;
        flex-shrink: 0;
        display: inline-block;
    }
    .sq.live {
        background: var(--accent);
    }
</style>
