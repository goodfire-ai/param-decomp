<script lang="ts">
    import { onMount } from "svelte";
    import { fmtCount, matrixKind, parseSite } from "$lib/format";
    import type { Catalog, Site } from "$lib/types";

    let { data } = $props();
    let polled: Catalog | null = $state(null);
    const catalog: Catalog = $derived(polled ?? data.catalog);

    onMount(() => {
        const id = setInterval(async () => {
            const res = await fetch("/api/catalog");
            if (!res.ok) throw new Error(`catalog poll failed: ${res.status}`);
            polled = await res.json();
        }, 5000);
        return () => clearInterval(id);
    });

    type Cell =
        | { state: "absent" }
        | { state: "in_flight"; progress: number; site: Site }
        | { state: "present"; site: Site };

    function cellOf(site: Site | undefined): Cell {
        if (!site || site.subruns.length === 0) return { state: "absent" };
        if (site.subruns.some((s) => s.status === "present")) return { state: "present", site };
        const progress = Math.max(...site.subruns.map((s) => s.progress));
        return { state: "in_flight", progress, site };
    }

    type RunGrid = {
        run_id: string;
        kinds: string[];
        rows: { layer: number; cells: Cell[] }[];
        nComponents: number;
        nLabeled: number;
    };

    const grids: RunGrid[] = $derived(
        catalog.runs.map((run) => {
            const parsed = run.sites.map((s) => ({ site: s, ...parseSite(s.site) }));
            const kinds = [...new Set(parsed.map((p) => p.kind))].sort();
            const layers = [...new Set(parsed.map((p) => p.layer))].sort((a, b) => a - b);
            const rows = layers.map((layer) => ({
                layer,
                cells: kinds.map((kind) =>
                    cellOf(parsed.find((p) => p.layer === layer && p.kind === kind)?.site),
                ),
            }));
            return {
                run_id: run.run_id,
                kinds,
                rows,
                nComponents: run.sites.reduce((a, s) => a + s.n_components, 0),
                nLabeled: run.sites.reduce((a, s) => a + s.n_labeled, 0),
            };
        }),
    );
</script>

<svelte:head><title>Scope · Catalogue</title></svelte:head>

<h1>Catalogue of decompositions</h1>
<p class="preamble subline">
    Each run decomposes a set of weight matrices (sites) into sparse components. Harvest subruns
    arrive incrementally — cells below fill in as postprocessing lands. Present sites open in the
    component browser.
</p>

{#each grids as grid (grid.run_id)}
    <section>
        <h2>{grid.run_id}</h2>
        <p class="run-stats subline">
            {fmtCount(grid.nComponents)} components · {fmtCount(grid.nLabeled)} labeled
        </p>
        <table class="grid">
            <thead>
                <tr>
                    <th class="r col-layer">layer</th>
                    {#each grid.kinds as kind (kind)}
                        {@const mk = matrixKind(kind)}
                        <th>{#if mk}<span class="mtag {mk}">{mk}</span>{:else}{kind}{/if}</th>
                    {/each}
                </tr>
            </thead>
            <tbody>
                {#each grid.rows as row (row.layer)}
                    <tr>
                        <td class="num r layer-cell">{row.layer}</td>
                        {#each row.cells as cell, i (i)}
                            <td>
                                {#if cell.state === "present"}
                                    <a
                                        class="present"
                                        href="/r/{grid.run_id}/s/{cell.site.site}"
                                        title="{fmtCount(cell.site.n_components)} components, {fmtCount(
                                            cell.site.n_labeled,
                                        )} labeled"
                                    >
                                        <span class="sq ok"></span>
                                        <span class="num">{fmtCount(cell.site.n_components)}</span>
                                    </a>
                                {:else if cell.state === "in_flight"}
                                    <span class="inflight">
                                        <span class="sq live pulse"></span>
                                        <span class="num">{Math.round(cell.progress * 100)}%</span>
                                        <span class="track"
                                            ><span
                                                class="fill"
                                                style="width: {cell.progress * 100}%"
                                            ></span></span
                                        >
                                    </span>
                                {:else}
                                    <span class="faint">—</span>
                                {/if}
                            </td>
                        {/each}
                    </tr>
                {/each}
            </tbody>
        </table>
    </section>
{/each}

<p class="colophon eyebrow">catalogue refreshes every five seconds</p>

<style>
    h1 {
        font-size: 22px;
        margin: 0 0 8px;
    }
    .preamble {
        max-width: 640px;
        margin: 0 0 48px;
    }
    section {
        margin-bottom: 48px;
    }
    h2 {
        font-family: var(--mono);
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0;
        margin: 0;
    }
    .run-stats {
        margin: 2px 0 16px;
    }
    table.grid {
        max-width: 900px;
    }
    .col-layer,
    .layer-cell {
        width: 64px;
    }
    .layer-cell {
        color: var(--dim);
    }
    .sq {
        width: 7px;
        height: 7px;
        flex-shrink: 0;
    }
    .sq.ok {
        background: var(--ok);
    }
    .sq.live {
        background: var(--accent);
    }
    a.present {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: var(--fg);
        text-decoration: none;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 2px 10px;
    }
    a.present:hover {
        border-color: var(--line-2);
        background: var(--panel-2);
    }
    .inflight {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--dim);
        font-size: 13px;
    }
    .track {
        display: inline-block;
        width: 56px;
        height: 4px;
        background: var(--line);
        border-radius: 2px;
        overflow: hidden;
    }
    .fill {
        display: block;
        height: 100%;
        background: rgba(var(--hl), 0.55);
    }
    .colophon {
        margin-top: 16px;
    }
</style>
