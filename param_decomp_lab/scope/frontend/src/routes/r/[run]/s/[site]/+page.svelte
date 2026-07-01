<script lang="ts">
    import { goto } from "$app/navigation";
    import { fmtAct, fmtCount, fmtDensity } from "$lib/format";
    import SiteCurve from "$lib/ui/SiteCurve.svelte";
    import type { ComponentLabel } from "$lib/types";

    let { data } = $props();

    // svelte-ignore state_referenced_locally — resynced from data on navigation below
    let qInput = $state(data.q);
    $effect(() => {
        qInput = data.q;
    });

    let postedLabels: Record<string, string> = $state({});
    let labelingIdx: number | null = $state(null);

    const labelKey = (idx: number) => `${data.run}|${data.site}|${idx}`;

    const lastPage = $derived(Math.max(0, Math.ceil(data.listing.total / data.pageSize) - 1));

    function navigate(overrides: { sort?: string; page?: number; q?: string }) {
        const params = new URLSearchParams({
            sort: overrides.sort ?? data.sort,
            page: String(overrides.page ?? 0),
            q: overrides.q ?? data.q,
        });
        goto(`/r/${data.run}/s/${data.site}?${params}`);
    }

    async function requestLabel(idx: number) {
        labelingIdx = idx;
        const res = await fetch(
            `/api/runs/${data.run}/sites/${data.site}/components/${idx}/label`,
            { method: "POST" },
        );
        if (!res.ok) throw new Error(`label request failed: ${res.status}`);
        const label: ComponentLabel = await res.json();
        postedLabels[labelKey(idx)] = label.text;
        labelingIdx = null;
    }
</script>

<svelte:head><title>Scope · {data.site}</title></svelte:head>

<hgroup>
    <h1>{data.site}</h1>
    <p class="subline">
        run <a href="/">{data.run}</a> · {fmtCount(data.listing.total)} components{data.q
            ? ` matching “${data.q}”`
            : ""}
    </p>
</hgroup>

<SiteCurve curve={data.curve} currentRank={null} run={data.run} site={data.site} />

<form
    class="controls"
    onsubmit={(e) => {
        e.preventDefault();
        navigate({ q: qInput, page: 0 });
    }}
>
    <label class="eyebrow">
        sort
        <select
            value={data.sort}
            onchange={(e) => navigate({ sort: e.currentTarget.value, page: 0 })}
        >
            <option value="density">density</option>
            <option value="max_act">max activation</option>
            <option value="unlabeled_first">unlabeled first</option>
        </select>
    </label>
    <label class="eyebrow search">
        search labels
        <input type="search" placeholder="e.g. maritime" bind:value={qInput} />
    </label>
    <button class="quiet" type="submit">search</button>
</form>

{#if data.listing.items.length === 0 && data.q}
    <div class="empty panel">
        <p>No components match “{data.q}”.</p>
        <p class="subline">Clear the search to see all {fmtCount(data.curve.total)}.</p>
        <button class="quiet" onclick={() => navigate({ q: "", page: 0 })}>clear search</button>
    </div>
{:else}
    <table>
        <thead>
            <tr>
                <th class="r col-idx">idx</th>
                <th class="r col-num">density</th>
                <th class="r col-num">max act</th>
                <th>label</th>
            </tr>
        </thead>
        <tbody>
            {#each data.listing.items as row (row.idx)}
                {@const labelText = postedLabels[labelKey(row.idx)] ?? row.label}
                <tr>
                    <td class="num r">
                        <a class="idx" href="/r/{data.run}/s/{data.site}/c/{row.idx}">{row.idx}</a>
                    </td>
                    <td class="num r">{fmtDensity(row.density)}</td>
                    <td class="num r">{fmtAct(row.max_act)}</td>
                    <td>
                        {#if labelText !== null}
                            {labelText}
                        {:else if labelingIdx === row.idx}
                            <span class="dim">labelling…</span>
                        {:else}
                            <button class="quiet" onclick={() => requestLabel(row.idx)}>
                                label · $0.03
                            </button>
                        {/if}
                    </td>
                </tr>
            {/each}
        </tbody>
    </table>

    <nav class="pager">
        <button class="quiet" disabled={data.page === 0} onclick={() => navigate({ page: 0 })}>
            ⇤ first
        </button>
        <button
            class="quiet"
            disabled={data.page === 0}
            onclick={() => navigate({ page: data.page - 1 })}
        >
            ← prev
        </button>
        <span class="eyebrow">page <span class="num">{data.page + 1}</span> of
            <span class="num">{fmtCount(lastPage + 1)}</span></span>
        <button
            class="quiet"
            disabled={data.page >= lastPage}
            onclick={() => navigate({ page: data.page + 1 })}
        >
            next →
        </button>
        <button
            class="quiet"
            disabled={data.page >= lastPage}
            onclick={() => navigate({ page: lastPage })}
        >
            last ⇥
        </button>
    </nav>
{/if}

<style>
    hgroup h1 {
        font-family: var(--mono);
        font-size: 20px;
        font-weight: 500;
        letter-spacing: 0;
        margin: 0;
    }
    hgroup p {
        margin: 2px 0 24px;
    }
    .controls {
        display: flex;
        align-items: baseline;
        gap: 24px;
        margin-bottom: 16px;
    }
    .controls label {
        display: inline-flex;
        align-items: baseline;
        gap: 8px;
    }
    .search input {
        width: 240px;
    }
    .col-idx {
        width: 88px;
    }
    .col-num {
        width: 120px;
    }
    td {
        font-size: 14px;
    }
    a.idx {
        color: var(--accent);
        text-decoration: none;
    }
    a.idx:hover {
        text-decoration: underline;
        text-decoration-color: var(--accent);
        text-underline-offset: 3px;
    }
    .empty {
        padding: 32px;
        text-align: center;
    }
    .empty p {
        margin: 0 0 4px;
    }
    .empty button {
        margin-top: 12px;
    }
    .pager {
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin-top: 16px;
    }
    .pager > span {
        margin: 0 8px;
    }
</style>
