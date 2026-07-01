<script lang="ts">
    import { goto } from "$app/navigation";
    import { page } from "$app/state";
    import { fmtCi, fmtDensity, matrixKind, parseSite } from "$lib/format";
    import type { Snippet } from "svelte";
    import type { ComponentListing } from "$lib/types";

    let {
        data,
        children,
    }: {
        data: {
            run: string;
            site: string;
            sites: string[];
            sort: string;
            page: number;
            q: string;
            listing: ComponentListing;
            pageSize: number;
        };
        children: Snippet;
    } = $props();

    const mk = $derived(matrixKind(data.site));

    // svelte-ignore state_referenced_locally — resynced from data on navigation
    let qInput = $state(data.q);
    $effect(() => {
        qInput = data.q;
    });

    const selectedIdx = $derived((page.params as { idx?: string }).idx ?? null);
    const lastPage = $derived(Math.max(0, Math.ceil(data.listing.total / data.pageSize) - 1));

    const carryQuery = $derived(
        new URLSearchParams({ sort: data.sort, page: String(data.page), q: data.q }).toString(),
    );

    function go(overrides: { sort?: string; page?: number; q?: string }) {
        const params = new URLSearchParams({
            sort: overrides.sort ?? data.sort,
            page: String(overrides.page ?? 0),
            q: overrides.q ?? data.q,
        });
        goto(`/r/${data.run}/s/${data.site}?${params}`);
    }

    function switchSite(site: string) {
        goto(`/r/${data.run}/s/${site}?sort=${data.sort}`);
    }
</script>

<div class="split">
    <aside class="sidebar">
        <div class="head">
            <div class="site-row">
                {#if mk}<span class="mtag {mk}">{mk}</span>{/if}
                <select
                    class="site-select"
                    value={data.site}
                    onchange={(e) => switchSite(e.currentTarget.value)}
                >
                    {#each data.sites as s (s)}
                        {@const p = parseSite(s)}
                        <option value={s}>L{p.layer} · {p.kind}</option>
                    {/each}
                </select>
            </div>

            <form
                class="search"
                onsubmit={(e) => {
                    e.preventDefault();
                    go({ q: qInput, page: 0 });
                }}
            >
                <input type="search" placeholder="search labels…" bind:value={qInput} />
            </form>

            <div class="sort-row">
                <span class="eyebrow">sort</span>
                <select value={data.sort} onchange={(e) => go({ sort: e.currentTarget.value, page: 0 })}>
                    <option value="mean_ci">mean CI</option>
                    <option value="density">density</option>
                    <option value="max_act">max activation</option>
                    <option value="unlabeled_first">unlabeled first</option>
                </select>
                <span class="count num">{data.listing.total.toLocaleString("en-US")}</span>
            </div>
        </div>

        <ol class="list">
            {#each data.listing.items as row (row.idx)}
                <li>
                    <a
                        class="item"
                        class:selected={selectedIdx === String(row.idx)}
                        href="/r/{data.run}/s/{data.site}/c/{row.idx}?{carryQuery}"
                    >
                        <div class="item-top">
                            <span class="idx num">#{row.idx}</span>
                            <span class="metric num">ci {fmtCi(row.mean_ci)}</span>
                            <span class="metric num dim">ρ {fmtDensity(row.density)}</span>
                        </div>
                        <div class="label" class:unlabeled={row.label === null}>
                            {row.label ?? "unlabeled"}
                        </div>
                    </a>
                </li>
            {/each}
        </ol>

        {#if data.listing.total === 0}
            <div class="empty">
                <p class="subline">No components{data.q ? ` matching “${data.q}”` : ""}.</p>
                {#if data.q}
                    <button class="quiet" onclick={() => go({ q: "", page: 0 })}>clear search</button>
                {/if}
            </div>
        {:else}
            <div class="pager">
                <button class="quiet" disabled={data.page === 0} onclick={() => go({ page: data.page - 1 })}>
                    ← prev
                </button>
                <span class="eyebrow">
                    {data.page + 1} / {lastPage + 1}
                </span>
                <button
                    class="quiet"
                    disabled={data.page >= lastPage}
                    onclick={() => go({ page: data.page + 1 })}
                >
                    next →
                </button>
            </div>
        {/if}
    </aside>

    <div class="detail">
        {@render children()}
    </div>
</div>

<style>
    .split {
        display: flex;
        height: 100%;
    }
    .sidebar {
        width: 420px;
        flex-shrink: 0;
        border-right: 1px solid var(--line);
        display: flex;
        flex-direction: column;
        min-height: 0;
    }
    .head {
        flex-shrink: 0;
        padding: 14px 16px 12px;
        border-bottom: 1px solid var(--line);
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .site-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .site-select {
        flex: 1;
    }
    .search input {
        width: 100%;
    }
    .sort-row {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sort-row select {
        flex: 1;
    }
    .count {
        color: var(--dim);
        font-size: 12px;
    }
    .list {
        list-style: none;
        margin: 0;
        padding: 0;
        flex: 1;
        overflow-y: auto;
        min-height: 0;
    }
    .item {
        display: block;
        padding: 8px 16px;
        border-bottom: 1px solid var(--line);
        text-decoration: none;
        color: var(--fg);
        border-left: 2px solid transparent;
    }
    .item:hover {
        background: var(--panel);
    }
    .item.selected {
        background: var(--accent-wash);
        border-left-color: var(--accent);
    }
    .item-top {
        display: flex;
        align-items: baseline;
        gap: 10px;
    }
    .idx {
        color: var(--accent);
        font-size: 12.5px;
    }
    .metric {
        font-size: 11.5px;
    }
    .label {
        font-size: 13px;
        line-height: 1.45;
        margin-top: 2px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .label.unlabeled {
        color: var(--faint);
        font-style: italic;
    }
    .empty {
        padding: 24px 16px;
    }
    .empty button {
        margin-top: 12px;
    }
    .pager {
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 10px 16px;
        border-top: 1px solid var(--line);
    }
    .detail {
        flex: 1;
        min-height: 0;
        overflow-y: auto;
    }
</style>
