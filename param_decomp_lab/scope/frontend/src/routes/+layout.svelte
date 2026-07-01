<script lang="ts">
    import "../app.css";
    import { page } from "$app/state";
    import { matrixKind } from "$lib/format";
    import type { Snippet } from "svelte";

    let { children }: { children: Snippet } = $props();

    const crumb = $derived.by(() => {
        const p = page.params as { run?: string; site?: string; idx?: string };
        if (!p.run || !p.site) return null;
        return { run: p.run, site: p.site, mk: matrixKind(p.site), idx: p.idx };
    });
</script>

<div class="app">
    <header>
        <a href="/" class="brand">goodfire // scope</a>
        {#if crumb}
            <span class="sep">/</span>
            <span class="here">
                <span class="run mono">{crumb.run}</span>
                <span class="sep">/</span>
                {#if crumb.mk}<span class="mtag {crumb.mk}">{crumb.mk}</span>{/if}
                <span class="site mono">{crumb.site}</span>
                {#if crumb.idx !== undefined}
                    <span class="sep">/</span>
                    <span class="mono">#{crumb.idx}</span>
                {/if}
            </span>
        {/if}
    </header>
    <div class="content">
        {@render children()}
    </div>
</div>

<style>
    .app {
        height: 100vh;
        display: flex;
        flex-direction: column;
    }
    header {
        flex-shrink: 0;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 16px;
        border-bottom: 1px solid var(--line);
        white-space: nowrap;
        overflow: hidden;
    }
    .brand {
        font-family: var(--mono);
        font-size: 11px;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--dim);
        text-decoration: none;
    }
    .brand:hover {
        color: var(--fg);
    }
    .here {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .run {
        color: var(--dim);
    }
    .site {
        color: var(--fg);
    }
    .sep {
        color: var(--faint);
    }
    .content {
        flex: 1;
        min-height: 0;
    }
</style>
