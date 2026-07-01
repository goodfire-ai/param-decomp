<script lang="ts">
    import "../app.css";
    import { page } from "$app/state";
    import type { Snippet } from "svelte";

    let { children }: { children: Snippet } = $props();

    function toggleTheme() {
        const html = document.documentElement;
        const next = html.dataset.theme === "dark" ? "light" : "dark";
        html.dataset.theme = next;
        localStorage.setItem("scope-theme", next);
    }

    const crumbs = $derived.by(() => {
        const p = page.params as { run?: string; site?: string; idx?: string };
        const out: { href: string; text: string }[] = [];
        if (p.run && p.site) {
            out.push({ href: `/r/${p.run}/s/${p.site}`, text: `${p.run} · ${p.site}` });
            if (p.idx !== undefined) {
                out.push({
                    href: `/r/${p.run}/s/${p.site}/c/${p.idx}`,
                    text: `component ${p.idx}`,
                });
            }
        }
        return out;
    });
</script>

<header>
    <div class="bar">
        <a href="/" class="brand">goodfire // scope</a>
        <nav>
            <a href="/" class:current={crumbs.length === 0}>catalogue</a>
            {#each crumbs as crumb, i (crumb.href)}
                <span class="sep">/</span>
                <a href={crumb.href} class:current={i === crumbs.length - 1}>{crumb.text}</a>
            {/each}
        </nav>
        <button class="theme" onclick={toggleTheme}>
            <span class="to-dark">◐ dark mode</span><span class="to-light">◐ light mode</span>
        </button>
    </div>
</header>

<main>
    {@render children()}
</main>

<footer>
    <p>scope · param-decomp</p>
</footer>

<style>
    header {
        border-bottom: 1px solid var(--line);
    }
    .bar {
        max-width: 1100px;
        margin: 0 auto;
        padding: 12px 32px;
        display: flex;
        align-items: center;
        gap: 24px;
    }
    .brand {
        font-family: var(--mono);
        font-size: 11px;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--dim);
        text-decoration: none;
        white-space: nowrap;
    }
    .brand:hover {
        color: var(--fg);
    }
    nav {
        margin-left: auto;
        font-family: var(--mono);
        font-size: 11px;
        letter-spacing: 0.02em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 55%;
    }
    nav a {
        color: var(--dim);
        text-decoration: none;
    }
    nav a:hover {
        color: var(--fg);
    }
    nav a.current {
        color: var(--fg);
    }
    .sep {
        color: var(--faint);
        margin: 0 8px;
    }
    .theme {
        font-family: var(--mono);
        font-size: 10.5px;
        font-weight: 400;
        color: var(--dim);
        background: none;
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 4px 10px;
        cursor: pointer;
        white-space: nowrap;
    }
    .theme:hover {
        color: var(--fg);
        border-color: var(--line-2);
    }
    .theme .to-light {
        display: none;
    }
    :global(html[data-theme="dark"]) .theme .to-dark {
        display: none;
    }
    :global(html[data-theme="dark"]) .theme .to-light {
        display: inline;
    }
    main {
        padding-top: 32px;
    }
    footer {
        max-width: 1100px;
        margin: 0 auto;
        padding: 0 32px 24px;
    }
    footer p {
        font-family: var(--mono);
        font-size: 10px;
        color: var(--faint);
    }
</style>
