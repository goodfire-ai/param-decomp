<script lang="ts">
    import type { Snippet } from "svelte";

    type Props = {
        variant?: "muted" | "pending" | "running" | "completed" | "failed";
        compact?: boolean;
        children: Snippet;
    };

    let { variant = "muted", compact = false, children }: Props = $props();
</script>

<p class="status-text {variant}" class:compact>
    {@render children()}
</p>

<style>
    .status-text {
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        margin: 0;
    }

    .status-text.compact {
        font-size: var(--text-xs);
    }

    .status-text.muted {
        color: var(--text-muted);
    }

    .status-text.pending,
    .status-text.running {
        color: var(--accent-primary);
    }

    .status-text.completed {
        color: var(--status-positive);
    }

    .status-text.failed {
        color: var(--status-negative-bright);
    }
</style>
