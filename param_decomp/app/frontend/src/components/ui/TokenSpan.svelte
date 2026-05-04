<script lang="ts">
    import { sanitizeToken } from "../../lib/tokenUtils";

    interface Props {
        token: string;
        tooltip?: string;
        backgroundColor?: string;
    }

    let { token, tooltip = "", backgroundColor = "transparent" }: Props = $props();
</script>

<span class="token-span" style="background-color: {backgroundColor}" data-tooltip={tooltip}>{sanitizeToken(token)}</span
>

<style>
    .token-span {
        white-space: pre;
        font-family: var(--font-mono);
        position: relative;
    }

    .token-span::after {
        content: attr(data-tooltip);
        position: absolute;
        top: calc(100% + 4px);
        left: 0;
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        color: var(--text-primary);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        z-index: 1000;
    }

    .token-span:hover::after {
        opacity: 1;
    }
</style>
