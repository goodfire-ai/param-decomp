<script lang="ts">
    type Props = {
        scale: number;
        onZoomIn: () => void;
        onZoomOut: () => void;
        onReset: () => void;
        hint?: string;
    };

    let { scale, onZoomIn, onZoomOut, onReset, hint }: Props = $props();
</script>

<div class="zoom-controls">
    <button class="zoom-btn" onclick={onZoomIn} title="Zoom in">+</button>
    <span class="zoom-level">{Math.round(scale * 100)}%</span>
    <button class="zoom-btn" onclick={onZoomOut} title="Zoom out">−</button>
    <button class="zoom-btn reset" onclick={onReset} title="Reset zoom">⟲</button>
    {#if hint}
        <span class="zoom-hint">{hint}</span>
    {/if}
</div>

<style>
    .zoom-controls {
        position: absolute;
        top: var(--space-2);
        right: var(--space-2);
        z-index: 20;
        display: flex;
        align-items: center;
        gap: var(--space-1);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        padding: var(--space-1);
    }

    .zoom-btn {
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        color: var(--text-secondary);
        font-size: var(--text-sm);
        cursor: pointer;
        transition: all var(--transition-fast);
    }

    .zoom-btn:hover {
        background: var(--bg-inset);
        border-color: var(--border-default);
        color: var(--text-primary);
    }

    .zoom-btn.reset {
        font-size: var(--text-base);
    }

    .zoom-level {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        min-width: 40px;
        text-align: center;
    }

    .zoom-hint {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        margin-left: var(--space-1);
    }
</style>
