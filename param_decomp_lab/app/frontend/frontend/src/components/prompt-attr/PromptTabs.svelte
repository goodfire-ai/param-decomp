<script lang="ts">
    import type { PromptCard, TabViewState } from "./types";

    type Props = {
        cards: PromptCard[];
        tabView: TabViewState;
        onSelectCard: (cardId: number) => void;
        onCloseCard: (cardId: number) => void;
        onSelectDraft: () => void;
        onAddClick: () => void;
    };

    let { cards, tabView, onSelectCard, onCloseCard, onSelectDraft, onAddClick }: Props = $props();

    function getCardLabel(card: PromptCard): string {
        const nCharsToShow = 30;
        const str = card.tokens.join("");
        return str.slice(0, nCharsToShow) + (str.length > nCharsToShow ? "..." : "");
    }
</script>

<div class="card-tabs">
    {#each cards as card (card.id)}
        <div class="card-tab" class:active={tabView.view === "card" && tabView.cardId === card.id}>
            <button class="card-tab-label" onclick={() => onSelectCard(card.id)}>
                {getCardLabel(card)}
            </button>
            <button class="card-tab-close" onclick={() => onCloseCard(card.id)}>×</button>
        </div>
    {/each}
    {#if tabView.view === "loading"}
        <div class="card-tab loading-tab active">
            <span class="card-tab-label loading-label">Loading...</span>
        </div>
    {:else if tabView.view === "error"}
        <div class="card-tab error-tab active">
            <span class="card-tab-label error-label">Error</span>
        </div>
    {:else if tabView.view === "draft"}
        <div class="card-tab active">
            <button class="card-tab-label" onclick={onSelectDraft}> New Prompt </button>
        </div>
    {/if}
    <button class="btn-add-tab" onclick={onAddClick} disabled={tabView.view === "draft" || tabView.view === "loading"}
        >+</button
    >
</div>

<style>
    .card-tabs {
        display: flex;
        gap: var(--space-2);
        flex: 1;
        overflow-x: auto;
    }

    .card-tab {
        display: flex;
        align-items: center;
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        color: var(--text-muted);
        flex-shrink: 0;
    }

    .card-tab:hover {
        background: var(--bg-inset);
        border-color: var(--border-strong);
    }

    .card-tab.active {
        background: var(--bg-elevated);
        border-color: var(--border-strong);
        color: var(--text-primary);
    }

    .card-tab.loading-tab {
        border-style: dashed;
        border-color: var(--text-muted);
    }

    .loading-label {
        color: var(--text-muted);
        font-style: italic;
    }

    .card-tab.error-tab {
        border-style: dashed;
        border-color: var(--status-negative);
    }

    .error-label {
        color: var(--status-negative);
        font-style: italic;
    }

    .card-tab-label {
        padding: var(--space-1) var(--space-2);
        background: transparent;
        border: none;
        font-size: inherit;
        font-family: inherit;
        color: inherit;
        cursor: pointer;
        max-width: 140px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .card-tab-close {
        padding: var(--space-1);
        background: transparent;
        border: none;
        border-left: 1px solid var(--border-subtle);
        font-size: var(--text-sm);
        line-height: 1;
        opacity: 0.5;
        cursor: pointer;
        color: inherit;
    }

    .card-tab-close:hover {
        opacity: 1;
        color: var(--status-negative-bright);
    }

    .btn-add-tab {
        padding: var(--space-1) var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        color: var(--text-muted);
        flex-shrink: 0;
        font-size: var(--text-sm);
        font-family: var(--font-mono);
        cursor: pointer;
    }

    .btn-add-tab:hover:not(:disabled) {
        background: var(--bg-inset);
        border-color: var(--border-strong);
        color: var(--text-primary);
    }

    .btn-add-tab:disabled {
        opacity: 0.4;
        cursor: not-allowed;
    }
</style>
