<script lang="ts">
    import type { TokenSearchResult } from "../../lib/promptAttributionsTypes";
    import * as api from "../../lib/api";
    import { sanitizeToken } from "../../lib/tokenUtils";

    type Props = {
        value: string;
        selectedTokenId: number | null;
        onSelect: (tokenId: number | null, tokenString: string) => void;
        promptId: number;
        position: number;
        placeholder?: string;
    };

    let { value, onSelect, promptId, position, placeholder = "Search tokens..." }: Props = $props();

    let inputValue = $derived(value);
    let isOpen = $state(false);
    let highlightedIndex = $state(0);
    let inputElement: HTMLInputElement | null = $state(null);
    let dropdownPos = $state({ top: 0, left: 0 });
    let searchResults = $state<TokenSearchResult[]>([]);
    let searchTimer: ReturnType<typeof setTimeout> | null = null;

    function doSearch(query: string) {
        if (searchTimer) clearTimeout(searchTimer);
        if (!query.trim()) {
            searchResults = [];
            return;
        }
        searchTimer = setTimeout(async () => {
            try {
                searchResults = await api.searchTokens(query, promptId, position);
            } catch {
                searchResults = [];
            }
        }, 150);
    }

    function updateDropdownPosition() {
        if (!inputElement) return;
        const rect = inputElement.getBoundingClientRect();
        dropdownPos = { top: rect.bottom + 2, left: rect.left };
    }

    function handleSelect(token: TokenSearchResult) {
        onSelect(token.id, token.string);
        isOpen = false;
    }

    function handleKeydown(e: KeyboardEvent) {
        if (!isOpen || searchResults.length === 0) {
            if (e.key === "ArrowDown" && inputValue.trim()) {
                e.preventDefault();
                updateDropdownPosition();
                isOpen = true;
            }
            return;
        }

        switch (e.key) {
            case "ArrowDown":
                e.preventDefault();
                highlightedIndex = Math.min(highlightedIndex + 1, searchResults.length - 1);
                break;
            case "ArrowUp":
                e.preventDefault();
                highlightedIndex = Math.max(highlightedIndex - 1, 0);
                break;
            case "Enter":
                e.preventDefault();
                if (searchResults[highlightedIndex]) {
                    handleSelect(searchResults[highlightedIndex]);
                }
                break;
            case "Escape":
                e.preventDefault();
                isOpen = false;
                break;
        }
    }

    function handleInput(e: Event) {
        updateDropdownPosition();
        isOpen = true;
        highlightedIndex = 0;
        const target = e.target as HTMLInputElement;
        onSelect(null, target.value);
        doSearch(target.value);
    }

    function handleFocus() {
        if (inputValue.trim()) {
            updateDropdownPosition();
            isOpen = true;
            doSearch(inputValue);
        }
    }

    function handleBlur() {
        setTimeout(() => {
            isOpen = false;
        }, 150);
    }

    function formatProb(prob: number): string {
        return `${(prob * 100).toFixed(1)}%`;
    }
</script>

<div class="token-dropdown">
    <input
        bind:this={inputElement}
        type="text"
        bind:value={inputValue}
        onfocus={handleFocus}
        onblur={handleBlur}
        onkeydown={handleKeydown}
        oninput={handleInput}
        {placeholder}
        class="dropdown-input"
    />

    {#if isOpen && searchResults.length > 0}
        <ul class="dropdown-list" style="top: {dropdownPos.top}px; left: {dropdownPos.left}px;">
            {#each searchResults as token, i (token.id)}
                <li>
                    <button
                        type="button"
                        class="dropdown-item"
                        class:highlighted={i === highlightedIndex}
                        onmousedown={() => handleSelect(token)}
                        onmouseenter={() => (highlightedIndex = i)}
                    >
                        <span class="token-string">{sanitizeToken(token.string)}</span>
                        <span class="token-meta">
                            <span class="token-prob">{formatProb(token.prob)}</span>
                            <span class="token-id">#{token.id}</span>
                        </span>
                    </button>
                </li>
            {/each}
        </ul>
    {:else if isOpen && inputValue.trim() && searchResults.length === 0}
        <div class="dropdown-empty" style="top: {dropdownPos.top}px; left: {dropdownPos.left}px;">
            No matching tokens
        </div>
    {/if}
</div>

<style>
    .token-dropdown {
        position: relative;
        display: inline-block;
    }

    .dropdown-input {
        width: 120px;
        padding: var(--space-1);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .dropdown-input:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .dropdown-input::placeholder {
        color: var(--text-muted);
    }

    .dropdown-list {
        position: fixed;
        min-width: 250px;
        max-height: 300px;
        overflow-y: auto;
        margin: 0;
        padding: 0;
        list-style: none;
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-sm);
        box-shadow: var(--shadow-md);
        z-index: 10000;
    }

    .dropdown-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        padding: var(--space-2) var(--space-3);
        background: transparent;
        border: none;
        cursor: pointer;
        text-align: left;
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-size: var(--text-sm);
        gap: var(--space-3);
    }

    .dropdown-item:hover,
    .dropdown-item.highlighted {
        background: var(--bg-surface);
    }

    .token-string {
        white-space: pre;
        background: var(--bg-base);
        padding: 1px 3px;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
    }

    .token-meta {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        flex-shrink: 0;
    }

    .token-prob {
        font-size: var(--text-xs);
        color: var(--text-secondary);
        font-variant-numeric: tabular-nums;
    }

    .token-id {
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .dropdown-empty {
        position: fixed;
        min-width: 150px;
        padding: var(--space-2) var(--space-3);
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-sm);
        color: var(--text-muted);
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        z-index: 10000;
    }
</style>
