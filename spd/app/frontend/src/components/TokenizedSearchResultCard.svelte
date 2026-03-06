<script lang="ts">
    import { SvelteSet } from "svelte/reactivity";
    import type { TokenizedSearchResult } from "../lib/api/dataset";
    import { getNextTokenProbBgColor } from "../lib/colors";
    import { getProbAtPosition, formatProb } from "../lib/tokenUtils";
    import TokenSpan from "./ui/TokenSpan.svelte";

    interface Props {
        result: TokenizedSearchResult;
        index: number;
        query: string;
    }

    let { result, index, query }: Props = $props();

    const matchPositions = $derived.by(() => {
        if (!query) return new SvelteSet<number>();

        const positions = new SvelteSet<number>();
        const lowerQuery = query.toLowerCase();

        let text = "";
        const tokenStarts: number[] = [];
        const tokenEnds: number[] = [];

        for (const tok of result.tokens) {
            tokenStarts.push(text.length);
            text += tok;
            tokenEnds.push(text.length);
        }

        const lowerText = text.toLowerCase();

        let searchStart = 0;
        while (true) {
            const matchStart = lowerText.indexOf(lowerQuery, searchStart);
            if (matchStart === -1) break;
            const matchEnd = matchStart + lowerQuery.length;

            for (let i = 0; i < result.tokens.length; i++) {
                if (tokenStarts[i] < matchEnd && tokenEnds[i] > matchStart) {
                    positions.add(i);
                }
            }

            searchStart = matchStart + 1;
        }

        return positions;
    });
</script>

<div class="result-card">
    <div class="result-header">
        <span class="result-index">#{index + 1}</span>
        {#if result.occurrence_count > 0}
            <span class="occurrence-badge"
                >{result.occurrence_count} occurrence{result.occurrence_count !== 1 ? "s" : ""}</span
            >
        {/if}
        {#each Object.entries(result.metadata) as [metaKey, metaVal] (metaKey)}
            <span class="tag">{metaVal}</span>
        {/each}
    </div>
    <div class="tokens-container">
        <span class="prob-tokens"
            >{#each result.tokens as tok, i (i)}{@const prob = getProbAtPosition(result.next_token_probs, i)}<span
                    class="prob-token-wrapper"
                    class:match={matchPositions.has(i)}
                    ><TokenSpan
                        token={tok}
                        backgroundColor={getNextTokenProbBgColor(prob)}
                        tooltip={prob !== null ? `P: ${formatProb(prob)}` : ""}
                    /></span
                >{/each}</span
        >
    </div>
</div>

<style>
    .result-card {
        padding: var(--space-3);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .result-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        flex-wrap: wrap;
    }

    .result-index {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        font-weight: 600;
    }

    .occurrence-badge {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--accent-primary);
        background: color-mix(in srgb, var(--accent-primary) 10%, transparent);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
    }

    .tag {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
        background: var(--bg-inset);
        color: var(--text-secondary);
    }

    .tokens-container {
        font-size: var(--text-sm);
        line-height: 1.6;
    }

    .prob-tokens {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 1px;
    }

    .prob-token-wrapper {
        border: 1px solid transparent;
        padding: 1px 0;
    }

    .prob-token-wrapper.match {
        border-color: var(--accent-primary);
        border-radius: 2px;
    }
</style>
