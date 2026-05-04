<script lang="ts">
    import type { TokenizedSample } from "../lib/api";
    import ProbColoredTokens from "./ProbColoredTokens.svelte";

    interface Props {
        sample: TokenizedSample;
        index: number;
    }

    let { sample, index }: Props = $props();
</script>

<div class="sample-card">
    <div class="sample-header">
        <span class="sample-index">#{index + 1}</span>
        {#each Object.entries(sample.metadata) as [metaKey, metaVal] (metaKey)}
            <span class="tag">{metaVal}</span>
        {/each}
    </div>
    <div class="tokens-container">
        <ProbColoredTokens tokens={sample.tokens} nextTokenProbs={sample.next_token_probs} />
    </div>
</div>

<style>
    .sample-card {
        padding: var(--space-3);
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .sample-header {
        display: flex;
        align-items: center;
        gap: var(--space-2);
        flex-wrap: wrap;
    }

    .sample-index {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--text-muted);
        font-weight: 600;
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
</style>
