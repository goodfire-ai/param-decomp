<script lang="ts">
    import { getNextTokenProbBgColor } from "../lib/colors";
    import { getProbAtPosition, formatProb } from "../lib/tokenUtils";
    import TokenSpan from "./ui/TokenSpan.svelte";

    interface Props {
        tokens: string[];
        nextTokenProbs: (number | null)[];
    }

    let { tokens, nextTokenProbs }: Props = $props();
</script>

<span class="prob-tokens"
    >{#each tokens as tok, i (i)}{@const prob = getProbAtPosition(nextTokenProbs, i)}<span class="prob-token-wrapper"
            ><TokenSpan
                token={tok}
                backgroundColor={getNextTokenProbBgColor(prob)}
                tooltip={prob !== null ? `P: ${formatProb(prob)}` : ""}
            /></span
        >{/each}</span
>

<style>
    .prob-tokens {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 1px;
    }

    .prob-token-wrapper {
        border-right: 1px solid var(--border-subtle);
        padding: 1px 0;
    }
</style>
