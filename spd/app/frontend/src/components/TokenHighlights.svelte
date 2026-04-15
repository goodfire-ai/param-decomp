<script lang="ts">
    import { getTokenHighlightBg, getComponentActivationColor } from "../lib/colors";
    import { sanitizeToken, getProbAtPosition as getProbAt, formatProb } from "../lib/tokenUtils";

    interface Props {
        tokenStrings: string[];
        tokenCi: number[]; // CI values (0-1)
        tokenComponentActs: number[]; // Component activations (can be negative)
        maxAbsComponentAct: number; // For normalizing component act colors
        tokenNextProbs?: (number | null)[]; // Probability of next token (optional)
    }

    let { tokenStrings, tokenCi, tokenComponentActs, maxAbsComponentAct, tokenNextProbs }: Props = $props();

    function getTooltipText(ci: number, componentAct: number, prob: number | null): string {
        let text = `CI: ${ci.toFixed(3)} | Act: ${componentAct.toFixed(3)}`;
        if (prob !== null) {
            text += ` | P: ${formatProb(prob)}`;
        }
        return text;
    }

    function getProbAtPosition(i: number): number | null {
        if (!tokenNextProbs) return null;
        return getProbAt(tokenNextProbs, i);
    }

    function getBgColor(ci: number): string {
        return getTokenHighlightBg(ci);
    }

    function getUnderlineColor(componentAct: number): string {
        const normalizedAbs = Math.abs(componentAct) / maxAbsComponentAct;
        return getComponentActivationColor(componentAct, normalizedAbs);
    }
</script>

<span class="token-highlights"
    >{#each tokenStrings as tok, i (i)}<span
            class="token-highlight"
            style="background-color:{getBgColor(tokenCi[i])};--underline-color:{getUnderlineColor(
                tokenComponentActs[i],
            )}"
            data-tooltip={getTooltipText(tokenCi[i], tokenComponentActs[i], getProbAtPosition(i))}
            >{sanitizeToken(tok)}</span
        >{/each}</span
>

<style>
    .token-highlights {
        display: inline;
        white-space: pre;
        font-family: var(--font-mono);
    }

    .token-highlight {
        display: inline;
        padding: var(--space-1) 0 0 0;
        margin-right: 1px;
        margin-bottom: var(--space-1);
        border-right: 1px solid var(--border-subtle);
        position: relative;
        white-space: pre;
    }

    .token-highlight::before {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        bottom: -3px;
        height: 3px;
        background-color: var(--underline-color, transparent);
    }

    .token-highlight::after {
        content: attr(data-tooltip);
        position: absolute;
        top: calc(100% + 8px);
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

    .token-highlight:hover::after {
        opacity: 1;
    }
</style>
