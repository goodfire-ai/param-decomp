<script lang="ts">
    import type { OptimizationResult } from "../../lib/promptAttributionsTypes";

    type Props = {
        optimization: OptimizationResult;
        tokens: string[];
    };

    let { optimization, tokens }: Props = $props();

    const tokenAtPos = $derived(
        optimization.loss.position >= 0 && optimization.loss.position < tokens.length
            ? tokens[optimization.loss.position]
            : null,
    );

    const formatProb = (prob: number | null): string => {
        if (prob === null) return "-";
        return prob.toFixed(3);
    };
</script>

<div class="opt-params">
    <span class="param"><span class="key">steps</span>{optimization.steps}</span>
    <span class="param"><span class="key">imp_min</span>{optimization.imp_min_coeff}</span>
    <span class="param"><span class="key">pnorm</span>{optimization.pnorm}</span>
    <span class="param"><span class="key">beta</span>{optimization.beta}</span>
    <span class="param"><span class="key">mask</span>{optimization.mask_type}</span>
    <span class="param">
        <span class="key">{optimization.loss.type}</span>{optimization.loss.coeff}
    </span>
    <span class="param">
        <span class="key">pos</span>
        {optimization.loss.position}
        {#if tokenAtPos !== null}
            (<span class="token">{tokenAtPos}</span>)
        {/if}
    </span>
    {#if optimization.loss.type === "ce"}
        <span class="param">
            <span class="key">label</span>(<span class="token">{optimization.loss.label_str}</span>)
        </span>
    {/if}
    {#if optimization.pgd}
        <span class="param">
            <span class="key">pgd_steps</span>{optimization.pgd.n_steps}
        </span>
        <span class="param">
            <span class="key">pgd_lr</span>{optimization.pgd.step_size}
        </span>
    {/if}
    <span class="divider"></span>
    <span class="param metric">
        <span class="key">L0</span>{optimization.metrics.l0_total.toFixed(1)}
    </span>
    {#if optimization.loss.type === "ce"}
        <span class="param metric">
            <span class="key">CI prob</span>{formatProb(optimization.metrics.ci_masked_label_prob)}
        </span>
        <span class="param metric">
            <span class="key">stoch prob</span>{formatProb(optimization.metrics.stoch_masked_label_prob)}
        </span>
        <span class="param metric">
            <span class="key">adv prob</span>{formatProb(optimization.metrics.adv_pgd_label_prob)}
        </span>
    {/if}
</div>

<style>
    .opt-params {
        display: flex;
        flex-wrap: wrap;
        gap: var(--space-4);
        align-items: center;
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        color: var(--text-secondary);
    }

    .param {
        display: flex;
        align-items: center;
        gap: var(--space-1);
    }

    .key {
        color: var(--text-muted);
    }

    .key::after {
        content: ":";
    }

    .token {
        white-space: pre;
        font-family: var(--font-mono);
        background: var(--bg-inset);
        padding: 0 var(--space-1);
        border-radius: var(--radius-sm);
    }

    .divider {
        width: 1px;
        height: 12px;
        background: var(--border-secondary);
        margin: 0 var(--space-2);
    }

    .metric {
        color: var(--text-primary);
    }
</style>
