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
    <span class="param" title="Number of optimization steps used to find this sparse circuit"
        ><span class="key">steps</span>{optimization.steps}</span
    >
    <span
        class="param"
        title="Importance minimality coefficient — controls how aggressively L0 (active component count) is penalized"
        ><span class="key">imp_min</span>{optimization.imp_min_coeff}</span
    >
    <span class="param" title="p-norm exponent for L0 approximation — lower values approximate hard L0 more closely"
        ><span class="key">pnorm</span>{optimization.pnorm}</span
    >
    <span class="param" title="Importance minimality β — temperature for the soft-min penalty on component CIs"
        ><span class="key">beta</span>{optimization.beta}</span
    >
    <span
        class="param"
        title="Mask type used during optimization (stochastic = Bernoulli sampling, ci = deterministic CI values)"
        ><span class="key">mask</span>{optimization.mask_type}</span
    >
    <span
        class="param"
        title={optimization.loss.type === "ce"
            ? "Cross-entropy loss coefficient — optimizes P(label_token) at the target position"
            : optimization.loss.type === "logit"
              ? "Logit loss coefficient — maximizes raw pre-softmax logit for label_token at the target position"
              : "KL divergence loss coefficient — optimizes match to target model's distribution at the target position"}
    >
        <span class="key">{optimization.loss.type}</span>{optimization.loss.coeff}
    </span>
    <span class="param" title="Sequence position being optimized (0-indexed)">
        <span class="key">pos</span>
        {optimization.loss.position}
        {#if tokenAtPos !== null}
            (<span class="token">{tokenAtPos}</span>)
        {/if}
    </span>
    {#if optimization.loss.type === "ce" || optimization.loss.type === "logit"}
        <span
            class="param"
            title="The target token whose {optimization.loss.type === 'logit'
                ? 'logit is being maximized'
                : 'probability is being maximized'}"
        >
            <span class="key">label</span>(<span class="token">{optimization.loss.label_str}</span>)
        </span>
    {/if}
    {#if optimization.pgd}
        <span
            class="param"
            title="Number of PGD (projected gradient descent) steps used during training to find adversarial counterexamples"
        >
            <span class="key">pgd_steps</span>{optimization.pgd.n_steps}
        </span>
        <span class="param" title="Step size for training PGD adversarial optimization">
            <span class="key">pgd_lr</span>{optimization.pgd.step_size}
        </span>
    {/if}
    <span class="divider"></span>
    <span class="param metric" title="Total number of active components (CI > 0) in this sparse circuit">
        <span class="key">L0</span>{optimization.metrics.l0_total.toFixed(1)}
    </span>
    {#if optimization.loss.type === "ce" || optimization.loss.type === "logit"}
        <span
            class="param metric"
            title="P(label) when only selected components are active (all others zeroed) — measures circuit sufficiency"
        >
            <span class="key">CI prob</span>{formatProb(optimization.metrics.ci_masked_label_prob)}
        </span>
        <span
            class="param metric"
            title="P(label) when unselected components get random masks — tests robustness to noise"
        >
            <span class="key">stoch prob</span>{formatProb(optimization.metrics.stoch_masked_label_prob)}
        </span>
        <span
            class="param metric"
            title="P(label) when unselected components are adversarially optimized to suppress the label — worst-case circuit sufficiency"
        >
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
