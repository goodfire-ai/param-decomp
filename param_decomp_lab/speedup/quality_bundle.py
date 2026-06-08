"""The frozen quality-bundle "ruler" for Track-2 (method speedups).

A change must report this whole *vector* vs the locked baseline so it can't hide a
regression in one metric by improving another. This file is the ruler, not the thing
being measured: an experiment must NOT edit it (see track2/README.md §read-only). Adding
or removing a bundle metric is a deliberate, separately-flagged change.

Keys are the full strings as they land in `metrics.jsonl` / wandb:
`eval/{log_namespace}/{key}` (see `param_decomp/optimize.py` eval logging and
`param_decomp/metrics/output.py`). The `l0` key embeds the metric's rounding threshold
(`{threshold}_{group}`); the canonical configs use threshold `0.0` and a `total` group.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BundleMetric:
    key: str
    label: str
    lower_is_better: bool


# Numeric guardrails, compared as a vector against the baseline.
QUALITY_BUNDLE: list[BundleMetric] = [
    BundleMetric("eval/ce_kl/ce_difference_ci_masked", "CE diff (CI-masked)", True),
    BundleMetric("eval/ce_kl/ce_unrecovered_ci_masked", "CE unrecovered (CI-masked)", True),
    BundleMetric("eval/ce_kl/kl_ci_masked", "KL (CI-masked)", True),
    BundleMetric("eval/l0/0.0_total", "L0 total (sparsity)", True),
    BundleMetric("eval/loss/StochasticHiddenActsReconLoss", "Stochastic hidden-acts recon", True),
    BundleMetric("eval/loss/PGDReconLoss", "PGD recon", True),
]

# Visual-only guardrails — figures, not scalars, so they aren't in the numeric diff.
# An agent eyeballs these in wandb; they don't gate promotion on their own.
VISUAL_GUARDRAILS: list[str] = [
    "eval/figures/component_activation_density",
    "eval/figures/ci_mean_per_component",
]
