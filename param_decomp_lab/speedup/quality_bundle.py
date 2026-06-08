"""The frozen quality-bundle "ruler" for Track-2 (method speedups).

A change must report this whole *vector* vs the locked baseline so it can't hide a
regression in one metric by improving another. This file is the ruler, not the thing
being measured: an experiment must NOT edit it (see track2/README.md §read-only). Adding
or removing a bundle metric is a deliberate, separately-flagged change.

What we're actually looking for (tiers), see also plan_t1.md:
  - `primary`   — the objective. A speedup must hold these vs baseline: PPGD recon loss
                  and the beta-independent importance-minimality term (sparsity). These
                  dominate promote/kill calls.
  - `secondary` — weighted less: stochastic-mask recon, CI-mask recon, and L0.
  - `guardrail` — faithfulness (CE/KL under the CI mask) must not blow up; sanity, not
                  the thing we optimize.

Sparsity note: the *primary* sparsity metric is `ImportanceMinimalityLoss/no_beta` — the
pure L_p importance-minimality term with `beta=0`, the actual penalty the optimizer drives
down. It stays comparable across experiments that tune `beta` (unlike the headline,
beta-weighted `ImportanceMinimalityLoss`) and is smooth (unlike L0, which is non-monotone
early — see plan_t1.md). L0 is kept as a secondary, informational read only.

PPGD note: the *primary* recon metric is the eval-time `PGDReconLoss` (a fresh PGD attack
on the masked reconstruction), NOT the train-time `PersistentPGDReconLoss`. The eval metric
is the same procedure regardless of what an experiment changes about training-time PPGD, so
it stays comparable across exactly the speedups this track targets.

Keys are the full strings as they land in `metrics.jsonl` / wandb:
`eval/{log_namespace}/{key}` (see `param_decomp/optimize.py` eval logging and
`param_decomp/metrics/output.py`). The `l0` key embeds the metric's rounding threshold
(`{threshold}_{group}`); the canonical configs use threshold `0.0` and a `total` group.
Several of these are *slow* eval metrics (logged only on `slow_every`), so a same-step
comparison should land on a slow-eval step — see `compare_runs.py --at_step`.
"""

from dataclasses import dataclass
from typing import Literal

Tier = Literal["primary", "secondary", "guardrail"]


@dataclass(frozen=True)
class BundleMetric:
    key: str
    label: str
    lower_is_better: bool
    tier: Tier


QUALITY_BUNDLE: list[BundleMetric] = [
    # --- primary: the objective ---
    BundleMetric("eval/loss/PGDReconLoss", "PGD recon (PPGD)", True, "primary"),
    BundleMetric(
        "eval/loss/ImportanceMinimalityLoss/no_beta",
        "Importance-minimality (no_beta)",
        True,
        "primary",
    ),
    # --- secondary: weighted less ---
    BundleMetric(
        "eval/loss/StochasticHiddenActsReconLoss", "Stochastic hidden-acts recon", True, "secondary"
    ),
    BundleMetric(
        "eval/loss/CIHiddenActsReconLoss", "CI-masked hidden-acts recon", True, "secondary"
    ),
    BundleMetric("eval/l0/0.0_total", "L0 total (informational)", True, "secondary"),
    # --- guardrail: faithfulness must not blow up ---
    BundleMetric("eval/ce_kl/ce_difference_ci_masked", "CE diff (CI-masked)", True, "guardrail"),
    BundleMetric("eval/ce_kl/kl_ci_masked", "KL (CI-masked)", True, "guardrail"),
    BundleMetric(
        "eval/ce_kl/ce_unrecovered_ci_masked", "CE unrecovered (CI-masked)", True, "guardrail"
    ),
]

# Visual-only guardrails — figures, not scalars, so they aren't in the numeric diff.
# An agent eyeballs these in wandb; they don't gate promotion on their own.
VISUAL_GUARDRAILS: list[str] = [
    "eval/figures/component_activation_density",
    "eval/figures/ci_mean_per_component",
]
