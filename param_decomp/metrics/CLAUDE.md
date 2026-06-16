# `param_decomp/metrics/`

Loss `Metric` classes plus the dispatch wiring that turns a `PDConfig.loss_metrics` YAML
entry into a bound, runnable `Metric` instance.

Loss metrics are **canonical and curated** — adding one is a deliberate change to the
core library. For eval metrics (user-extensible, lab-side), see
[`param_decomp_lab/eval_metrics/CLAUDE.md`](../../param_decomp_lab/eval_metrics/CLAUDE.md).

## File map

| File | Purpose |
|---|---|
| `base.py` | `Metric` ABC (lifecycle: `__init__(cfg)` → `bind` → `update` → `compute`) + `before_backward` / `after_backward` hooks |
| `context.py` | `MetricContext` — the per-step bundle every `Metric.update(ctx)` receives; `ctx.model` is a `ComponentModelProtocol` (core `ComponentModel`, FSDP adapter, or vendored `LMComponentModel`) |
| `dispatch.py` | `LOSS_METRIC_CLASSES` type→class table + `instantiate_metrics(...)` |
| `<loss_name>.py` | One file per metric: the `<Name>Loss` class; its `<Name>LossConfig` lives in `param_decomp_config/losses.py` |
| `persistent_pgd_state.py` | PPGD adversarial-source state machine (shared by `persistent_pgd_recon.py`) |
| `pgd_utils.py` | Shared PGD helpers used by the regular PGD recon metrics |
| `output.py` | Shared output-extraction helpers used across recon losses |

## Adding a loss metric

1. Define `<Name>LossConfig(LossMetricConfig)` in `param_decomp_config/losses.py`. The
   config must carry a unique `type: Literal["<Name>Loss"]` discriminator.
2. Define `<Name>Loss(Metric[<Name>LossConfig])` in `<name>.py`, importing the config
   from `param_decomp_config.losses`.
3. Append the config to `AnyLossMetricConfig` in `param_decomp_config/pd.py`.
4. Append the class to `LOSS_METRIC_CLASSES` in `dispatch.py`.

The pydantic discriminated union validates `pd.loss_metrics` YAML entries without any
custom validator. `instantiate_loss_metrics()` builds and `bind()`s one instance per
entry. Duplicate `type` literals in a single config are rejected.

A metric that wants to manipulate state coupled to backward overrides `before_backward`
and/or `after_backward` (see PPGD for the canonical example).

## Metric identity (`instance_key`) and same-class loss + eval

Metric instances are keyed everywhere — instance dicts, state-dict, and log-key
suffixes — by `Metric.instance_key`, which defaults to the class name. A loss-capable
config can override it by setting `name` (on `LossMetricConfig`). This is what lets the
*same* metric class appear under both `pd.loss_metrics` and `eval.metrics`: without a
distinct `name` their instance keys collide and `instantiate_metrics` rejects the
overlap. Example — a 1-step PGD training loss plus a 20-step PGD eval probe:

```yaml
pd:
  loss_metrics:
    - type: PGDReconLoss        # instance_key "PGDReconLoss", auto-evaluated too
      coeff: 0.5
      n_steps: 1
eval:
  metrics:
    - type: PGDReconLoss        # distinct instance_key -> no collision
      name: PGDReconLoss_20step
      n_steps: 20
```

`name` disambiguates scalar-output metrics (the log key is `{log_namespace}/{instance_key}`).
A dict-returning metric flattens under its own internal keys, so two dict-returning
instances of one class would still collide — namespace their keys if you need that.

A `compute()` dict key that already contains a `/` keeps its own namespace (skips the
`log_namespace` prefix), letting a metric emit keys outside its headline namespace — the
imp-min metrics put both their eval headline and `_no_beta` proxy under `imp_min/`, so the
pair groups together off the `loss/` panel (the live training loss still logs under
`loss/`, so the eval headline no longer overlays the train headline on one chart).

## Importance-minimality variants

Two interchangeable CI-sparsity penalties, same `(sum + beta·entropy)` shape and the
same exact-DDP reduction (the additive split `lp_and_entropy_terms` / `finalize_imp_min`
is penalty-agnostic and shared):

- `ImportanceMinimalityLoss` (`importance_minimality.py`) — `L_p`: `(c + eps)^p` with `p`
  annealed toward 0. Gradient `p·c^(p-1)` blows up as `c→0` for `p<1`.
- `SmoothL0ImportanceMinimalityLoss` (`smooth_l0_importance_minimality.py`) — bounded
  Geman–McClure `c²/(c²+γ²)`: flat at 0, saturating to 1, bounded gradient `~0.65/γ` near
  `c≈γ`. Drop-in replacement avoiding the `L_p` cliff; `γ` anneals like `p`.

Both are registered eval-side too (`EVAL_METRIC_CLASSES` + `AnyEvalMetricConfig`), so a run
driven by one can log the other as an eval-only sparsity proxy. The directly comparable
cross-run sparsity number is `CI_L0` (a hard count); each penalty's own `_no_beta` proxy is
on its own scale.

## Config placement rule

Every config class lives in the torch-free `param_decomp_config` package — loss-metric
configs in `param_decomp_config/losses.py`, the union in `param_decomp_config/pd.py`.
Implementation modules import their config from there; never define a pydantic config
in `param_decomp/` or add torch imports to `param_decomp_config/`.

Never use `if TYPE_CHECKING:` + forward-reference strings to paper over a cycle. If
you're reaching for that, the config placement is wrong; move the config instead.

## Sources vs masks (PGD terminology)

These two concepts both show up in the PGD metrics and are easy to confuse:

- **Sources** (`adv_sources`, `PPGDSources`, `self.sources`) — the raw values PGD
  optimizes adversarially. They get interpolated with CI to produce component masks:
  `mask = ci + (1 - ci) * source`. Used in `pgd_utils.py` (regular PGD) and
  `persistent_pgd_state.py` (PPGD).
- **Masks** (`component_masks`, `RoutingMasks`, `make_mask_infos`, `n_mask_samples`) —
  the materialized per-component masks consumed by forward passes. Produced from
  sources (in PGD) or from stochastic sampling (otherwise). This is the general PD
  concept — sources are a PGD-internal stepping stone.

## PPGD note

PPGD's state machine lives in `persistent_pgd_state.py` (shared); its `Metric`
classes + configs live in `persistent_pgd_recon.py`. The split is so the subset
variant (`PersistentPGDReconSubsetLoss`) can reuse the same state machine.
