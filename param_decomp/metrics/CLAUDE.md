# `param_decomp/metrics/`

Loss `Metric` classes plus the dispatch wiring that turns a `PDConfig.loss_metrics` YAML
entry into a bound, runnable `Metric` instance.

Loss metrics are **canonical and curated** — adding one is a deliberate change to the
core library. For eval metrics (user-extensible, lab-side), see
[`param_decomp_lab/eval_metrics/CLAUDE.md`](../../param_decomp_lab/eval_metrics/CLAUDE.md).

## File map

| File | Purpose |
|---|---|
| `base.py` | `Metric` ABC (lifecycle: `__init__(cfg)` → `bind` → `update` → `compute`) + `LossMetricConfig` base + `before_backward` / `after_backward` hooks |
| `context.py` | `MetricContext` — the per-step bundle every `Metric.update(ctx)` receives |
| `dispatch.py` | `LOSS_METRIC_CLASSES` type→class table + `instantiate_metrics(...)` |
| `<loss_name>.py` | One file per metric: `<Name>Loss` class + `<Name>LossConfig` config side-by-side |
| `persistent_pgd_state.py` | PPGD adversarial-source state machine (shared by `persistent_pgd_recon.py`) |
| `pgd_utils.py` | Shared PGD helpers used by the regular PGD recon metrics |
| `output.py` | Shared output-extraction helpers used across recon losses |

## Adding a loss metric

1. Define `<Name>Loss(Metric[<Name>LossConfig])` and its `<Name>LossConfig(LossMetricConfig)`
   in `<name>.py`. The config must carry a unique `type: Literal["<Name>Loss"]` discriminator.
2. Append the config to `AnyLossMetricConfig` in `param_decomp/configs.py`.
3. Append the class to `LOSS_METRIC_CLASSES` in `dispatch.py`.

The pydantic discriminated union validates `pd.loss_metrics` YAML entries without any
custom validator. `instantiate_loss_metrics()` builds and `bind()`s one instance per
entry. Duplicate `type` literals in a single config are rejected.

A metric that wants to manipulate state coupled to backward overrides `before_backward`
and/or `after_backward` (see PPGD for the canonical example).

## Config placement rule

The default home for a config is `param_decomp/configs.py`. Move a config next to its
implementation only when leaving it in `configs.py` would close an import cycle —
concretely, when the implementation module `M` is also (transitively) imported by
`configs.py` (usually via the metric union). Then `M → configs` would loop; put the
config in `M` and update callers to import it from `M` directly.

Configs currently kept next to their implementation for this reason:

- `ScheduleConfig` → `param_decomp.schedule`
- `DecompositionTargetConfig` → `param_decomp.decomposition_targets`
- `CiConfig` family (`LayerwiseCiConfig`, `AttnConfig`, `GlobalSharedTransformerCiConfig`,
  `GlobalCiConfig`) → `param_decomp.ci_fns`
- `SamplingType`, `SubsetRoutingType` + members → `param_decomp.masks`
- Each loss metric's `LossMetricConfig` subclass → `param_decomp/metrics/<name>.py`

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
