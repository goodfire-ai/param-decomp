"""Generate the L18-23 6-layer MLP VPD ablation-sweep configs.

Reads the baked-default config (`configs/llama8b_l18-23_6layer_ablation_base.yaml`) and
emits one self-contained run yaml per ablation point into `configs/l18-23_ablations/`.
Ablations are MULTIPLICATIVE factors around the baked default (the centre):

  - learning rate   x[0.31]        (scales BOTH components & ci_fn optimizer start_val.
                                    The x3.1 up-factor is dropped: higher LR doesn't train
                                    here — Lucius, 2026-06-19.)
  - tokens/batch    x[2, 4]        (scales pd.batch_size; seq_len held at 512. NB: the
                                    larger batches need a GPU-count / dp-layout change at
                                    launch — the team handles that, it is not in the yaml.)
  - imp-min coeff   x[3.1, 10]     (scales ImportanceMinimalityLoss.coeff; want it higher
                                    than the base, which was badly non-minimal.)

NOT auto-generated: Dan's new imp-min loss *form* (a different functional form, not a coeff
change). Its config shape isn't pinned yet — add it here as its own axis once it lands.

Every emitted config is validated against the torch-free `LMExperimentConfig` schema before
writing, so a broken variant fails loudly here rather than at launch. Idempotent: rewrites
the output dir each run.

  python -m jax_single_pool.tools.gen_l18_23_ablations
"""

import copy
from math import floor, log10
from pathlib import Path
from typing import Any

import yaml

from param_decomp_config.lm import LMExperimentConfig

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
BASE_CONFIG = CONFIGS_DIR / "llama8b_l18-23_6layer_ablation_base.yaml"
OUT_DIR = CONFIGS_DIR / "l18-23_ablations"

LR_FACTORS = [0.31]
BATCH_FACTORS = [2, 4]
IMPMIN_FACTORS = [3.1, 10]


def _round_sig(x: float, sig: int = 4) -> float:
    if x == 0:
        return 0.0
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


def _loss_metric(cfg: dict[str, Any], loss_type: str) -> dict[str, Any]:
    matches = [m for m in cfg["pd"]["loss_metrics"] if m["type"] == loss_type]
    assert len(matches) == 1, f"expected exactly one {loss_type}, found {len(matches)}"
    return matches[0]


def _scaled_lr(base: dict[str, Any], factor: float) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    for opt in ("components_optimizer", "ci_fn_optimizer"):
        sched = cfg["pd"][opt]["lr_schedule"]
        sched["start_val"] = _round_sig(sched["start_val"] * factor)
    return cfg


def _scaled_batch(base: dict[str, Any], factor: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["pd"]["batch_size"] = cfg["pd"]["batch_size"] * factor
    return cfg


def _scaled_impmin(base: dict[str, Any], factor: float) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    impmin = _loss_metric(cfg, "ImportanceMinimalityLoss")
    impmin["coeff"] = _round_sig(impmin["coeff"] * factor)
    return cfg


def _factor_tag(factor: float) -> str:
    return (f"{factor:g}").replace(".", "p")


def build_variants(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {"ablbase": copy.deepcopy(base)}
    for f in LR_FACTORS:
        variants[f"lr_x{_factor_tag(f)}"] = _scaled_lr(base, f)
    for f in BATCH_FACTORS:
        variants[f"batch_x{_factor_tag(f)}"] = _scaled_batch(base, f)
    for f in IMPMIN_FACTORS:
        variants[f"impmin_x{_factor_tag(f)}"] = _scaled_impmin(base, f)
    return variants


def main() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    variants = build_variants(base)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("llama8b_l18-23_6layer_*.yaml"):
        path.unlink()

    for tag, cfg in variants.items():
        cfg["run_name"] = f"jax-l18-23-6L-{tag}"
        LMExperimentConfig.model_validate(cfg)  # fail loudly on an invalid variant
        out_path = OUT_DIR / f"llama8b_l18-23_6layer_{tag}.yaml"
        out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote {out_path.relative_to(CONFIGS_DIR.parent)}  ({cfg['run_name']})")


if __name__ == "__main__":
    main()
