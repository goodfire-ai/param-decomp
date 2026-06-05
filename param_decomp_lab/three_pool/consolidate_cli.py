"""Manual CPU-only consolidation of a 3-pool run's scratch partials.

Recovery tool for a failed/preempted async consolidation. The train loop always
leaves a step's partials on disk until that step consolidates successfully, so a
failed consolidation is fixed by re-running this:

    python -m param_decomp_lab.three_pool.consolidate_cli p-xxxxxxxx
    python -m param_decomp_lab.three_pool.consolidate_cli p-xxxxxxxx --step 5000

With no ``--step`` it consolidates every step that has partials but no
``training_<step>.pth`` yet (see `unconsolidated_steps`).

Separate module from `consolidate` (the pure assembly logic) so the
`experiments.lm` config imports here stay out of `consolidate`, which `optimize`
imports on the train-loop side.
"""

from pathlib import Path

import fire
import yaml

from param_decomp.log import logger
from param_decomp_lab.experiments.lm.run import build_target, make_run_batch
from param_decomp_lab.experiments.lm.three_pool_run import ThreePoolLMExperimentConfig
from param_decomp_lab.experiments.lm.two_pool_run import TwoPoolLMExperimentConfig
from param_decomp_lab.experiments.utils import EXPERIMENT_CONFIG_FILENAME
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.three_pool.consolidate import (
    DEFAULT_KEEP_LAST_N_TRAINING,
    SNAPSHOT_SCRATCH_DIRNAME,
    consolidate_step,
    unconsolidated_steps,
)


def cli(
    run: str,
    *,
    step: int | None = None,
    keep_last_n_training: int = DEFAULT_KEEP_LAST_N_TRAINING,
) -> None:
    out_dir = Path(run) if Path(run).is_dir() else PARAM_DECOMP_OUT_DIR / "runs" / run
    cfg_path = out_dir / EXPERIMENT_CONFIG_FILENAME
    # The 2-pool and 3-pool configs differ only in runtime.topology (pool_a vs ci/ppgd);
    # consolidation reads only pool-agnostic fields. Pick the class from the topology keys.
    topology = yaml.safe_load(cfg_path.read_text())["runtime"]["topology"]
    cfg_cls = TwoPoolLMExperimentConfig if "pool_a" in topology else ThreePoolLMExperimentConfig
    cfg = cfg_cls.from_file(cfg_path)
    target_model = build_target(cfg.target)
    run_batch = make_run_batch(cfg.target)

    steps = [step] if step is not None else unconsolidated_steps(out_dir)
    assert steps, f"nothing to consolidate under {out_dir} (no unconsolidated scratch steps)"
    logger.info(f"consolidate CLI: steps {steps} under {out_dir}")
    for s in steps:
        consolidate_step(
            scratch_dir=out_dir / SNAPSHOT_SCRATCH_DIRNAME,
            out_dir=out_dir,
            step=s,
            target_model=target_model,
            run_batch=run_batch,
            ci_config=cfg.pd.ci_config,
            sigmoid_type=cfg.pd.sigmoid_type,
            keep_last_n_training=keep_last_n_training,
        )


if __name__ == "__main__":
    fire.Fire(cli)
