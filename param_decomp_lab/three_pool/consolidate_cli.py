"""Manual CPU-only consolidation of a 3-pool run's scratch partials.

Recovery tool for a failed/preempted async consolidation. The train loop always
leaves a step's partials on disk until that step consolidates successfully, so a
failed consolidation is fixed by re-running this:

    python -m param_decomp_lab.three_pool.consolidate_cli p-xxxxxxxx
    python -m param_decomp_lab.three_pool.consolidate_cli p-xxxxxxxx --step 5000

With no ``--step`` it consolidates every step that has partials but no
``training_<step>.pth`` yet (see `unconsolidated_steps`).

Separate module from `consolidate` so the `experiments.lm.run` import here
doesn't form an import cycle (`run` imports `consolidate`).
"""

from pathlib import Path

import fire

from param_decomp.log import logger
from param_decomp_lab.experiments.lm.run import (
    LMExperimentConfig,
    build_target,
    make_run_batch,
)
from param_decomp_lab.experiments.utils import RUN_META_FILENAME
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
    out_dir = Path(run) if Path(run).is_dir() else PARAM_DECOMP_OUT_DIR / "decompositions" / run
    cfg = LMExperimentConfig.from_file(out_dir / RUN_META_FILENAME)
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
