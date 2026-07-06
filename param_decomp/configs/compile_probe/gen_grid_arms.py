"""Generate the compile-time scaling-grid arm configs (btdr, 1 node).

Derived from llama8b_full32L_1node_b8_dp8_PROFILE.yaml (full 32L / 224 sites / dp8 /
b8 / seq512 / 12 steps / eval never fires), one mutation per arm, so the grid
attributes jit_step compile time to specific graph sections:

    base4    the production shape: 4 recon chunks + PPGD(nw2) + faith
    c1/c2/c8 recon chunk count 1/2/8 (sites_per_chunk 224/112/28)
    nw0      PPGD n_warmup_steps=0 (drops the warmup-ascent scan, keeps the ppgd forward)
    noppgd   PersistentPGDReconLoss removed entirely
    nofaith  FaithfulnessLoss removed (also dodges the 38GB faith transient -> steps run)
    passes   base4 + TF_CPP hlo_pass_pipeline timing (per-XLA-pass durations in the log)

Every arm exports JAX_LOG_COMPILES=1 (per-jit compile durations in the slurm log).
Launch each arm with an ISOLATED submit-time out dir so arms never share an XLA cache:

    PARAM_DECOMP_OUT_DIR=$SCRATCH/<arm> pd-lm <arm>.yaml

with SCRATCH=/mnt/delicate-frog/artifacts/mechanisms/param-decomp/compile_probe_scratch.
"""

import sys
from pathlib import Path
from typing import Any

import yaml

BASE_CONFIG = Path(__file__).parent.parent / "llama8b_full32L_1node_b8_dp8_PROFILE.yaml"
BTDR_DATA = (
    "/mnt/delicate-frog/artifacts/mechanisms/param-decomp/datasets/fineweb_llama_tok_512/*.parquet"
)

PASS_TIMING_ENV = {"TF_CPP_MIN_LOG_LEVEL": "0", "TF_CPP_VMODULE": "hlo_pass_pipeline=1"}


def loss_metric(cfg: dict[str, Any], type_name: str) -> dict[str, Any]:
    (m,) = [m for m in cfg["pd"]["loss_metrics"] if m["type"] == type_name]
    return m


def drop_loss_metric(cfg: dict[str, Any], type_name: str) -> None:
    kept = [m for m in cfg["pd"]["loss_metrics"] if m["type"] != type_name]
    assert len(kept) == len(cfg["pd"]["loss_metrics"]) - 1, type_name
    cfg["pd"]["loss_metrics"] = kept


def set_chunks(cfg: dict[str, Any], sites_per_chunk: int) -> None:
    loss_metric(cfg, "ChunkwiseSubsetReconLoss")["sites_per_chunk"] = sites_per_chunk


ARMS: dict[str, Any] = {
    "base4": lambda cfg: None,
    "c1": lambda cfg: set_chunks(cfg, 224),
    "c2": lambda cfg: set_chunks(cfg, 112),
    "c8": lambda cfg: set_chunks(cfg, 28),
    "nw0": lambda cfg: loss_metric(cfg, "PersistentPGDReconLoss").update(n_warmup_steps=0),
    "noppgd": lambda cfg: drop_loss_metric(cfg, "PersistentPGDReconLoss"),
    "nofaith": lambda cfg: drop_loss_metric(cfg, "FaithfulnessLoss"),
    "passes": lambda cfg: cfg["runtime"]["launch_env"]["env"].update(PASS_TIMING_ENV),
}


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for arm, mutate in ARMS.items():
        cfg = yaml.safe_load(BASE_CONFIG.read_text())
        cfg["run_name"] = f"cgrid-{arm}"
        cfg.pop("wandb", None)
        cfg["data"]["data_files"] = BTDR_DATA
        cfg["runtime"]["launch_env"] = {"env": {"JAX_LOG_COMPILES": "1"}}
        mutate(cfg)
        path = out / f"cgrid_{arm}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1])
