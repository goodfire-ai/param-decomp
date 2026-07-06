"""Generate the compile-time scaling-grid arm configs (btdr, dp32 = 4 nodes/arm).

Derived from llama8b_full32L_HSDP_b32_dp32.yaml (full 32L / 224 sites / per-rank
batch 1 / seq512), cut to 12 steps with eval never firing and checkpoints off, one
mutation per arm, so the grid attributes jit_step compile time to specific graph
sections. (dp8/1-node was tried first and OOMs in SETUP: the fp32 masters + Adam
shard ÷N, and ÷8 at full 32L/full-C exceeds a rank's HBM before compile starts.)

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

BASE_CONFIG = Path(__file__).parent.parent / "llama8b_full32L_HSDP_b32_dp32.yaml"
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


def set_dp(cfg: dict[str, Any], dp: int) -> None:
    cfg["runtime"]["dp"] = dp
    cfg["pd"]["batch_size"] = dp
    cfg["eval"]["batch_size"] = dp


def halve_c(cfg: dict[str, Any]) -> None:
    for t in cfg["pd"]["decomposition_targets"]:
        t["C"] //= 2


def ci_blocks(cfg: dict[str, Any], n: int) -> None:
    cfg["pd"]["ci_config"]["n_blocks"] = n


ARMS: dict[str, Any] = {
    "base4": lambda cfg: None,
    "c1": lambda cfg: set_chunks(cfg, 224),
    "c2": lambda cfg: set_chunks(cfg, 112),
    "c8": lambda cfg: set_chunks(cfg, 28),
    "nw0": lambda cfg: loss_metric(cfg, "PersistentPGDReconLoss").update(n_warmup_steps=0),
    "noppgd": lambda cfg: drop_loss_metric(cfg, "PersistentPGDReconLoss"),
    "nofaith": lambda cfg: drop_loss_metric(cfg, "FaithfulnessLoss"),
    "passes": lambda cfg: cfg["runtime"]["launch_env"]["env"].update(PASS_TIMING_ENV),
    # round-4 attribution arms (shape/CI scale — the btdr Chalf-ci2blk runs compiled ~3 min
    # while the full-shape cw-east runs took ~24: these isolate which knob buys that)
    "chalf": halve_c,
    "ci2blk": lambda cfg: ci_blocks(cfg, 2),
    # topology arms: round 3 showed jit_step compile flat (~5 min) across EVERY
    # graph-structure knob at dp32 — these measure the mesh-size curve on fixed hardware
    "dp64": lambda cfg: set_dp(cfg, 64),
    "dp128": lambda cfg: set_dp(cfg, 128),
}


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for arm, mutate in ARMS.items():
        cfg = yaml.safe_load(BASE_CONFIG.read_text())
        cfg["run_name"] = f"cgrid-{arm}"
        cfg.pop("wandb", None)
        cfg["data"]["data_files"] = BTDR_DATA
        cfg["pd"]["steps"] = 12
        cfg["eval"]["every"] = 100000
        cfg["eval"]["slow_every"] = 1000000
        cfg["runtime"]["launch_env"] = {
            "env": {"JAX_LOG_COMPILES": "1"},
            "profile": {"no_checkpoint": True},
        }
        mutate(cfg)
        path = out / f"cgrid_{arm}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1])
