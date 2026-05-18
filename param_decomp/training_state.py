"""Resumption checkpoint for a PD training run.

A ``TrainingState`` snapshots everything needed to continue ``run_pd.optimize`` from where
it left off:

- ``step``: next step to execute on resume (i.e. the run was about to enter ``step``)
- model and optimizer state dicts
- per-PPGD-config state (sources + inner optimizer state)
- per-rank RNG state for ``torch``, ``torch.cuda``, ``numpy``, ``random``
- ``StatefulLoop`` state for train / eval data (epoch + within-epoch position)
- ``wandb_run_id``: the W&B run id that produced this checkpoint, so a resumed run can
  fork from it.

Written to a single rolling file ``<out_dir>/training_state.pt`` via atomic rename. Only
rank 0 writes; all ranks save+restore their own RNG so per-rank divergence (see
``seed_per_rank``) is preserved.

Bit-exact resumption is not promised: stochastic mask sampling, CUDA nondeterminism, and
synthetic data generators (TMS / ResidMLP) all introduce drift. The goal is *equivalent*
training: same step counter, same model + optimizer state at the cut, same data sequence
for map-style / DistributedSampler / IterableDataset loaders, and same PGD sources.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

TRAINING_STATE_FILENAME = "training_state.pt"


@dataclass
class TrainingState:
    """Resumable snapshot of an in-progress PD training loop. See module docstring."""

    step: int
    model_sd: dict[str, Any]
    components_opt_sd: dict[str, Any]
    ci_fn_opt_sd: dict[str, Any]
    # PPGD state keyed by index into ``persistent_pgd_configs`` (positional, since config
    # objects are not hashable across processes).
    ppgd_sd: list[dict[str, Any]]
    train_loop_sd: dict[str, int]
    eval_loop_sd: dict[str, int]
    rng_sd: dict[str, Any]
    wandb_run_id: str | None

    def save(self, path: Path) -> None:
        """Atomic-write the state to ``path``. Caller ensures parent dir exists."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.to_dict(), tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path, *, map_location: str | torch.device = "cpu") -> "TrainingState":
        assert path.exists(), f"TrainingState file not found: {path}"
        # weights_only=False because the state contains plain dicts / numpy arrays alongside
        # tensors. Resume data is user-trusted (it's the user's own prior checkpoint).
        data = torch.load(path, map_location=map_location, weights_only=False)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "model_sd": self.model_sd,
            "components_opt_sd": self.components_opt_sd,
            "ci_fn_opt_sd": self.ci_fn_opt_sd,
            "ppgd_sd": self.ppgd_sd,
            "train_loop_sd": self.train_loop_sd,
            "eval_loop_sd": self.eval_loop_sd,
            "rng_sd": self.rng_sd,
            "wandb_run_id": self.wandb_run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingState":
        return cls(
            step=int(data["step"]),
            model_sd=data["model_sd"],
            components_opt_sd=data["components_opt_sd"],
            ci_fn_opt_sd=data["ci_fn_opt_sd"],
            ppgd_sd=list(data["ppgd_sd"]),
            train_loop_sd=dict(data["train_loop_sd"]),
            eval_loop_sd=dict(data["eval_loop_sd"]),
            rng_sd=data["rng_sd"],
            wandb_run_id=data.get("wandb_run_id"),
        )


def capture_rng_state() -> dict[str, Any]:
    """Snapshot Python, NumPy, and Torch (CPU+CUDA) RNG state for this rank."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state previously captured by ``capture_rng_state``."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])
