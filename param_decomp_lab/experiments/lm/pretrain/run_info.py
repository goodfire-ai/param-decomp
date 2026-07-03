"""Locate a pretrained target's cache dir from its run id (torch-free).

`pretrain.train` writes its weights to `PARAM_DECOMP_OUT_DIR/pretrain_cache/<project>-<run_id>/`
(safetensors + `model_config.yaml`) — the layout the decomposition trainer's loader
(`param_decomp.llama_simple_mlp.load_target_from_pretrain_cache`) reads, keyed by the
wandb run path `<entity>/<project>/<run_id>` in a `kind: pretrained` decomposition target
spec. This is the read-side index: given a run id, find the cache and parse its config.

The torch `PretrainRunInfo` (`torch-oracle:.../run_info.py`) additionally downloaded from
wandb and loaded torch state dicts; both are gone — `pretrain.train` writes the cache
directly to shared FS at every save, so there is nothing to download, and the weights are
loaded JAX-side by the decomposition trainer.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR


@dataclass(frozen=True)
class PretrainCache:
    cache_dir: Path
    model_config: dict[str, object]
    checkpoint: Path

    @property
    def model_type(self) -> str:
        return str(self.model_config["model_type"])


def find_pretrain_cache(project: str, run_id: str) -> PretrainCache:
    cache_dir = PARAM_DECOMP_OUT_DIR / "pretrain_cache" / f"{project}-{run_id}"
    assert cache_dir.is_dir(), f"no pretrain cache at {cache_dir}"
    ckpts = sorted(cache_dir.glob("model_step_*.safetensors"))
    assert len(ckpts) == 1, f"expected one model_step_*.safetensors in {cache_dir}, found {ckpts}"
    model_config = yaml.safe_load((cache_dir / "model_config.yaml").read_text())
    return PretrainCache(cache_dir=cache_dir, model_config=model_config, checkpoint=ckpts[0])
