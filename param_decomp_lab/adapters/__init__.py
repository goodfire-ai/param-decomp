"""PD run-loading adapter: recover model metadata from a saved JAX single-pool run.

The adapter loads the target model and reports its layer structure and vocab size.
Construct via adapter_from_config(method_config).
"""

from param_decomp_lab.adapters.pd import PDAdapter, is_jax_run
from param_decomp_lab.harvest.config import ParamDecompHarvestConfig


def adapter_from_config(method_config: ParamDecompHarvestConfig) -> PDAdapter:
    assert is_jax_run(method_config.wandb_path), (
        f"{method_config.wandb_path}: not a loadable PD run (missing launch_config.yaml or orbax ckpts/)."
    )
    return PDAdapter(method_config.wandb_path)


def adapter_from_id(decomposition_id: str) -> PDAdapter:
    """Construct an adapter from a decomposition ID (e.g. "s-abc123", "p-1a2b3c4d").

    Recovers the full method config from the harvest DB (which is always populated
    before downstream steps like autointerp run).
    """
    from pydantic import TypeAdapter

    from param_decomp_lab.harvest.repo import HarvestRepo

    repo = HarvestRepo.open_most_recent(decomposition_id)
    assert repo is not None, (
        f"No harvest data found for {decomposition_id!r}. "
        f"Run pd-harvest first to populate the method config."
    )
    method_config_raw = repo.get_config()["method_config"]
    method_config = TypeAdapter(ParamDecompHarvestConfig).validate_python(method_config_raw)
    return adapter_from_config(method_config)
