"""PD run-loading adapters: recover model metadata + a dataloader from a saved run.

A PD adapter knows how to load the target model, report its layer structure and vocab
size, and build a streaming dataloader. Torch runs route to `PDAdapter`, JAX single-pool
runs to `JaxPDAdapter`.

Construct via adapter_from_config(method_config).
"""

from param_decomp_lab.adapters.base import DecompositionAdapter
from param_decomp_lab.harvest.config import ParamDecompHarvestConfig


def adapter_from_config(method_config: ParamDecompHarvestConfig) -> DecompositionAdapter:
    from param_decomp_lab.adapters.jax_pd import JaxPDAdapter, is_jax_run

    if is_jax_run(method_config.wandb_path):
        return JaxPDAdapter(method_config.wandb_path)

    from param_decomp_lab.adapters.pd import PDAdapter

    return PDAdapter(method_config.wandb_path)


def adapter_from_id(decomposition_id: str) -> DecompositionAdapter:
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
