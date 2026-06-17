"""PD run-loading adapters: recover model metadata from a saved JAX single-pool run.

An adapter loads the target model and reports its layer structure and vocab size.
Training is JAX now, so the only adapter is `JaxPDAdapter`; the torch-run loader was
dropped with the torch-trainer shed and is slated to return JAX-native (the #10
torch->jax adapter).

Construct via adapter_from_config(method_config).
"""

from param_decomp_lab.adapters.base import DecompositionAdapter
from param_decomp_lab.harvest.config import ParamDecompHarvestConfig


def adapter_from_config(method_config: ParamDecompHarvestConfig) -> DecompositionAdapter:
    from param_decomp_lab.adapters.jax_pd import JaxPDAdapter, is_jax_run

    assert is_jax_run(method_config.wandb_path), (
        f"{method_config.wandb_path}: only JAX single-pool runs are loadable; "
        "torch-run loading was dropped (re-add tracked as the #10 torch->jax adapter)."
    )
    return JaxPDAdapter(method_config.wandb_path)


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
