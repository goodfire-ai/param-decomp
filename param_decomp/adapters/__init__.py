"""Harvest method adapters: method-specific logic for the generic harvest pipeline.

Each decomposition method (PD, CLT, MOLT, Transcoder) provides an adapter that knows how to:
- Load the model and build a dataloader
- Compute firings and activations from a batch (harvest_fn)
- Report layer structure and vocab size

Construct via adapter_from_config(method_config).
"""

from param_decomp.adapters.base import DecompositionAdapter
from param_decomp.harvest.config import DecompositionMethodHarvestConfig


def adapter_from_config(method_config: DecompositionMethodHarvestConfig) -> DecompositionAdapter:
    from param_decomp.harvest.config import (
        CLTHarvestConfig,
        ParamDecompHarvestConfig,
        TranscoderHarvestConfig,
    )

    match method_config:
        case ParamDecompHarvestConfig():
            from param_decomp.adapters.param_decomp import ParamDecompAdapter

            return ParamDecompAdapter(method_config.wandb_path)
        case TranscoderHarvestConfig():
            from param_decomp.adapters.transcoder import TranscoderAdapter

            return TranscoderAdapter(method_config)
        case CLTHarvestConfig():
            from param_decomp.adapters.clt import CLTAdapter

            return CLTAdapter(method_config)


def adapter_from_id(decomposition_id: str) -> DecompositionAdapter:
    """Construct an adapter from a decomposition ID (e.g. "s-abc123", "tc-1a2b3c4d").

    Recovers the full method config from the harvest DB (which is always populated
    before downstream steps like autointerp run).
    """
    return adapter_from_config(_load_method_config(decomposition_id))


def _load_method_config(decomposition_id: str) -> DecompositionMethodHarvestConfig:
    from pydantic import TypeAdapter

    from param_decomp.harvest.repo import HarvestRepo

    repo = HarvestRepo.open_most_recent(decomposition_id)
    assert repo is not None, (
        f"No harvest data found for {decomposition_id!r}. "
        f"Run pd-harvest first to populate the method config."
    )
    config_dict = repo.get_config()
    method_config_raw = config_dict["method_config"]
    ta = TypeAdapter(DecompositionMethodHarvestConfig)
    return ta.validate_python(method_config_raw)
