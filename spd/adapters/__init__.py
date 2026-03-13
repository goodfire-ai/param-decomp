"""Harvest method adapters: method-specific logic for the generic harvest pipeline.

Each decomposition method (SPD, CLT, MOLT, Transcoder) provides an adapter that knows how to:
- Load the model and build a dataloader
- Compute firings and activations from a batch (harvest_fn)
- Report layer structure and vocab size

Construct via adapter_from_config(method_config).
"""

from spd.adapters.base import DecompositionAdapter
from spd.harvest.config import DecompositionMethodHarvestConfig


def adapter_from_config(method_config: DecompositionMethodHarvestConfig) -> DecompositionAdapter:
    from spd.harvest.config import (
        CLTHarvestConfig,
        SPDHarvestConfig,
        TranscoderHarvestConfig,
    )

    match method_config:
        case SPDHarvestConfig():
            from spd.adapters.spd import SPDAdapter

            return SPDAdapter(method_config.id)
        case TranscoderHarvestConfig():
            from spd.adapters.transcoder import TranscoderAdapter

            return TranscoderAdapter(method_config)
        case CLTHarvestConfig():
            from spd.adapters.clt import CLTAdapter

            return CLTAdapter(method_config)


def adapter_from_id(decomposition_id: str) -> DecompositionAdapter:
    """Construct an adapter from a decomposition ID (e.g. "s-abc123", "tc-1a2b3c4d").

    For SPD runs, the ID is sufficient. For other methods, recovers the full
    method config from the harvest DB (which is always populated before downstream
    steps like autointerp run).
    """
    if decomposition_id.startswith("s-"):
        from spd.adapters.spd import SPDAdapter

        return SPDAdapter(decomposition_id)

    return adapter_from_config(_load_method_config(decomposition_id))


def _load_method_config(decomposition_id: str) -> DecompositionMethodHarvestConfig:
    from pydantic import TypeAdapter

    from spd.harvest.repo import HarvestRepo

    repo = HarvestRepo.open_most_recent(decomposition_id)
    assert repo is not None, (
        f"No harvest data found for {decomposition_id!r}. "
        f"Run spd-harvest first to populate the method config."
    )
    config_dict = repo.get_config()
    method_config_raw = config_dict["method_config"]
    ta = TypeAdapter(DecompositionMethodHarvestConfig)
    return ta.validate_python(method_config_raw)
