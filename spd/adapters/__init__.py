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
        MOLTHarvestConfig,
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
            raise NotImplementedError("CLT adapter not implemented yet")
        case MOLTHarvestConfig():
            raise NotImplementedError("MOLT adapter not implemented yet")


def adapter_from_id(id: str) -> DecompositionAdapter:
    """Construct an adapter from just a decomposition ID (e.g. "s-abc123").

    Only works for methods whose adapter can be constructed from an ID alone (SPD).
    For transcoders, use adapter_from_config() with the full method config.
    """
    from spd.adapters.spd import SPDAdapter

    if id.startswith("s-"):
        return SPDAdapter(id)

    raise ValueError(
        f"Cannot construct adapter from ID alone: {id!r}. "
        f"Use adapter_from_config() with the full method config."
    )
