import torch

from spd.adapters.base import DecompositionAdapter
from spd.adapters.spd import SPDAdapter
from spd.adapters.transcoder import TranscoderAdapter
from spd.harvest.config import (
    CLTHarvestConfig,
    DecompositionMethodHarvestConfig,
    MOLTHarvestConfig,
    SPDHarvestConfig,
    TranscoderHarvestConfig,
)
from spd.harvest.harvest_fn.base import HarvestFn
from spd.harvest.harvest_fn.spd import SPDHarvestFn
from spd.harvest.harvest_fn.transcoder import TranscoderHarvestFn


def make_harvest_fn(
    device: torch.device,
    method_config: DecompositionMethodHarvestConfig,
    adapter: DecompositionAdapter,
) -> HarvestFn:
    match method_config, adapter:
        case SPDHarvestConfig(), SPDAdapter():
            return SPDHarvestFn(method_config, adapter, device=device)
        case TranscoderHarvestConfig(), TranscoderAdapter():
            return TranscoderHarvestFn(adapter, method_config.activation_threshold, device=device)
        case CLTHarvestConfig(), _:
            raise NotImplementedError("CLT harvest not implemented yet")
        case MOLTHarvestConfig(), _:
            raise NotImplementedError("MOLT harvest not implemented yet")
        case _, _:
            raise ValueError(f"Unsupported method config: {method_config} and adapter: {adapter}")
