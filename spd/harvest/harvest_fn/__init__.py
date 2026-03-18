import torch

from spd.adapters.base import DecompositionAdapter
from spd.adapters.clt import CLTAdapter
from spd.adapters.spd import SPDAdapter
from spd.adapters.transcoder import TranscoderAdapter
from spd.harvest.config import (
    CLTHarvestConfig,
    DecompositionMethodHarvestConfig,
    SPDHarvestConfig,
    TranscoderHarvestConfig,
)
from spd.harvest.harvest_fn.base import HarvestFn
from spd.harvest.harvest_fn.clt import CLTHarvestFn
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
            return TranscoderHarvestFn(adapter, device=device)
        case CLTHarvestConfig(), CLTAdapter():
            return CLTHarvestFn(adapter, device=device)
        case _:
            raise ValueError(f"Unsupported method config: {method_config} and adapter: {adapter}")
