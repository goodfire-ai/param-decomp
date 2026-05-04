import torch

from param_decomp.adapters.base import DecompositionAdapter
from param_decomp.adapters.clt import CLTAdapter
from param_decomp.adapters.param_decomp import ParamDecompAdapter
from param_decomp.adapters.transcoder import TranscoderAdapter
from param_decomp.harvest.config import (
    CLTHarvestConfig,
    DecompositionMethodHarvestConfig,
    ParamDecompHarvestConfig,
    TranscoderHarvestConfig,
)
from param_decomp.harvest.harvest_fn.base import HarvestFn
from param_decomp.harvest.harvest_fn.clt import CLTHarvestFn
from param_decomp.harvest.harvest_fn.param_decomp import ParamDecompHarvestFn
from param_decomp.harvest.harvest_fn.transcoder import TranscoderHarvestFn


def make_harvest_fn(
    device: torch.device,
    method_config: DecompositionMethodHarvestConfig,
    adapter: DecompositionAdapter,
) -> HarvestFn:
    match method_config, adapter:
        case ParamDecompHarvestConfig(), ParamDecompAdapter():
            return ParamDecompHarvestFn(method_config, adapter, device=device)
        case TranscoderHarvestConfig(), TranscoderAdapter():
            return TranscoderHarvestFn(adapter, device=device)
        case CLTHarvestConfig(), CLTAdapter():
            return CLTHarvestFn(adapter, device=device)
        case _:
            raise ValueError(f"Unsupported method config: {method_config} and adapter: {adapter}")
