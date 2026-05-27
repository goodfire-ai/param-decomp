from collections.abc import Iterator
from pathlib import Path
from typing import override

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import EsmForMaskedLM

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.configs import Cadence, OptimizerConfig, PDConfig, RuntimeConfig
from param_decomp.decomposition_targets import DecompositionTargetConfig
from param_decomp.metrics.faithfulness import FaithfulnessLossConfig
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLossConfig
from param_decomp.metrics.stochastic_recon import StochasticReconLossConfig
from param_decomp.metrics.stochastic_recon_layerwise import (
    StochasticReconLayerwiseLossConfig,
)
from param_decomp.optimize import optimize
from param_decomp.schedule import ScheduleConfig
from param_decomp_lab.batch_and_loss_fns import make_run_batch, recon_loss_kl
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed


class _RandAATokens(IterableDataset[Tensor]):
    """Infinite stream of `(seq_len,)` random amino-acid token-id tensors."""

    def __init__(self, *, seq_len: int, seed: int):
        super().__init__()
        self.seq_len = seq_len
        self.seed = seed

    @override
    def __iter__(self) -> Iterator[Tensor]:
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        while True:
            yield torch.randint(4, 24, (self.seq_len,), generator=gen, dtype=torch.long)


def _collate(batch: list[Tensor]) -> Tensor:
    return torch.stack(batch)


@pytest.mark.slow
def test_esm2_decomposition_happy_path(tmp_path: Path) -> None:
    """Smoke-test PD on ESM2 with the smallest 8M variant.

    Decomposes two MLP linears in the first encoder layer for two steps with synthetic
    random amino-acid tokens. The 150M variant is left to the `pd-esm2` CLI smoke; this
    test uses the 8M variant so it stays cheap on CI.
    """
    set_seed(0)
    device = "cpu"

    pd_config = PDConfig(
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="vector_mlp", hidden_dims=[16]),
        decomposition_targets=[
            DecompositionTargetConfig(module_pattern="esm.encoder.layer.0.intermediate.dense", C=8),
            DecompositionTargetConfig(module_pattern="esm.encoder.layer.0.output.dense", C=8),
        ],
        loss_metrics=[
            ImportanceMinimalityLossConfig(coeff=1e-3, pnorm=2.0, beta=0.0, eps=1e-12),
            StochasticReconLayerwiseLossConfig(coeff=1.0),
            StochasticReconLossConfig(coeff=1.0),
            FaithfulnessLossConfig(coeff=1.0),
        ],
        components_optimizer=OptimizerConfig(
            lr_schedule=ScheduleConfig(
                start_val=1e-3, fn_type="cosine", warmup_pct=0.01, final_val_frac=0.0
            ),
        ),
        ci_fn_optimizer=OptimizerConfig(
            lr_schedule=ScheduleConfig(
                start_val=1e-3, fn_type="cosine", warmup_pct=0.01, final_val_frac=0.0
            ),
        ),
        batch_size=2,
        steps=2,
    )

    target_model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    target_model.eval()

    train_loader: DataLoader[Tensor] = DataLoader(
        _RandAATokens(seq_len=16, seed=0),
        batch_size=pd_config.batch_size,
        collate_fn=_collate,
    )

    sink = RunSink.local(tmp_path)
    cadence = Cadence(train_log_every=1, save_every=None)

    optimize(
        target_model=target_model,
        train_loader=train_loader,
        run_batch=make_run_batch("logits"),
        reconstruction_loss=recon_loss_kl,
        pd_config=pd_config,
        runtime_config=RuntimeConfig(device=device, autocast_bf16=False),
        sink=sink,
        cadence=cadence,
        eval_loop=None,
    )
