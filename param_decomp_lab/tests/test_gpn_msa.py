from pathlib import Path

import pytest

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
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.experiments.gpn_msa.run import (
    GPNMSADataConfig,
    GPNMSATargetConfig,
    build_gpn_msa_loader,
    build_target,
    make_run_batch,
)
from param_decomp_lab.run_sink import RunSink
from param_decomp_lab.seed import set_seed


@pytest.mark.slow
def test_gpn_msa_decomposition_happy_path(tmp_path: Path) -> None:
    """Smoke test: GPN-MSA loads + two PD steps run cleanly on a synthetic MSA batch."""
    gpn_model = pytest.importorskip("gpn.model")
    del gpn_model

    set_seed(0)
    device = "cpu"

    pd_config = PDConfig(
        seed=0,
        n_mask_samples=1,
        ci_config=LayerwiseCiConfig(fn_type="vector_mlp", hidden_dims=[16]),
        decomposition_targets=[
            DecompositionTargetConfig(
                module_pattern="model.encoder.layer.0.attention.self.query", C=4
            ),
            DecompositionTargetConfig(
                module_pattern="model.encoder.layer.0.intermediate.dense", C=4
            ),
        ],
        loss_metrics=[
            ImportanceMinimalityLossConfig(coeff=1e-2, pnorm=0.9, beta=0.5, eps=1e-12),
            StochasticReconLayerwiseLossConfig(coeff=1.0),
            StochasticReconLossConfig(coeff=1.0),
            FaithfulnessLossConfig(coeff=1.0),
        ],
        components_optimizer=OptimizerConfig(
            lr_schedule=ScheduleConfig(
                start_val=1e-3, fn_type="cosine", warmup_pct=0.0, final_val_frac=0.0
            ),
        ),
        ci_fn_optimizer=OptimizerConfig(
            lr_schedule=ScheduleConfig(
                start_val=1e-3, fn_type="cosine", warmup_pct=0.0, final_val_frac=0.0
            ),
        ),
        batch_size=2,
        steps=2,
    )

    target_cfg = GPNMSATargetConfig(model_name="songlab/gpn-msa-sapiens")
    data_cfg = GPNMSADataConfig(
        seq_len=16,
        n_species=89,
        vocab_size=6,
        aux_features_vocab_size=5,
        n_synthetic_samples=32,
    )

    target_model = build_target(target_cfg)
    train_loader = build_gpn_msa_loader(
        target_cfg,
        data_cfg,
        split="train",
        device=device,
        batch_size=pd_config.batch_size,
        seed=pd_config.seed,
    )

    optimize(
        target_model=target_model,
        train_loader=train_loader,
        run_batch=make_run_batch(target_cfg),
        reconstruction_loss=recon_loss_kl,
        pd_config=pd_config,
        runtime_config=RuntimeConfig(device=device),
        sink=RunSink.local(tmp_path),
        cadence=Cadence(train_log_every=1, save_every=None),
        eval_loop=None,
    )
