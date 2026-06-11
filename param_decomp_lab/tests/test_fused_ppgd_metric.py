"""Fused-KL PPGD recon (lab metric, `use_fused_kl=true`) == unfused PPGD recon.

Same tiny vendored Llama (real LM head), same seeds: the fused path computes the
identical KL through the LM head via the chunked fused kernel, so the live loss, the
post-step sources, the S14 source grads, and the V/U + CI grads from the single
backward must all match. The fused kernel chunks the vocab dim (float reassociation)
and two warmup Adam source steps compound the difference, so this is `assert_close`
within fp32 tolerances, not bit-identity.
"""

import pytest
import torch
from torch import Tensor

from param_decomp.component_model import CIOutputs
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.persistent_pgd_recon import (
    PersistentPGDReconLoss as CorePersistentPGDReconLoss,
)
from param_decomp_config.ci_fn import LayerwiseCiConfig
from param_decomp_config.losses import (
    AdamPGDConfig,
    PersistentPGDReconLossConfig,
    SCScope,
)
from param_decomp_config.schedule import ScheduleConfig
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.experiments.lm.vendored.llama_3_1.config import VendoredLlamaConfig
from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import VendoredLlama
from param_decomp_lab.metrics.dispatch import ALL_LOSS_METRIC_CLASSES
from param_decomp_lab.metrics.fused_persistent_pgd_recon import PersistentPGDReconLoss

_SITES = ("layers.1.mlp.gate_proj", "layers.1.mlp.up_proj", "layers.1.mlp.down_proj")
_C = 4
_VOCAB = 64


def _tiny_lm() -> LMComponentModel:
    cfg = VendoredLlamaConfig(
        model_type="VendoredLlama",
        max_position_embeddings=128,
        vocab_size=_VOCAB,
        n_layer=3,
        n_head=4,
        n_key_value_heads=2,
        n_embd=32,
        n_intermediate=64,
        rope_scaling=None,
        rms_norm_eps=1e-5,
    )
    torch.manual_seed(0)
    target = VendoredLlama(cfg)
    return LMComponentModel.build(
        target_model=target,
        decomposition_targets=[DecompositionTarget(module_path=s, C=_C) for s in _SITES],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
        sigmoid_type="leaky_hard",
    )


def _ppgd_cfg(use_fused_kl: bool) -> PersistentPGDReconLossConfig:
    return PersistentPGDReconLossConfig(
        coeff=0.5,
        optimizer=AdamPGDConfig(
            beta1=0.5,
            beta2=0.99,
            lr_schedule=ScheduleConfig(
                fn_type="constant", start_val=0.01, warmup_pct=0.0, final_val_frac=1.0
            ),
        ),
        scope=SCScope(),
        n_warmup_steps=2,
        use_fused_kl=use_fused_kl,
    )


def _run_one_step(use_fused_kl: bool) -> dict[str, Tensor | dict[str, Tensor]]:
    lm = _tiny_lm()
    torch.manual_seed(7)
    batch = torch.randint(0, _VOCAB, (2, 8))
    with torch.no_grad():
        target_out = lm(batch).detach()
    torch.manual_seed(11)
    ci_leaves = {s: torch.rand(2, 8, _C).requires_grad_(True) for s in _SITES}
    ctx = MetricContext(
        model=lm,
        batch=batch,
        target_out=target_out,
        pre_weight_acts={},
        ci=CIOutputs(lower_leaky=ci_leaves, upper_leaky=ci_leaves, pre_sigmoid=ci_leaves),
        weight_deltas=lm.calc_weight_deltas(),
        step=0,
        total_steps=10,
        use_delta_component=True,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=recon_loss_kl,
        is_eval=False,
    )

    metric = PersistentPGDReconLoss(_ppgd_cfg(use_fused_kl))
    metric.bind(model=lm, device="cpu")
    metric.reset()
    torch.manual_seed(123)  # the source-init rand draw inside the first update
    loss = metric.update(ctx)
    assert isinstance(loss, Tensor)

    metric.before_backward(loss)
    assert metric._pending_source_grads is not None
    source_grads = {k: v.clone() for k, v in metric._pending_source_grads.items()}
    loss.backward()
    metric.after_backward()

    assert metric.state is not None
    vu_grads: dict[str, Tensor] = {}
    for name, comp in lm.components.items():
        assert comp.V.grad is not None and comp.U.grad is not None
        vu_grads[f"{name}.V"] = comp.V.grad.clone()
        vu_grads[f"{name}.U"] = comp.U.grad.clone()
    ci_grads: dict[str, Tensor] = {}
    for site, leaf in ci_leaves.items():
        assert leaf.grad is not None
        ci_grads[site] = leaf.grad.clone()
    return {
        "loss": loss.detach(),
        "source_grads": source_grads,
        "sources": {k: v.detach().clone() for k, v in metric.state.sources.items()},
        "vu_grads": vu_grads,
        "ci_grads": ci_grads,
    }


def test_fused_ppgd_matches_unfused() -> None:
    fused = _run_one_step(use_fused_kl=True)
    unfused = _run_one_step(use_fused_kl=False)

    def close(a: Tensor, b: Tensor, what: str) -> None:
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-6, msg=lambda m: f"{what}: {m}")

    assert isinstance(fused["loss"], Tensor) and isinstance(unfused["loss"], Tensor)
    close(fused["loss"], unfused["loss"], "live loss")
    for group in ("source_grads", "sources", "vu_grads", "ci_grads"):
        f, u = fused[group], unfused[group]
        assert isinstance(f, dict) and isinstance(u, dict)
        assert f.keys() == u.keys()
        for k in u:
            close(f[k], u[k], f"{group}[{k}]")


def test_fused_ppgd_reuses_hidden_target_out() -> None:
    """`target_out_kind == "hidden"` (the trainer ran the step under the bypass) must be
    bit-identical to the metric recomputing its own bypassed clean target."""
    lm = _tiny_lm()
    torch.manual_seed(7)
    batch = torch.randint(0, _VOCAB, (2, 8))
    torch.manual_seed(11)
    ci_leaves = {s: torch.rand(2, 8, _C) for s in _SITES}

    def run(target_out: Tensor, kind: str) -> Tensor:
        ctx = MetricContext(
            model=lm,
            batch=batch,
            target_out=target_out,
            pre_weight_acts={},
            ci=CIOutputs(lower_leaky=ci_leaves, upper_leaky=ci_leaves, pre_sigmoid=ci_leaves),
            weight_deltas=lm.calc_weight_deltas(),
            step=0,
            total_steps=10,
            use_delta_component=True,
            sampling="continuous",
            n_mask_samples=1,
            reconstruction_loss=recon_loss_kl,
            is_eval=False,
            target_out_kind=kind,  # pyright: ignore[reportArgumentType]
        )
        metric = PersistentPGDReconLoss(_ppgd_cfg(use_fused_kl=True))
        metric.bind(model=lm, device="cpu")
        metric.reset()
        torch.manual_seed(123)
        loss = metric.update(ctx)
        assert isinstance(loss, Tensor)
        return loss.detach()

    with torch.no_grad():
        logits = lm(batch).detach()
        with lm.bypass_lm_head():
            hidden = lm(batch).detach()
    loss_recomputed = run(logits, "logits")
    loss_reused = run(hidden, "hidden")
    assert torch.equal(loss_recomputed, loss_reused)


def test_lab_dispatch_routes_ppgd_to_fused_classes() -> None:
    assert ALL_LOSS_METRIC_CLASSES["PersistentPGDReconLoss"] is PersistentPGDReconLoss


def test_core_metric_rejects_use_fused_kl() -> None:
    with pytest.raises(AssertionError, match="use_fused_kl"):
        CorePersistentPGDReconLoss(_ppgd_cfg(use_fused_kl=True))
