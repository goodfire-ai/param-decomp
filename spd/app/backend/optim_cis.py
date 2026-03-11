from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import torch.optim as optim
from jaxtyping import Bool, Float
from pydantic import BaseModel
from torch import Tensor
from tqdm.auto import tqdm

from spd.configs import ImportanceMinimalityLossConfig, PGDInitStrategy, SamplingType
from spd.metrics import importance_minimality_loss
from spd.metrics.pgd_utils import get_pgd_init_tensor, interpolate_pgd_mask
from spd.models.component_model import CIOutputs, ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.routing import AllLayersRouter
from spd.spd_types import Probability
from spd.utils.component_utils import calc_ci_l_zero, calc_stochastic_component_mask_info
from spd.utils.general_utils import bf16_autocast

MaskType = Literal["stochastic", "ci"]


class AdvPGDConfig(BaseModel):
    """PGD adversary config for robust CI optimization."""

    n_steps: int
    step_size: float
    init: PGDInitStrategy


class CELossConfig(BaseModel):
    """Cross-entropy loss: optimize for a specific token at a position."""

    type: Literal["ce"] = "ce"
    coeff: float
    position: int
    label_token: int


class KLLossConfig(BaseModel):
    """KL divergence loss: match target model distribution at a position."""

    type: Literal["kl"] = "kl"
    coeff: float
    position: int


class LogitLossConfig(BaseModel):
    """Logit loss: maximize the pre-softmax logit for a specific token at a position."""

    type: Literal["logit"] = "logit"
    coeff: float
    position: int
    label_token: int


class MeanKLLossConfig(BaseModel):
    """Mean KL divergence loss: match target model distribution across all positions."""

    type: Literal["mean_kl"] = "mean_kl"
    coeff: float = 1.0


PositionalLossConfig = CELossConfig | KLLossConfig | LogitLossConfig
LossConfig = CELossConfig | KLLossConfig | LogitLossConfig | MeanKLLossConfig


def compute_recon_loss(
    logits: Tensor,
    loss_config: LossConfig,
    target_out: Tensor,
    device: str,
) -> Tensor:
    """Compute recon loss (CE, KL, or mean KL) from model output logits."""
    match loss_config:
        case CELossConfig(position=pos, label_token=label_token):
            return F.cross_entropy(
                logits[0, pos, :].unsqueeze(0),
                torch.tensor([label_token], device=device),
            )
        case KLLossConfig(position=pos):
            target_probs = F.softmax(target_out[0, pos, :], dim=-1)
            pred_log_probs = F.log_softmax(logits[0, pos, :], dim=-1)
            return F.kl_div(pred_log_probs, target_probs, reduction="sum")
        case LogitLossConfig(position=pos, label_token=label_token):
            return -logits[0, pos, label_token]
        case MeanKLLossConfig():
            target_probs = F.softmax(target_out, dim=-1)
            pred_log_probs = F.log_softmax(logits, dim=-1)
            # sum over vocab, mean over positions (consistent with batched version)
            return F.kl_div(pred_log_probs, target_probs, reduction="none").sum(dim=-1).mean(dim=-1)


@dataclass
class AliveComponentInfo:
    """Info about which components are alive at each position for each layer."""

    alive_masks: dict[str, Bool[Tensor, "1 seq C"]]  # Per-layer masks of alive positions
    alive_counts: dict[str, list[int]]  # Number of alive components per position per layer


def compute_alive_info(
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]],
) -> AliveComponentInfo:
    """Compute which (position, component) pairs are alive (CI > 0)."""
    alive_masks: dict[str, Bool[Tensor, "1 seq C"]] = {}
    alive_counts: dict[str, list[int]] = {}

    for layer_name, ci in ci_lower_leaky.items():
        mask = ci > 0.0
        alive_masks[layer_name] = mask
        # Count alive components per position: mask is [1, seq, C], sum over C
        counts_per_pos = mask[0].sum(dim=-1)  # [seq]
        alive_counts[layer_name] = counts_per_pos.tolist()

    return AliveComponentInfo(alive_masks=alive_masks, alive_counts=alive_counts)


class OptimizationMetrics(BaseModel):
    """Final loss metrics from CI optimization."""

    ci_masked_label_prob: float | None = None  # Probability of label under CI mask (CE loss only)
    stoch_masked_label_prob: float | None = (
        None  # Probability of label under stochastic mask (CE loss only)
    )
    adv_pgd_label_prob: float | None = None  # Probability of label under adversarial mask (CE only)
    l0_total: float  # Total L0 (active components)


@dataclass
class OptimizableCIParams:
    """Container for optimizable CI pre-sigmoid parameters."""

    # List of pre-sigmoid tensors for alive positions at each sequence position
    ci_pre_sigmoid: dict[str, list[Tensor]]  # layer_name -> list of [alive_at_pos] values
    alive_info: AliveComponentInfo

    def create_ci_outputs(self, model: ComponentModel, device: str) -> CIOutputs:
        """Expand sparse pre-sigmoid values to full CI tensors and create CIOutputs."""
        pre_sigmoid: dict[str, Tensor] = {}

        for layer_name, mask in self.alive_info.alive_masks.items():
            # Create full tensors (default to 0 for non-alive positions)
            full_pre_sigmoid = torch.zeros_like(mask, dtype=torch.float32, device=device)

            # Get pre-sigmoid list for this layer
            layer_pre_sigmoid_list = self.ci_pre_sigmoid[layer_name]

            # For each position, place the values
            seq_len = mask.shape[1]
            for pos in range(seq_len):
                pos_mask = mask[0, pos, :]  # [C]
                pos_pre_sigmoid = layer_pre_sigmoid_list[pos]  # [alive_at_pos]
                full_pre_sigmoid[0, pos, pos_mask] = pos_pre_sigmoid

            pre_sigmoid[layer_name] = full_pre_sigmoid

        return CIOutputs(
            lower_leaky={k: model.lower_leaky_fn(v) for k, v in pre_sigmoid.items()},
            upper_leaky={k: model.upper_leaky_fn(v) for k, v in pre_sigmoid.items()},
            pre_sigmoid=pre_sigmoid,
        )

    def get_parameters(self) -> list[Tensor]:
        """Get all optimizable parameters."""
        params: list[Tensor] = []
        for layer_pre_sigmoid_list in self.ci_pre_sigmoid.values():
            params.extend(layer_pre_sigmoid_list)
        return params


def create_optimizable_ci_params(
    alive_info: AliveComponentInfo,
    initial_pre_sigmoid: dict[str, Tensor],
) -> OptimizableCIParams:
    """Create optimizable CI parameters for alive positions.

    Creates parameters initialized from the initial pre-sigmoid values for each
    (position, component) pair where initial CI > threshold.
    """
    ci_pre_sigmoid: dict[str, list[Tensor]] = {}

    for layer_name, mask in alive_info.alive_masks.items():
        # Get initial pre-sigmoid values for this layer
        layer_initial = initial_pre_sigmoid[layer_name]  # [1, seq, C]

        # Create a tensor for each position
        layer_pre_sigmoid_list: list[Tensor] = []
        seq_len = mask.shape[1]
        for pos in range(seq_len):
            pos_mask = mask[0, pos, :]  # [C]
            # Extract initial values for alive positions at this position
            initial_values = layer_initial[0, pos, pos_mask].clone().detach()
            initial_values.requires_grad_(True)
            layer_pre_sigmoid_list.append(initial_values)
        ci_pre_sigmoid[layer_name] = layer_pre_sigmoid_list

    return OptimizableCIParams(
        ci_pre_sigmoid=ci_pre_sigmoid,
        alive_info=alive_info,
    )


@dataclass
class OptimCIConfig:
    """Configuration for optimizing CI values on a single prompt."""

    seed: int

    # Optimization hyperparameters
    lr: float
    steps: int
    weight_decay: float
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"]
    lr_exponential_halflife: float | None
    lr_warmup_pct: Probability

    log_freq: int

    # Loss config (CE or KL — must target a specific position)
    imp_min_config: ImportanceMinimalityLossConfig
    loss_config: PositionalLossConfig

    sampling: SamplingType

    ce_kl_rounding_threshold: float
    mask_type: MaskType
    adv_pgd: AdvPGDConfig | None


ProgressCallback = Callable[[int, int, str], None]  # (current, total, stage)


class CISnapshot(BaseModel):
    """Snapshot of alive component counts during CI optimization for visualization."""

    step: int
    total_steps: int
    layers: list[str]
    seq_len: int
    initial_alive: list[list[int]]  # layers × seq
    current_alive: list[list[int]]  # layers × seq
    l0_total: float
    loss: float


CISnapshotCallback = Callable[[CISnapshot], None]


@dataclass
class OptimizeCIResult:
    """Result from CI optimization including params and final metrics."""

    params: OptimizableCIParams
    metrics: OptimizationMetrics


def run_adv_pgd(
    model: ComponentModel,
    tokens: Tensor,
    ci: dict[str, Float[Tensor, "1 seq C"]],
    alive_masks: dict[str, Bool[Tensor, "1 seq C"]],
    adv_config: AdvPGDConfig,
    target_out: Tensor,
    loss_config: LossConfig,
) -> dict[str, Float[Tensor, "1 seq C"]]:
    """Run PGD to find adversarial sources maximizing loss.

    Sources are optimized via signed gradient ascent. Only alive positions are optimized.
    Masks are computed as ci + (1 - ci) * source (same interpolation as training PGD).

    Returns detached adversarial source tensors.
    """
    ci_detached = {k: v.detach() for k, v in ci.items()}

    adv_sources: dict[str, Tensor] = {}
    for layer_name, ci_val in ci_detached.items():
        source = get_pgd_init_tensor(adv_config.init, tuple(ci_val.shape), str(ci_val.device))
        source[~alive_masks[layer_name]] = 0.0
        source.requires_grad_(True)
        adv_sources[layer_name] = source

    source_list = list(adv_sources.values())

    for _ in range(adv_config.n_steps):
        mask_infos = make_mask_infos(interpolate_pgd_mask(ci_detached, adv_sources))

        with bf16_autocast():
            out = model(tokens, mask_infos=mask_infos)

        loss = compute_recon_loss(out, loss_config, target_out, str(tokens.device))

        grads = torch.autograd.grad(loss, source_list)
        with torch.no_grad():
            for (layer_name, source), grad in zip(adv_sources.items(), grads, strict=True):
                source.add_(adv_config.step_size * grad.sign())
                source.clamp_(0.0, 1.0)
                source[~alive_masks[layer_name]] = 0.0

    return {k: v.detach() for k, v in adv_sources.items()}


def optimize_ci_values(
    model: ComponentModel,
    tokens: Tensor,
    config: OptimCIConfig,
    device: str,
    on_progress: ProgressCallback | None = None,
    on_ci_snapshot: CISnapshotCallback | None = None,
) -> OptimizeCIResult:
    """Optimize CI values for a single prompt.

    Args:
        model: The ComponentModel (weights will be frozen).
        tokens: Tokenized prompt of shape [1, seq_len].
        config: Optimization configuration (includes loss configs).
        device: Device to run on.

    Returns:
        OptimizeCIResult containing params and final metrics.
    """
    imp_min_coeff = config.imp_min_config.coeff
    assert imp_min_coeff is not None, "Importance minimality loss coefficient must be set"

    model.requires_grad_(False)

    with torch.no_grad(), bf16_autocast():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")
        initial_ci_outputs = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=config.sampling,
            detach_inputs=False,
        )
        target_out = output_with_cache.output.detach()

    alive_info = compute_alive_info(initial_ci_outputs.lower_leaky)
    ci_params: OptimizableCIParams = create_optimizable_ci_params(
        alive_info=alive_info,
        initial_pre_sigmoid=initial_ci_outputs.pre_sigmoid,
    )

    weight_deltas = model.calc_weight_deltas()

    # Precompute snapshot metadata for CI visualization
    snapshot_layers = list(alive_info.alive_counts.keys())
    snapshot_initial_alive = [alive_info.alive_counts[layer] for layer in snapshot_layers]
    snapshot_seq_len = tokens.shape[1]

    params = ci_params.get_parameters()
    optimizer = optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    progress_interval = max(1, config.steps // 20)  # Report ~20 times during optimization
    latest_loss: float = 0.0
    for step in tqdm(range(config.steps), desc="Optimizing CI values"):
        if step % progress_interval == 0:
            if on_progress is not None:
                on_progress(step, config.steps, "optimizing")

            if on_ci_snapshot is not None:
                with torch.no_grad():
                    snap_ci = ci_params.create_ci_outputs(model, device)
                    current_alive = [
                        (snap_ci.lower_leaky[layer][0] > 0.0).sum(dim=-1).tolist()
                        for layer in snapshot_layers
                    ]
                on_ci_snapshot(
                    CISnapshot(
                        step=step,
                        total_steps=config.steps,
                        layers=snapshot_layers,
                        seq_len=snapshot_seq_len,
                        initial_alive=snapshot_initial_alive,
                        current_alive=current_alive,
                        l0_total=sum(sum(row) for row in current_alive),
                        loss=latest_loss,
                    )
                )

        optimizer.zero_grad()

        ci_outputs = ci_params.create_ci_outputs(model, device)

        # Recon forward pass (stochastic or CI masking)
        match config.mask_type:
            case "stochastic":
                recon_mask_infos = calc_stochastic_component_mask_info(
                    causal_importances=ci_outputs.lower_leaky,
                    component_mask_sampling=config.sampling,
                    weight_deltas=weight_deltas,
                    router=AllLayersRouter(),
                )
            case "ci":
                recon_mask_infos = make_mask_infos(component_masks=ci_outputs.lower_leaky)

        with bf16_autocast():
            recon_out = model(tokens, mask_infos=recon_mask_infos)

        imp_min_loss = importance_minimality_loss(
            ci_upper_leaky=ci_outputs.upper_leaky,
            current_frac_of_training=step / config.steps,
            pnorm=config.imp_min_config.pnorm,
            beta=config.imp_min_config.beta,
            eps=config.imp_min_config.eps,
            p_anneal_start_frac=config.imp_min_config.p_anneal_start_frac,
            p_anneal_final_p=config.imp_min_config.p_anneal_final_p,
            p_anneal_end_frac=config.imp_min_config.p_anneal_end_frac,
            coeff_warmup_frac=config.imp_min_config.coeff_warmup_frac,
            coeff_peak_multiplier=config.imp_min_config.coeff_peak_multiplier,
            coeff_anneal_start_frac=config.imp_min_config.coeff_anneal_start_frac,
            coeff_anneal_end_frac=config.imp_min_config.coeff_anneal_end_frac,
        )

        recon_loss = compute_recon_loss(recon_out, config.loss_config, target_out, device)
        total_loss = config.loss_config.coeff * recon_loss + imp_min_coeff * imp_min_loss
        latest_loss = total_loss.item()

        # PGD adversarial loss (runs in tandem with recon)
        if config.adv_pgd is not None:
            adv_sources = run_adv_pgd(
                model=model,
                tokens=tokens,
                ci=ci_outputs.lower_leaky,
                alive_masks=alive_info.alive_masks,
                adv_config=config.adv_pgd,
                loss_config=config.loss_config,
                target_out=target_out,
            )
            pgd_mask_infos = make_mask_infos(
                interpolate_pgd_mask(ci_outputs.lower_leaky, adv_sources)
            )

            with bf16_autocast():
                pgd_out = model(tokens, mask_infos=pgd_mask_infos)

            pgd_loss = compute_recon_loss(pgd_out, config.loss_config, target_out, device)
            total_loss = total_loss + config.loss_config.coeff * pgd_loss

        total_loss.backward()
        optimizer.step()

    # Compute final metrics after optimization
    with torch.no_grad():
        final_ci_outputs = ci_params.create_ci_outputs(model, device)

        total_l0 = sum(
            calc_ci_l_zero(layer_ci, 0.0) for layer_ci in final_ci_outputs.lower_leaky.values()
        )

        final_ci_masked_label_prob: float | None = None
        final_stoch_masked_label_prob: float | None = None

        if isinstance(config.loss_config, CELossConfig | LogitLossConfig):
            pos = config.loss_config.position
            label_token = config.loss_config.label_token

            # CI-masked probability
            ci_mask_infos = make_mask_infos(final_ci_outputs.lower_leaky, routing_masks="all")
            ci_logits = model(tokens, mask_infos=ci_mask_infos)
            ci_probs = F.softmax(ci_logits[0, pos, :], dim=-1)
            final_ci_masked_label_prob = float(ci_probs[label_token].item())

            # Stochastic-masked probability (sample once for final metric)
            stoch_mask_infos = calc_stochastic_component_mask_info(
                causal_importances=final_ci_outputs.lower_leaky,
                component_mask_sampling=config.sampling,
                weight_deltas=weight_deltas,
                router=AllLayersRouter(),
            )
            stoch_logits = model(tokens, mask_infos=stoch_mask_infos)
            stoch_probs = F.softmax(stoch_logits[0, pos, :], dim=-1)
            final_stoch_masked_label_prob = float(stoch_probs[label_token].item())

    # Adversarial PGD final evaluation (needs gradients for PGD, so outside no_grad block)
    final_adv_pgd_label_prob: float | None = None

    if config.adv_pgd is not None:
        final_adv_sources = run_adv_pgd(
            model=model,
            tokens=tokens,
            ci=final_ci_outputs.lower_leaky,
            alive_masks=alive_info.alive_masks,
            adv_config=config.adv_pgd,
            target_out=target_out,
            loss_config=config.loss_config,
        )
        with torch.no_grad():
            adv_pgd_masks = make_mask_infos(
                interpolate_pgd_mask(final_ci_outputs.lower_leaky, final_adv_sources)
            )
            with bf16_autocast():
                adv_logits = model(tokens, mask_infos=adv_pgd_masks)

            if isinstance(config.loss_config, CELossConfig | LogitLossConfig):
                pos = config.loss_config.position
                label_token = config.loss_config.label_token
                adv_probs = F.softmax(adv_logits[0, pos, :], dim=-1)
                final_adv_pgd_label_prob = float(adv_probs[label_token].item())

    metrics = OptimizationMetrics(
        ci_masked_label_prob=final_ci_masked_label_prob,
        stoch_masked_label_prob=final_stoch_masked_label_prob,
        adv_pgd_label_prob=final_adv_pgd_label_prob,
        l0_total=total_l0,
    )

    return OptimizeCIResult(
        params=ci_params,
        metrics=metrics,
    )


def compute_recon_loss_batched(
    logits: Float[Tensor, "N seq vocab"],
    loss_config: LossConfig,
    target_out: Float[Tensor, "N seq vocab"],
    device: str,
) -> Float[Tensor, " N"]:
    """Compute per-element reconstruction loss for batched logits."""
    match loss_config:
        case CELossConfig(position=pos, label_token=label_token):
            labels = torch.full((logits.shape[0],), label_token, device=device)
            return F.cross_entropy(logits[:, pos, :], labels, reduction="none")
        case KLLossConfig(position=pos):
            target_probs = F.softmax(target_out[:, pos, :], dim=-1)
            pred_log_probs = F.log_softmax(logits[:, pos, :], dim=-1)
            return F.kl_div(pred_log_probs, target_probs, reduction="none").sum(dim=-1)
        case LogitLossConfig(position=pos, label_token=label_token):
            return -logits[:, pos, label_token]
        case MeanKLLossConfig():
            target_probs = F.softmax(target_out, dim=-1)
            pred_log_probs = F.log_softmax(logits, dim=-1)
            return F.kl_div(pred_log_probs, target_probs, reduction="none").sum(dim=-1).mean(dim=-1)


def importance_minimality_loss_per_element(
    ci_upper_leaky_batched: dict[str, Float[Tensor, "N seq C"]],
    n_batch: int,
    current_frac_of_training: float,
    pnorm: float,
    beta: float,
    eps: float,
    p_anneal_start_frac: float,
    p_anneal_final_p: float | None,
    p_anneal_end_frac: float,
) -> Float[Tensor, " N"]:
    """Compute importance minimality loss independently for each batch element."""
    losses = []
    for i in range(n_batch):
        element_ci = {k: v[i : i + 1] for k, v in ci_upper_leaky_batched.items()}
        losses.append(
            importance_minimality_loss(
                ci_upper_leaky=element_ci,
                current_frac_of_training=current_frac_of_training,
                pnorm=pnorm,
                beta=beta,
                eps=eps,
                p_anneal_start_frac=p_anneal_start_frac,
                p_anneal_final_p=p_anneal_final_p,
                p_anneal_end_frac=p_anneal_end_frac,
            )
        )
    return torch.stack(losses)


def run_adv_pgd_batched(
    model: ComponentModel,
    tokens: Float[Tensor, "N seq"],
    ci: dict[str, Float[Tensor, "N seq C"]],
    alive_masks: dict[str, Bool[Tensor, "N seq C"]],
    adv_config: AdvPGDConfig,
    target_out: Float[Tensor, "N seq vocab"],
    loss_config: LossConfig,
) -> dict[str, Float[Tensor, "N seq C"]]:
    """Run PGD adversary with batched tensors. Returns detached adversarial sources."""
    ci_detached = {k: v.detach() for k, v in ci.items()}

    adv_sources: dict[str, Tensor] = {}
    for layer_name, ci_val in ci_detached.items():
        source = get_pgd_init_tensor(adv_config.init, tuple(ci_val.shape), str(ci_val.device))
        source[~alive_masks[layer_name]] = 0.0
        source.requires_grad_(True)
        adv_sources[layer_name] = source

    source_list = list(adv_sources.values())

    for _ in range(adv_config.n_steps):
        mask_infos = make_mask_infos(interpolate_pgd_mask(ci_detached, adv_sources))

        with bf16_autocast():
            out = model(tokens, mask_infos=mask_infos)

        losses = compute_recon_loss_batched(out, loss_config, target_out, str(tokens.device))
        loss = losses.sum()

        grads = torch.autograd.grad(loss, source_list)
        with torch.no_grad():
            for (layer_name, source), grad in zip(adv_sources.items(), grads, strict=True):
                source.add_(adv_config.step_size * grad.sign())
                source.clamp_(0.0, 1.0)
                source[~alive_masks[layer_name]] = 0.0

    return {k: v.detach() for k, v in adv_sources.items()}


def optimize_ci_values_batched(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    configs: list[OptimCIConfig],
    device: str,
    on_progress: ProgressCallback | None = None,
    on_ci_snapshot: CISnapshotCallback | None = None,
) -> list[OptimizeCIResult]:
    """Optimize CI values for N sparsity coefficients in a single batched loop.

    All configs must share the same loss_config, steps, mask_type, adv_pgd settings —
    only imp_min_config.coeff varies between them.
    """
    N = len(configs)
    assert N > 0

    config = configs[0]
    imp_min_coeffs = torch.tensor([c.imp_min_config.coeff for c in configs], device=device)
    for c in configs:
        assert c.imp_min_config.coeff is not None

    model.requires_grad_(False)

    with torch.no_grad(), bf16_autocast():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")
        initial_ci_outputs = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=config.sampling,
            detach_inputs=False,
        )
        target_out = output_with_cache.output.detach()

    alive_info = compute_alive_info(initial_ci_outputs.lower_leaky)

    ci_params_list = [
        create_optimizable_ci_params(
            alive_info=alive_info,
            initial_pre_sigmoid=initial_ci_outputs.pre_sigmoid,
        )
        for _ in range(N)
    ]

    weight_deltas = model.calc_weight_deltas()

    all_params: list[Tensor] = []
    for ci_params in ci_params_list:
        all_params.extend(ci_params.get_parameters())

    optimizer = optim.AdamW(all_params, lr=config.lr, weight_decay=config.weight_decay)

    tokens_batched = tokens.expand(N, -1)
    target_out_batched = target_out.expand(N, -1, -1)

    snapshot_layers = list(alive_info.alive_counts.keys())
    snapshot_initial_alive = [alive_info.alive_counts[layer] for layer in snapshot_layers]
    snapshot_seq_len = tokens.shape[1]

    progress_interval = max(1, config.steps // 20)
    latest_loss = 0.0

    for step in tqdm(range(config.steps), desc="Optimizing CI values (batched)"):
        if step % progress_interval == 0:
            if on_progress is not None:
                on_progress(step, config.steps, "optimizing")

            if on_ci_snapshot is not None:
                with torch.no_grad():
                    snap_ci = ci_params_list[0].create_ci_outputs(model, device)
                    current_alive = [
                        (snap_ci.lower_leaky[layer][0] > 0.0).sum(dim=-1).tolist()
                        for layer in snapshot_layers
                    ]
                on_ci_snapshot(
                    CISnapshot(
                        step=step,
                        total_steps=config.steps,
                        layers=snapshot_layers,
                        seq_len=snapshot_seq_len,
                        initial_alive=snapshot_initial_alive,
                        current_alive=current_alive,
                        l0_total=sum(sum(row) for row in current_alive),
                        loss=latest_loss,
                    )
                )

        optimizer.zero_grad()

        ci_outputs_list = [cp.create_ci_outputs(model, device) for cp in ci_params_list]

        layers = list(ci_outputs_list[0].lower_leaky.keys())
        batched_ci_lower_leaky: dict[str, Tensor] = {
            layer: torch.cat([co.lower_leaky[layer] for co in ci_outputs_list], dim=0)
            for layer in layers
        }
        batched_ci_upper_leaky: dict[str, Tensor] = {
            layer: torch.cat([co.upper_leaky[layer] for co in ci_outputs_list], dim=0)
            for layer in layers
        }

        match config.mask_type:
            case "stochastic":
                recon_mask_infos = calc_stochastic_component_mask_info(
                    causal_importances=batched_ci_lower_leaky,
                    component_mask_sampling=config.sampling,
                    weight_deltas=weight_deltas,
                    router=AllLayersRouter(),
                )
            case "ci":
                recon_mask_infos = make_mask_infos(component_masks=batched_ci_lower_leaky)

        with bf16_autocast():
            recon_out = model(tokens_batched, mask_infos=recon_mask_infos)

        imp_min_losses = importance_minimality_loss_per_element(
            ci_upper_leaky_batched=batched_ci_upper_leaky,
            n_batch=N,
            current_frac_of_training=step / config.steps,
            pnorm=config.imp_min_config.pnorm,
            beta=config.imp_min_config.beta,
            eps=config.imp_min_config.eps,
            p_anneal_start_frac=config.imp_min_config.p_anneal_start_frac,
            p_anneal_final_p=config.imp_min_config.p_anneal_final_p,
            p_anneal_end_frac=config.imp_min_config.p_anneal_end_frac,
        )

        recon_losses = compute_recon_loss_batched(
            recon_out, config.loss_config, target_out_batched, device
        )

        loss_coeff = config.loss_config.coeff
        total_loss = (loss_coeff * recon_losses + imp_min_coeffs * imp_min_losses).sum()
        latest_loss = total_loss.item()

        if config.adv_pgd is not None:
            batched_alive_masks = {
                k: v.expand(N, -1, -1) for k, v in alive_info.alive_masks.items()
            }
            adv_sources = run_adv_pgd_batched(
                model=model,
                tokens=tokens_batched,
                ci=batched_ci_lower_leaky,
                alive_masks=batched_alive_masks,
                adv_config=config.adv_pgd,
                target_out=target_out_batched,
                loss_config=config.loss_config,
            )
            pgd_masks = interpolate_pgd_mask(batched_ci_lower_leaky, adv_sources)
            pgd_mask_infos = make_mask_infos(pgd_masks)
            with bf16_autocast():
                pgd_out = model(tokens_batched, mask_infos=pgd_mask_infos)
            pgd_losses = compute_recon_loss_batched(
                pgd_out, config.loss_config, target_out_batched, device
            )
            total_loss = total_loss + (loss_coeff * pgd_losses).sum()

        total_loss.backward()
        optimizer.step()

    # Compute final metrics per element
    results: list[OptimizeCIResult] = []
    for ci_params in ci_params_list:
        with torch.no_grad():
            final_ci = ci_params.create_ci_outputs(model, device)
            total_l0 = sum(
                calc_ci_l_zero(layer_ci, 0.0) for layer_ci in final_ci.lower_leaky.values()
            )

            ci_masked_label_prob: float | None = None
            stoch_masked_label_prob: float | None = None

            if isinstance(config.loss_config, CELossConfig | LogitLossConfig):
                pos = config.loss_config.position
                label_token = config.loss_config.label_token

                ci_mask_infos = make_mask_infos(final_ci.lower_leaky, routing_masks="all")
                ci_logits = model(tokens, mask_infos=ci_mask_infos)
                ci_probs = F.softmax(ci_logits[0, pos, :], dim=-1)
                ci_masked_label_prob = float(ci_probs[label_token].item())

                stoch_mask_infos = calc_stochastic_component_mask_info(
                    causal_importances=final_ci.lower_leaky,
                    component_mask_sampling=config.sampling,
                    weight_deltas=weight_deltas,
                    router=AllLayersRouter(),
                )
                stoch_logits = model(tokens, mask_infos=stoch_mask_infos)
                stoch_probs = F.softmax(stoch_logits[0, pos, :], dim=-1)
                stoch_masked_label_prob = float(stoch_probs[label_token].item())

        adv_pgd_label_prob: float | None = None
        if config.adv_pgd is not None:
            final_adv_sources = run_adv_pgd(
                model=model,
                tokens=tokens,
                ci=final_ci.lower_leaky,
                alive_masks=alive_info.alive_masks,
                adv_config=config.adv_pgd,
                target_out=target_out,
                loss_config=config.loss_config,
            )
            with torch.no_grad():
                adv_masks = make_mask_infos(
                    interpolate_pgd_mask(final_ci.lower_leaky, final_adv_sources)
                )
                with bf16_autocast():
                    adv_logits = model(tokens, mask_infos=adv_masks)
                if isinstance(config.loss_config, CELossConfig | LogitLossConfig):
                    pos = config.loss_config.position
                    label_token = config.loss_config.label_token
                    adv_probs = F.softmax(adv_logits[0, pos, :], dim=-1)
                    adv_pgd_label_prob = float(adv_probs[label_token].item())

        results.append(
            OptimizeCIResult(
                params=ci_params,
                metrics=OptimizationMetrics(
                    ci_masked_label_prob=ci_masked_label_prob,
                    stoch_masked_label_prob=stoch_masked_label_prob,
                    adv_pgd_label_prob=adv_pgd_label_prob,
                    l0_total=total_l0,
                ),
            )
        )

    return results


def get_out_dir() -> Path:
    """Get the output directory for optimization results."""
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
