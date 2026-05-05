"""Evaluation utilities using the new Metric classes."""

from collections.abc import Iterator
from typing import Any

from PIL import Image
from torch import Tensor
from torch.types import Number
from wandb.plot.custom_chart import CustomChart

from param_decomp.configs import (
    CEandKLLossesConfig,
    CI_L0Config,
    CIHiddenActsReconLossConfig,
    CIHistogramsConfig,
    CIMaskedAttnPatternsReconLossConfig,
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    CIMeanPerComponentConfig,
    ComponentActivationDensityConfig,
    Config,
    FaithfulnessLossConfig,
    IdentityCIErrorConfig,
    ImportanceMinimalityLossConfig,
    MetricConfigType,
    PermutedCIPlotsConfig,
    PersistentPGDReconEvalConfig,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetEvalConfig,
    PersistentPGDReconSubsetLossConfig,
    PGDMultiBatchReconLossConfig,
    PGDMultiBatchReconSubsetLossConfig,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    StochasticAttnPatternsReconLossConfig,
    StochasticHiddenActsReconLossConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetCEAndKLConfig,
    StochasticReconSubsetLossConfig,
    UnmaskedReconLossConfig,
    UVPlotsConfig,
)
from param_decomp.metrics.attn_patterns_recon_loss import (
    CIMaskedAttnPatternsReconLoss,
    StochasticAttnPatternsReconLoss,
)
from param_decomp.metrics.base import Metric
from param_decomp.metrics.ce_and_kl_losses import CEandKLLosses
from param_decomp.metrics.ci_histograms import CIHistograms
from param_decomp.metrics.ci_l0 import CI_L0
from param_decomp.metrics.ci_masked_recon_layerwise_loss import CIMaskedReconLayerwiseLoss
from param_decomp.metrics.ci_masked_recon_loss import CIMaskedReconLoss
from param_decomp.metrics.ci_masked_recon_subset_loss import CIMaskedReconSubsetLoss
from param_decomp.metrics.ci_mean_per_component import CIMeanPerComponent
from param_decomp.metrics.component_activation_density import ComponentActivationDensity
from param_decomp.metrics.faithfulness_loss import FaithfulnessLoss
from param_decomp.metrics.hidden_acts_recon_loss import (
    CIHiddenActsReconLoss,
    StochasticHiddenActsReconLoss,
)
from param_decomp.metrics.identity_ci_error import IdentityCIError
from param_decomp.metrics.importance_minimality_loss import ImportanceMinimalityLoss
from param_decomp.metrics.permuted_ci_plots import PermutedCIPlots
from param_decomp.metrics.pgd_masked_recon_layerwise_loss import PGDReconLayerwiseLoss
from param_decomp.metrics.pgd_masked_recon_loss import PGDReconLoss
from param_decomp.metrics.pgd_masked_recon_subset_loss import PGDReconSubsetLoss
from param_decomp.metrics.pgd_utils import CreateDataIter, calc_multibatch_pgd_masked_recon_loss
from param_decomp.metrics.ppgd_eval_losses import PPGDReconEval
from param_decomp.metrics.stochastic_recon_layerwise_loss import StochasticReconLayerwiseLoss
from param_decomp.metrics.stochastic_recon_loss import StochasticReconLoss
from param_decomp.metrics.stochastic_recon_subset_ce_and_kl import StochasticReconSubsetCEAndKL
from param_decomp.metrics.stochastic_recon_subset_loss import StochasticReconSubsetLoss
from param_decomp.metrics.unmasked_recon_loss import UnmaskedReconLoss
from param_decomp.metrics.uv_plots import UVPlots
from param_decomp.models.batch_and_loss_fns import ReconstructionLoss
from param_decomp.models.component_model import (
    ComponentModel,
    OutputWithCache,
    move_batch_to_device,
)
from param_decomp.persistent_pgd import PersistentPGDState
from param_decomp.routing import AllLayersRouter, get_subset_router
from param_decomp.utils.distributed_utils import avg_metrics_across_ranks, is_distributed
from param_decomp.utils.general_utils import dict_safe_update_

MetricOutType = dict[str, str | Number | Image.Image | CustomChart]
DistMetricOutType = dict[str, str | float | Image.Image | CustomChart]


def clean_metric_output(
    section: str,
    metric_name: str,
    computed_raw: Any,
) -> MetricOutType:
    """Clean metric output by converting tensors to floats/ints and ensuring the correct types.

    Expects outputs to be either a scalar tensor or a mapping of strings to scalars/images/tensors.
    """
    computed: MetricOutType = {}
    assert isinstance(computed_raw, dict | Tensor), f"{type(computed_raw)} not supported"
    if isinstance(computed_raw, Tensor):
        assert computed_raw.numel() == 1, (
            f"Only scalar tensors supported, got shape {computed_raw.shape}"
        )
        item = computed_raw.item()
        computed[f"{section}/{metric_name}"] = item
    else:
        for k, v in computed_raw.items():
            assert isinstance(k, str), f"Only supports string keys, got {type(k)}"
            assert isinstance(v, str | Number | Image.Image | CustomChart | Tensor), (
                f"{type(v)} not supported"
            )
            if isinstance(v, Tensor):
                v = v.item()

            computed[f"{section}/{k}"] = v
    return computed


def avg_eval_metrics_across_ranks(metrics: MetricOutType, device: str) -> DistMetricOutType:
    """Get the average of eval metrics across ranks.

    Ignores any metrics that are not numbers. Currently, the image metrics do not need to be
    averaged. If this changes for future metrics, we will need to do a reduce during calculcation
    of the metric.
    """
    assert is_distributed(), "Can only average metrics across ranks if running in distributed mode"
    metrics_keys_to_avg = {k: v for k, v in metrics.items() if isinstance(v, Number)}
    if metrics_keys_to_avg:
        avg_metrics = avg_metrics_across_ranks(metrics_keys_to_avg, device)
    else:
        avg_metrics = {}
    return {**metrics, **avg_metrics}


def init_metric(
    cfg: MetricConfigType,
    model: ComponentModel,
    run_config: Config,
    device: str,
    reconstruction_loss: ReconstructionLoss,
    ppgd_states: dict[
        PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig, PersistentPGDState
    ],
) -> Metric:
    match cfg:
        case ImportanceMinimalityLossConfig():
            metric = ImportanceMinimalityLoss(
                model=model,
                device=device,
                pnorm=cfg.pnorm,
                beta=cfg.beta,
                p_anneal_start_frac=cfg.p_anneal_start_frac,
                p_anneal_final_p=cfg.p_anneal_final_p,
                p_anneal_end_frac=cfg.p_anneal_end_frac,
            )
        case FaithfulnessLossConfig():
            metric = FaithfulnessLoss(
                model=model,
                device=device,
            )
        case CEandKLLossesConfig():
            metric = CEandKLLosses(
                model=model,
                device=device,
                sampling=run_config.sampling,
                rounding_threshold=cfg.rounding_threshold,
            )
        case CIHistogramsConfig():
            metric = CIHistograms(model=model, n_batches_accum=cfg.n_batches_accum)
        case CI_L0Config():
            metric = CI_L0(
                model=model,
                device=device,
                ci_alive_threshold=run_config.ci_alive_threshold,
                groups=cfg.groups,
            )
        case CIMaskedReconSubsetLossConfig():
            metric = CIMaskedReconSubsetLoss(
                model=model,
                device=device,
                routing=cfg.routing,
                reconstruction_loss=reconstruction_loss,
            )
        case CIMaskedReconLayerwiseLossConfig():
            metric = CIMaskedReconLayerwiseLoss(
                model=model,
                device=device,
                reconstruction_loss=reconstruction_loss,
            )
        case CIMaskedReconLossConfig():
            metric = CIMaskedReconLoss(
                model=model,
                device=device,
                reconstruction_loss=reconstruction_loss,
            )
        case CIMeanPerComponentConfig():
            metric = CIMeanPerComponent(model=model, device=device)
        case ComponentActivationDensityConfig():
            metric = ComponentActivationDensity(
                model=model, device=device, ci_alive_threshold=run_config.ci_alive_threshold
            )
        case IdentityCIErrorConfig():
            metric = IdentityCIError(
                model=model,
                sampling=run_config.sampling,
                identity_ci=cfg.identity_ci,
                dense_ci=cfg.dense_ci,
            )
        case PermutedCIPlotsConfig():
            metric = PermutedCIPlots(
                model=model,
                sampling=run_config.sampling,
                identity_patterns=cfg.identity_patterns,
                dense_patterns=cfg.dense_patterns,
            )
        case StochasticReconLayerwiseLossConfig():
            metric = StochasticReconLayerwiseLoss(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                reconstruction_loss=reconstruction_loss,
            )
        case StochasticReconLossConfig():
            metric = StochasticReconLoss(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                reconstruction_loss=reconstruction_loss,
            )
        case StochasticReconSubsetLossConfig():
            metric = StochasticReconSubsetLoss(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                routing=cfg.routing,
                reconstruction_loss=reconstruction_loss,
            )
        case PGDReconLossConfig():
            metric = PGDReconLoss(
                model=model,
                device=device,
                use_delta_component=run_config.use_delta_component,
                pgd_config=cfg,
                reconstruction_loss=reconstruction_loss,
            )
        case PGDReconSubsetLossConfig():
            metric = PGDReconSubsetLoss(
                model=model,
                device=device,
                use_delta_component=run_config.use_delta_component,
                pgd_config=cfg,
                routing=cfg.routing,
                reconstruction_loss=reconstruction_loss,
            )
        case PGDReconLayerwiseLossConfig():
            metric = PGDReconLayerwiseLoss(
                model=model,
                device=device,
                use_delta_component=run_config.use_delta_component,
                pgd_config=cfg,
                reconstruction_loss=reconstruction_loss,
            )
        case StochasticReconSubsetCEAndKLConfig():
            metric = StochasticReconSubsetCEAndKL(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                include_patterns=cfg.include_patterns,
                exclude_patterns=cfg.exclude_patterns,
            )
        case StochasticHiddenActsReconLossConfig():
            metric = StochasticHiddenActsReconLoss(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
            )
        case CIHiddenActsReconLossConfig():
            metric = CIHiddenActsReconLoss(model=model, device=device)
        case PersistentPGDReconEvalConfig():
            matching = [
                s for k, s in ppgd_states.items() if isinstance(k, PersistentPGDReconLossConfig)
            ]
            assert len(matching) == 1
            metric = PPGDReconEval(
                model=model,
                device=device,
                ppgd_state=matching[0],
                use_delta_component=run_config.use_delta_component,
                reconstruction_loss=reconstruction_loss,
                metric_name=cfg.classname,
            )
        case PersistentPGDReconSubsetEvalConfig():
            matching = [
                s
                for k, s in ppgd_states.items()
                if isinstance(k, PersistentPGDReconSubsetLossConfig)
            ]
            assert len(matching) == 1
            metric = PPGDReconEval(
                model=model,
                device=device,
                ppgd_state=matching[0],
                use_delta_component=run_config.use_delta_component,
                reconstruction_loss=reconstruction_loss,
                metric_name=cfg.classname,
            )
        case CIMaskedAttnPatternsReconLossConfig():
            metric = CIMaskedAttnPatternsReconLoss(
                model=model,
                device=device,
                n_heads=cfg.n_heads,
                q_proj_path=cfg.q_proj_path,
                k_proj_path=cfg.k_proj_path,
                c_attn_path=cfg.c_attn_path,
            )
        case StochasticAttnPatternsReconLossConfig():
            metric = StochasticAttnPatternsReconLoss(
                model=model,
                device=device,
                sampling=run_config.sampling,
                use_delta_component=run_config.use_delta_component,
                n_mask_samples=run_config.n_mask_samples,
                n_heads=cfg.n_heads,
                q_proj_path=cfg.q_proj_path,
                k_proj_path=cfg.k_proj_path,
                c_attn_path=cfg.c_attn_path,
            )
        case UVPlotsConfig():
            metric = UVPlots(
                model=model,
                sampling=run_config.sampling,
                identity_patterns=cfg.identity_patterns,
                dense_patterns=cfg.dense_patterns,
            )
        case UnmaskedReconLossConfig():
            metric = UnmaskedReconLoss(
                model=model,
                device=device,
                reconstruction_loss=reconstruction_loss,
            )
        case (
            PGDMultiBatchReconLossConfig()
            | PGDMultiBatchReconSubsetLossConfig()
            | PersistentPGDReconLossConfig()
            | PersistentPGDReconSubsetLossConfig()
        ):
            raise ValueError(f"Unsupported metric config for eval: {cfg}")
    return metric


def evaluate(
    eval_metric_configs: list[MetricConfigType],
    model: ComponentModel,
    eval_iterator: Iterator[Any],
    device: str,
    run_config: Config,
    slow_step: bool,
    n_eval_steps: int,
    current_frac_of_training: float,
    reconstruction_loss: ReconstructionLoss,
    ppgd_states: dict[
        PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig, PersistentPGDState
    ],
) -> MetricOutType:
    """Run evaluation and return a mapping of metric names to values/images."""

    # Persistent PGD losses are training-only (sources are coupled to train batch size)
    eval_metric_configs = [
        cfg
        for cfg in eval_metric_configs
        if not isinstance(cfg, PersistentPGDReconLossConfig | PersistentPGDReconSubsetLossConfig)
    ]

    metrics: list[Metric] = []
    for cfg in eval_metric_configs:
        metric = init_metric(
            cfg=cfg,
            model=model,
            run_config=run_config,
            device=device,
            reconstruction_loss=reconstruction_loss,
            ppgd_states=ppgd_states,
        )
        if metric.slow and not slow_step:
            continue
        metrics.append(metric)

    # Weight deltas can be computed once per eval since params are frozen
    weight_deltas = model.calc_weight_deltas()

    for _ in range(n_eval_steps):
        batch = move_batch_to_device(next(eval_iterator), device)

        target_output: OutputWithCache = model(batch, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=target_output.cache,
            detach_inputs=False,
            sampling=run_config.sampling,
        )

        for metric in metrics:
            metric.update(
                batch=batch,
                target_out=target_output.output,
                pre_weight_acts=target_output.cache,
                ci=ci,
                current_frac_of_training=current_frac_of_training,
                weight_deltas=weight_deltas,
            )

    outputs: MetricOutType = {}
    for metric in metrics:
        computed_raw: Any = metric.compute()
        computed = clean_metric_output(
            section=metric.metric_section,
            metric_name=type(metric).__name__,
            computed_raw=computed_raw,
        )
        dict_safe_update_(outputs, computed)

    return outputs


def evaluate_multibatch_pgd(
    multibatch_pgd_eval_configs: list[
        PGDMultiBatchReconLossConfig | PGDMultiBatchReconSubsetLossConfig
    ],
    model: ComponentModel,
    create_data_iter: CreateDataIter,
    config: Config,
    device: str,
    reconstruction_loss: ReconstructionLoss,
) -> dict[str, float]:
    """Calculate multibatch PGD metrics."""
    weight_deltas = model.calc_weight_deltas() if config.use_delta_component else None

    metrics: dict[str, float] = {}
    for multibatch_pgd_config in multibatch_pgd_eval_configs:
        match multibatch_pgd_config:
            case PGDMultiBatchReconLossConfig():
                router = AllLayersRouter()
            case PGDMultiBatchReconSubsetLossConfig():
                router = get_subset_router(multibatch_pgd_config.routing, device)

        assert multibatch_pgd_config.classname not in metrics, (
            f"Metric {multibatch_pgd_config.classname} already exists"
        )

        metrics[multibatch_pgd_config.classname] = calc_multibatch_pgd_masked_recon_loss(
            pgd_config=multibatch_pgd_config,
            model=model,
            weight_deltas=weight_deltas,
            create_data_iter=create_data_iter,
            router=router,
            sampling=config.sampling,
            use_delta_component=config.use_delta_component,
            device=device,
            reconstruction_loss=reconstruction_loss,
        ).item()
    return metrics
