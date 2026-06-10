"""Target-probe vs nontarget CI heatmap for targeted decomposition runs."""

from typing import ClassVar, Literal, Self, override

import torch
from pydantic import PositiveInt, model_validator
from torch import Tensor
from torch.distributed import ReduceOp
from transformers import AutoTokenizer

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import move_batch_to_device
from param_decomp.distributed import all_reduce
from param_decomp.masks import SamplingType
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.eval_metrics.plotting import plot_targeted_ci_heatmaps
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset


class TargetedCIHeatmapConfig(BaseConfig):
    """Target-probe spec: exactly one of `active_indices` (toy one-hots) or
    `prompts_file` + `tokenizer_name` (LM prompts)."""

    type: Literal["TargetedCIHeatmap"] = "TargetedCIHeatmap"
    active_indices: list[int] | None = None
    prompts_file: str | None = None
    tokenizer_name: str | None = None
    max_seq_len: PositiveInt = 512

    @model_validator(mode="after")
    def validate_probe_spec(self) -> Self:
        assert (self.active_indices is None) != (self.prompts_file is None), (
            "set exactly one of active_indices / prompts_file"
        )
        assert (self.prompts_file is None) == (self.tokenizer_name is None), (
            "tokenizer_name is required with prompts_file (and meaningless without)"
        )
        return self


class TargetedCIHeatmap(Metric[TargetedCIHeatmapConfig]):
    """One figure: a target row of per-probe CI over a nontarget row of mean CI.

    The nontarget row accumulates from the nontarget eval loop's `ctx`; the target row
    is synthesized in `compute` from the configured probes (one-hots over
    `active_indices`, or prompts tokenized once at construction).
    """

    log_namespace = "figures"
    slow = True
    short_name = "TgtCIHeatmap"
    eval_distribution = "nontarget"

    input_magnitude: ClassVar[float] = 0.75

    def __init__(self, cfg: TargetedCIHeatmapConfig) -> None:
        super().__init__(cfg)
        self.prompt_ids: Tensor | None = None
        if cfg.prompts_file is not None:
            assert cfg.tokenizer_name is not None
            tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
            self.prompt_ids = load_prompts_dataset(cfg.prompts_file, tokenizer, cfg.max_seq_len)

    @override
    def reset(self) -> None:
        self.batch_shape: tuple[int, ...] | None = None
        self.batch_is_pair = False
        self.sampling: SamplingType | None = None
        self.component_ci_sums: dict[str, Tensor] = {
            module_name: torch.zeros(self.model.module_to_c[module_name], device=self.device)
            for module_name in self.model.components
        }
        self.examples_seen = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> None:
        if self.batch_shape is None:
            self.batch_is_pair = isinstance(ctx.batch, tuple | list)
            input_tensor = ctx.batch[0] if self.batch_is_pair else ctx.batch
            self.batch_shape = tuple(input_tensor.shape)
            self.sampling = ctx.sampling
        ci_sample = next(iter(ctx.ci.lower_leaky.values()))
        self.examples_seen += ci_sample.shape[:-1].numel()
        for module_name, ci_vals in ctx.ci.lower_leaky.items():
            self.component_ci_sums[module_name] += ci_vals.detach().sum(
                dim=tuple(range(ci_vals.ndim - 1))
            )
        return None

    def _target_probe_ci(self) -> dict[str, Tensor]:
        """Per-layer `[n_probes, C]` CI on the synthesized target probes."""
        assert self.sampling is not None
        if self.prompt_ids is not None:
            probes = move_batch_to_device(self.prompt_ids, self.device)
        else:
            assert self.cfg.active_indices is not None and self.batch_shape is not None
            n_features = self.batch_shape[-1]
            assert all(0 <= i < n_features for i in self.cfg.active_indices), (
                f"TargetedCIHeatmap.active_indices must be in [0, {n_features}), "
                f"got {self.cfg.active_indices}"
            )
            probes = torch.zeros(len(self.cfg.active_indices), n_features, device=self.device)
            for row, feature_idx in enumerate(self.cfg.active_indices):
                probes[row, feature_idx] = self.input_magnitude
        # The model's run_batch may unwrap a (inputs, labels) pair — mirror the
        # structure of the batches seen in `update` so the probe survives it.
        probe_batch = (probes, probes) if self.batch_is_pair else probes
        pre_weight_acts = self.model(probe_batch, cache_type="input").cache
        ci = self.model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            detach_inputs=False,
            sampling=self.sampling,
        )
        out: dict[str, Tensor] = {}
        for module_name, ci_vals in ci.lower_leaky.items():
            if ci_vals.ndim == 3:
                ci_vals = ci_vals.mean(dim=1)
            assert ci_vals.ndim == 2
            out[module_name] = ci_vals
        return out

    @override
    def compute(self) -> MetricResult:
        assert self.batch_shape is not None, "haven't seen any inputs yet"
        examples_reduced = all_reduce(self.examples_seen, op=ReduceOp.SUM)
        nontarget_ci_mean = {
            module_name: all_reduce(self.component_ci_sums[module_name], op=ReduceOp.SUM)
            / examples_reduced
            for module_name in self.model.components
        }
        img = plot_targeted_ci_heatmaps(self._target_probe_ci(), nontarget_ci_mean)
        return {"targeted_ci_heatmap": img}
