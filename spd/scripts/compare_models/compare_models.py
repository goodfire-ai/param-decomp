"""Model comparison script for geometric similarity analysis.

This script compares two SPD models by computing geometric similarities between
their learned subcomponents. It's designed for post-hoc analysis of completed runs.

Usage:
    python spd/scripts/compare_models/compare_models.py spd/scripts/compare_models/compare_models_config.yaml
    python spd/scripts/compare_models/compare_models.py --current_model_path="wandb:..." --reference_model_path="wandb:..."
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import einops
import fire
import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import Field
from torch import Tensor

from spd.base_config import BaseConfig
from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data, get_obj_device
from spd.utils.run_utils import save_file


class CompareModelsConfig(BaseConfig):
    """Configuration for model comparison script."""

    current_model_path: str = Field(..., description="Path to current model (wandb: or local path)")
    reference_model_path: str = Field(
        ..., description="Path to reference model (wandb: or local path)"
    )

    mean_ci_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum mean causal importance for components to be included in comparison",
    )
    n_eval_steps: int = Field(
        ..., description="Number of evaluation steps to compute mean causal importances"
    )

    eval_batch_size: int = Field(..., description="Batch size for evaluation data loading")
    shuffle_data: bool = Field(..., description="Whether to shuffle the evaluation data")
    ci_alive_threshold: float = Field(
        ..., description="Threshold for considering components as 'alive'"
    )

    output_dir: str | None = Field(
        default=None,
        description="Directory to save results (defaults to 'out' directory relative to script location)",
    )


class ModelComparator:
    """Compare two SPD models for geometric similarity between subcomponents."""

    def __init__(self, config: CompareModelsConfig):
        """Initialize the model comparator.

        Args:
            config: CompareModelsConfig instance containing all configuration parameters
        """
        self.config = config
        self.mean_ci_threshold = config.mean_ci_threshold
        self.device = get_device()

        logger.info(f"Loading current model from: {config.current_model_path}")
        self.current_model, self.current_config = self._load_model_and_config(
            config.current_model_path
        )

        logger.info(f"Loading reference model from: {config.reference_model_path}")
        self.reference_model, self.reference_config = self._load_model_and_config(
            config.reference_model_path
        )

    def _load_model_and_config(self, model_path: str) -> tuple[ComponentModel, Config]:
        """Load model and config using the standard pattern from existing codebase."""
        run_info = SPDRunInfo.from_path(model_path)
        model = ComponentModel.from_run_info(run_info)
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        return model, run_info.config

    def create_eval_data_loader(self) -> Iterator[Any]:
        """Create evaluation data loader using exact same patterns as decomposition scripts."""
        task_name = self.current_config.task_config.task_name

        data_loader_fns: dict[str, Callable[[], Iterator[Any]]] = {
            "tms": self._create_tms_data_loader,
            "resid_mlp": self._create_resid_mlp_data_loader,
            "lm": self._create_lm_data_loader,
            "ih": self._create_ih_data_loader,
        }

        if task_name not in data_loader_fns:
            raise ValueError(
                f"Unsupported task type: {task_name}. Supported types: {', '.join(data_loader_fns.keys())}"
            )

        return data_loader_fns[task_name]()

    def _create_tms_data_loader(self) -> Iterator[Any]:
        """Create data loader for TMS task."""
        from spd.configs import TMSTaskConfig
        from spd.experiments.tms.models import TMSTargetRunInfo
        from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset

        assert isinstance(self.current_config.task_config, TMSTaskConfig)
        task_config = self.current_config.task_config

        assert self.current_config.pretrained_model_path, (
            "pretrained_model_path must be set for TMS models"
        )

        target_run_info = TMSTargetRunInfo.from_path(self.current_config.pretrained_model_path)

        dataset = SparseFeatureDataset(
            n_features=target_run_info.config.tms_model_config.n_features,
            feature_probability=task_config.feature_probability,
            device=self.device,
            data_generation_type=task_config.data_generation_type,
            value_range=(0.0, 1.0),
            synced_inputs=target_run_info.config.synced_inputs,
        )
        return iter(
            DatasetGeneratedDataLoader(
                dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=self.config.shuffle_data,
            )
        )

    def _create_resid_mlp_data_loader(self) -> Iterator[Any]:
        """Create data loader for ResidMLP task."""
        from spd.configs import ResidMLPTaskConfig
        from spd.experiments.resid_mlp.models import ResidMLPTargetRunInfo
        from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
        from spd.utils.data_utils import DatasetGeneratedDataLoader

        assert isinstance(self.current_config.task_config, ResidMLPTaskConfig)
        task_config = self.current_config.task_config

        assert self.current_config.pretrained_model_path, (
            "pretrained_model_path must be set for ResidMLP models"
        )

        target_run_info = ResidMLPTargetRunInfo.from_path(self.current_config.pretrained_model_path)

        dataset = ResidMLPDataset(
            n_features=target_run_info.config.resid_mlp_model_config.n_features,
            feature_probability=task_config.feature_probability,
            device=self.device,
            calc_labels=False,
            label_type=None,
            act_fn_name=None,
            label_fn_seed=None,
            synced_inputs=target_run_info.config.synced_inputs,
        )
        return iter(
            DatasetGeneratedDataLoader(
                dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=self.config.shuffle_data,
            )
        )

    def _create_lm_data_loader(self) -> Iterator[Any]:
        """Create data loader for LM task."""
        from spd.configs import LMTaskConfig
        from spd.data import DatasetConfig, create_data_loader

        assert self.current_config.tokenizer_name, "tokenizer_name must be set"
        assert isinstance(self.current_config.task_config, LMTaskConfig)
        task_config = self.current_config.task_config

        dataset_config = DatasetConfig(
            name=task_config.dataset_name,
            hf_tokenizer_path=self.current_config.tokenizer_name,
            split=task_config.eval_data_split,
            n_ctx=task_config.max_seq_len,
            is_tokenized=task_config.is_tokenized,
            streaming=task_config.streaming,
            column_name=task_config.column_name,
            shuffle_each_epoch=task_config.shuffle_each_epoch,
            seed=None,
        )
        loader, _ = create_data_loader(
            dataset_config=dataset_config,
            batch_size=self.config.eval_batch_size,
            buffer_size=task_config.buffer_size,
            global_seed=self.current_config.seed + 1,
        )
        return iter(loader)

    def _create_ih_data_loader(self) -> Iterator[Any]:
        """Create data loader for IH task."""
        from spd.configs import IHTaskConfig
        from spd.experiments.ih.model import InductionModelTargetRunInfo
        from spd.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset

        assert isinstance(self.current_config.task_config, IHTaskConfig)
        task_config = self.current_config.task_config

        assert self.current_config.pretrained_model_path, (
            "pretrained_model_path must be set for Induction Head models"
        )

        target_run_info = InductionModelTargetRunInfo.from_path(
            self.current_config.pretrained_model_path
        )

        dataset = InductionDataset(
            vocab_size=target_run_info.config.ih_model_config.vocab_size,
            seq_len=target_run_info.config.ih_model_config.seq_len,
            prefix_window=task_config.prefix_window
            or target_run_info.config.ih_model_config.seq_len - 3,
            device=self.device,
        )
        return iter(
            DatasetGeneratedDataLoader(
                dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=self.config.shuffle_data,
            )
        )

    def compute_ci_statistics(
        self, batches: list[Any]
    ) -> tuple[dict[str, Float[Tensor, " C"]], dict[str, Tensor]]:
        """Compute mean causal importances and cosine similarity matrices per component."""

        if not batches:
            raise ValueError("No evaluation batches provided for CI statistics computation.")

        device = get_obj_device(self.current_model)

        component_ci_sums: dict[str, Float[Tensor, " C"]] = {}
        component_example_counts: dict[str, Tensor] = {}
        ci_cross_dot_products: dict[str, Tensor] = {}
        ci_current_sq_sums: dict[str, Float[Tensor, " C"]] = {}
        ci_reference_sq_sums: dict[str, Tensor] = {}

        for module_name, current_module in self.current_model.components.items():
            component_dim_current = current_module.C
            component_ci_sums[module_name] = torch.zeros(component_dim_current, device=device)
            component_example_counts[module_name] = torch.tensor(0.0, device=device)
            ci_current_sq_sums[module_name] = torch.zeros(component_dim_current, device=device)

            reference_module = self.reference_model.components.get(module_name)
            if reference_module is not None:
                ci_cross_dot_products[module_name] = torch.zeros(
                    component_dim_current, reference_module.C, device=device
                )
                ci_reference_sq_sums[module_name] = torch.zeros(reference_module.C, device=device)

        self.current_model.eval()
        self.reference_model.eval()

        with torch.no_grad():
            for batch in batches:
                batch = batch.to(self.device)

                pre_weight_current = self.current_model(batch, cache_type="input").cache
                ci_current = self.current_model.calc_causal_importances(
                    pre_weight_current,
                    sampling=self.current_config.sampling,
                ).lower_leaky

                pre_weight_reference = self.reference_model(batch, cache_type="input").cache
                ci_reference = self.reference_model.calc_causal_importances(
                    pre_weight_reference,
                    sampling=self.reference_config.sampling,
                ).lower_leaky

                for module_name, ci_vals_current in ci_current.items():
                    ci_vals_current_fp32 = ci_vals_current.to(device=device, dtype=torch.float32)

                    n_leading_dims = ci_vals_current_fp32.ndim - 1
                    leading_dim_idxs = tuple(range(n_leading_dims))
                    n_examples = float(ci_vals_current_fp32.shape[:n_leading_dims].numel())

                    component_ci_sums[module_name] += ci_vals_current_fp32.sum(dim=leading_dim_idxs)
                    component_example_counts[module_name] += n_examples

                    if module_name not in ci_cross_dot_products:
                        continue

                    if module_name not in ci_reference:
                        logger.warning(
                            "Module %s not found in reference CI outputs. Skipping cosine similarity.",
                            module_name,
                        )
                        continue

                    ci_vals_reference = ci_reference[module_name]
                    if ci_vals_current.shape != ci_vals_reference.shape:
                        logger.warning(
                            "Shape mismatch for module %s between current and reference CI outputs "
                            "(%s vs %s). Skipping cosine similarity.",
                            module_name,
                            ci_vals_current.shape,
                            ci_vals_reference.shape,
                        )
                        continue

                    ci_vals_reference_fp32 = ci_vals_reference.to(
                        device=device, dtype=torch.float32
                    )

                    ci_current_flat = ci_vals_current_fp32.reshape(
                        -1, ci_vals_current_fp32.shape[-1]
                    )
                    ci_reference_flat = ci_vals_reference_fp32.reshape(
                        -1, ci_vals_reference_fp32.shape[-1]
                    )

                    ci_cross_dot_products[module_name] += (
                        ci_current_flat.transpose(0, 1) @ ci_reference_flat
                    )
                    ci_current_sq_sums[module_name] += (ci_current_flat.square()).sum(dim=0)
                    ci_reference_sq_sums[module_name] += (ci_reference_flat.square()).sum(dim=0)

        mean_component_cis = {
            module_name: component_ci_sums[module_name]
            / component_example_counts[module_name].clamp_min(1.0)
            for module_name in component_ci_sums
        }

        ci_cosine_matrices: dict[str, Tensor] = {}
        eps = 1e-12
        for module_name, dot_products in ci_cross_dot_products.items():
            current_norm = torch.sqrt(ci_current_sq_sums[module_name]).clamp_min(eps)
            reference_norm = torch.sqrt(ci_reference_sq_sums[module_name]).clamp_min(eps)
            denom = torch.outer(current_norm, reference_norm)
            cos_matrix = torch.zeros_like(dot_products)
            nonzero_mask = denom > 0
            cos_matrix[nonzero_mask] = dot_products[nonzero_mask] / denom[nonzero_mask]
            ci_cosine_matrices[module_name] = cos_matrix

        return mean_component_cis, ci_cosine_matrices

    def compute_geometric_similarities(
        self,
        mean_component_cis: dict[str, Float[Tensor, " C"]],
        ci_cosine_similarities: dict[str, Tensor],
    ) -> dict[str, float]:
        """Compute geometric similarities between subcomponents."""
        similarities = {}

        for layer_name in self.current_model.components:
            if layer_name not in self.reference_model.components:
                logger.warning(f"Layer {layer_name} not found in reference model, skipping")
                continue

            current_components = self.current_model.components[layer_name]
            reference_components = self.reference_model.components[layer_name]

            # Extract U and V matrices
            C_ref = reference_components.C
            current_U = current_components.U  # Shape: [C, d_out]
            current_V = current_components.V  # Shape: [d_in, C]
            ref_U = reference_components.U
            ref_V = reference_components.V

            # Filter out components that aren't active enough in the current model
            alive_mask = mean_component_cis[layer_name] > self.mean_ci_threshold
            C_curr_alive = int(alive_mask.sum().item())
            logger.info(
                f"Layer {layer_name}: {C_curr_alive} components above mean CI threshold "
                f"{self.mean_ci_threshold}"
            )
            if C_curr_alive == 0:
                logger.warning(
                    f"No components meet the mean CI threshold {self.mean_ci_threshold} in {layer_name}. Skipping."
                )
                continue

            current_U_alive = current_U[alive_mask]
            current_V_alive = current_V[:, alive_mask]

            # Compute rank-one matrices: V @ U for each component
            current_rank_one = einops.einsum(
                current_V_alive,
                current_U_alive,
                "d_in C_curr_alive, C_curr_alive d_out -> C_curr_alive d_in d_out",
            )
            ref_rank_one = einops.einsum(
                ref_V, ref_U, "d_in C_ref, C_ref d_out -> C_ref d_in d_out"
            )

            # Compute cosine similarities between all pairs
            current_flat = current_rank_one.reshape(C_curr_alive, -1)
            ref_flat = ref_rank_one.reshape(C_ref, -1)

            current_norm = F.normalize(current_flat, p=2, dim=1)
            ref_norm = F.normalize(ref_flat, p=2, dim=1)

            cosine_sim_matrix = einops.einsum(
                current_norm,
                ref_norm,
                "C_curr_alive d_in_d_out, C_ref d_in_d_out -> C_curr_alive C_ref",
            )
            cosine_sim_matrix = cosine_sim_matrix.abs()

            max_similarities = cosine_sim_matrix.max(dim=1).values
            similarities[f"mean_max_abs_cosine_sim/{layer_name}"] = max_similarities.mean().item()
            similarities[f"max_abs_cosine_sim_std/{layer_name}"] = max_similarities.std().item()
            similarities[f"max_abs_cosine_sim_min/{layer_name}"] = max_similarities.min().item()
            similarities[f"max_abs_cosine_sim_max/{layer_name}"] = max_similarities.max().item()

            if layer_name in ci_cosine_similarities:
                ci_cos_matrix = ci_cosine_similarities[layer_name]
                if ci_cos_matrix.shape[0] != alive_mask.shape[0]:
                    logger.warning(
                        "Mismatch between CI cosine matrix rows (%s) and component count (%s) for %s.",
                        ci_cos_matrix.shape[0],
                        alive_mask.shape[0],
                        layer_name,
                    )
                else:
                    ci_cos_alive = ci_cos_matrix[alive_mask]
                    if ci_cos_alive.numel() > 0:
                        ci_cos_max = ci_cos_alive.max(dim=1).values
                        similarities[f"ci_cosine_mean/{layer_name}"] = ci_cos_max.mean().item()
                        similarities[f"ci_cosine_std/{layer_name}"] = ci_cos_max.std(
                            unbiased=False
                        ).item()
                        similarities[f"ci_cosine_min/{layer_name}"] = ci_cos_max.min().item()
                        similarities[f"ci_cosine_max/{layer_name}"] = ci_cos_max.max().item()

        metric_names = [
            "mean_max_abs_cosine_sim",
            "max_abs_cosine_sim_std",
            "max_abs_cosine_sim_min",
            "max_abs_cosine_sim_max",
        ]

        cosine_metric_names = [
            "ci_cosine_mean",
            "ci_cosine_std",
            "ci_cosine_min",
            "ci_cosine_max",
        ]

        for metric_name in metric_names + cosine_metric_names:
            values = [
                similarities[f"{metric_name}/{layer_name}"]
                for layer_name in self.current_model.components
                if f"{metric_name}/{layer_name}" in similarities
            ]
            if values:
                similarities[f"{metric_name}/all_layers"] = sum(values) / len(values)

        return similarities

    def run_comparison(
        self, eval_iterator: Iterator[Any], n_steps: int | None = None
    ) -> dict[str, float]:
        """Run the full comparison pipeline."""
        if n_steps is None:
            n_steps = self.config.n_eval_steps
        assert isinstance(n_steps, int)

        batches: list[Any] = []
        for step in range(n_steps):
            try:
                batch = extract_batch_data(next(eval_iterator))
            except StopIteration:
                if step == 0:
                    raise ValueError("Evaluation iterator provided no batches.") from None
                logger.warning(
                    "Evaluation iterator exhausted after %s steps (requested %s).",
                    step,
                    n_steps,
                )
                break
            batches.append(batch)

        logger.info("Computing causal importance statistics for current and reference models...")
        mean_component_cis, ci_cosine_similarities = self.compute_ci_statistics(batches)

        logger.info("Computing geometric similarities...")
        similarities = self.compute_geometric_similarities(
            mean_component_cis, ci_cosine_similarities
        )

        return similarities


def main(config_path: Path | str) -> None:
    """Main execution function.

    Args:
        config_path: Path to YAML config
    """
    config = CompareModelsConfig.from_file(config_path)

    if config.output_dir is None:
        output_dir = Path(__file__).parent / "out"
    else:
        output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparator = ModelComparator(config)

    logger.info("Setting up evaluation data...")
    eval_iterator = comparator.create_eval_data_loader()

    logger.info("Starting model comparison...")
    similarities = comparator.run_comparison(eval_iterator)

    results_file = output_dir / "similarity_results.json"
    save_file(similarities, results_file)

    logger.info(f"Comparison complete! Results saved to {results_file}")
    logger.info("Similarity metrics:")
    for key, value in similarities.items():
        logger.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
