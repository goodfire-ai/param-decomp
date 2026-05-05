"""Model comparison script for geometric similarity analysis.

This script compares two PD models by computing geometric similarities between
their learned subcomponents. It's designed for post-hoc analysis of completed runs.

Usage:
    python param_decomp/scripts/compare_models/compare_models.py param_decomp/scripts/compare_models/compare_models_config.yaml
    python param_decomp/scripts/compare_models/compare_models.py --current_model_path="wandb:..." --reference_model_path="wandb:..."
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import einops
import fire
import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import Field
from torch import Tensor

from param_decomp.base_config import BaseConfig
from param_decomp.configs import Config
from param_decomp.log import logger
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.utils.distributed_utils import get_device
from param_decomp.utils.general_utils import get_obj_device
from param_decomp.utils.run_utils import save_file


class CompareModelsConfig(BaseConfig):
    """Configuration for model comparison script."""

    current_model_path: str = Field(..., description="Path to current model (wandb: or local path)")
    reference_model_path: str = Field(
        ..., description="Path to reference model (wandb: or local path)"
    )

    density_threshold: float = Field(
        ..., description="Minimum activation density for components to be included in comparison"
    )
    n_eval_steps: int = Field(
        ..., description="Number of evaluation steps to compute activation densities"
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
    """Compare two PD models for geometric similarity between subcomponents."""

    def __init__(self, config: CompareModelsConfig):
        """Initialize the model comparator.

        Args:
            config: CompareModelsConfig instance containing all configuration parameters
        """
        self.config = config
        self.density_threshold = config.density_threshold
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
        run_info = ParamDecompRunInfo.from_path(model_path)
        model = ComponentModel.from_run_info(run_info)
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        return model, run_info.config

    def create_eval_data_loader(self) -> Iterator[Tensor]:
        """Create evaluation data loader using exact same patterns as decomposition scripts.

        Each per-task loader yields input tensors directly so downstream code can treat
        batches uniformly regardless of task type.
        """
        task_name = self.current_config.task_config.task_name

        data_loader_fns: dict[str, Callable[[], Iterator[Tensor]]] = {
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

    def _create_tms_data_loader(self) -> Iterator[Tensor]:
        """Create data loader for TMS task."""
        from param_decomp.configs import TMSTaskConfig
        from param_decomp.experiments.tms.models import TMSTargetRunInfo
        from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset

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
        loader = DatasetGeneratedDataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=self.config.shuffle_data,
        )
        return (batch[0] for batch in loader)

    def _create_resid_mlp_data_loader(self) -> Iterator[Tensor]:
        """Create data loader for ResidMLP task."""
        from param_decomp.configs import ResidMLPTaskConfig
        from param_decomp.experiments.resid_mlp.models import ResidMLPTargetRunInfo
        from param_decomp.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
        from param_decomp.utils.data_utils import DatasetGeneratedDataLoader

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
        loader = DatasetGeneratedDataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=self.config.shuffle_data,
        )
        return (batch[0] for batch in loader)

    def _create_lm_data_loader(self) -> Iterator[Tensor]:
        """Create data loader for LM task."""
        from param_decomp.configs import LMTaskConfig
        from param_decomp.data import DatasetConfig, create_data_loader, input_ids_collate_fn

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
            collate_fn=input_ids_collate_fn,
        )
        return iter(loader)

    def _create_ih_data_loader(self) -> Iterator[Tensor]:
        """Create data loader for IH task."""
        from param_decomp.configs import IHTaskConfig
        from param_decomp.experiments.ih.model import InductionModelTargetRunInfo
        from param_decomp.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset

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
        loader = DatasetGeneratedDataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=self.config.shuffle_data,
        )
        return (batch[0] for batch in loader)

    def compute_activation_densities(
        self, model: ComponentModel, eval_iterator: Iterator[Tensor], n_steps: int
    ) -> dict[str, Float[Tensor, " C"]]:
        """Compute activation densities using same logic as ComponentActivationDensity."""

        model_config = self.current_config if model is self.current_model else self.reference_config
        ci_alive_threshold = self.config.ci_alive_threshold

        device = get_obj_device(model)
        n_tokens = 0
        component_activation_counts: dict[str, Float[Tensor, " C"]] = {
            module_name: torch.zeros(model.module_to_c[module_name], device=device)
            for module_name in model.components
        }

        model.eval()
        with torch.no_grad():
            for _step in range(n_steps):
                batch = next(eval_iterator).to(self.device)
                pre_weight_acts = model(batch, cache_type="input").cache

                ci = model.calc_causal_importances(
                    pre_weight_acts,
                    sampling=model_config.sampling,
                ).lower_leaky

                n_tokens_batch = next(iter(ci.values())).shape[:-1].numel()
                n_tokens += n_tokens_batch

                for module_name, ci_vals in ci.items():
                    active_components = ci_vals > ci_alive_threshold
                    n_activations_per_component = einops.reduce(
                        active_components, "... C -> C", "sum"
                    )
                    component_activation_counts[module_name] += n_activations_per_component

        densities = {
            module_name: component_activation_counts[module_name] / n_tokens
            for module_name in model.components
        }

        return densities

    def compute_geometric_similarities(
        self, activation_densities: dict[str, Float[Tensor, " C"]]
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
            alive_mask = activation_densities[layer_name] > self.config.density_threshold
            C_curr_alive = int(alive_mask.sum().item())
            logger.info(f"Number of active components in {layer_name}: {C_curr_alive}")
            if C_curr_alive == 0:
                logger.warning(
                    f"No components are active enough in {layer_name} for density threshold {self.config.density_threshold}. Skipping."
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

        metric_names = [
            "mean_max_abs_cosine_sim",
            "max_abs_cosine_sim_std",
            "max_abs_cosine_sim_min",
            "max_abs_cosine_sim_max",
        ]

        for metric_name in metric_names:
            values = [
                similarities[f"{metric_name}/{layer_name}"]
                for layer_name in self.current_model.components
                if f"{metric_name}/{layer_name}" in similarities
            ]
            if values:
                similarities[f"{metric_name}/all_layers"] = sum(values) / len(values)

        return similarities

    def run_comparison(
        self, eval_iterator: Iterator[Tensor], n_steps: int | None = None
    ) -> dict[str, float]:
        """Run the full comparison pipeline."""
        if n_steps is None:
            n_steps = self.config.n_eval_steps
        assert isinstance(n_steps, int)

        logger.info("Computing activation densities for current model...")
        activation_densities = self.compute_activation_densities(
            self.current_model, eval_iterator, n_steps
        )

        logger.info("Computing geometric similarities...")
        similarities = self.compute_geometric_similarities(activation_densities)

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
