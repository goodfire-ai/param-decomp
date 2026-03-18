"""Model comparison script for geometric similarity analysis.

Compares two SPD models by computing geometric similarities between their learned
subcomponents. Designed for post-hoc analysis of completed runs.

Usage:
    python spd/scripts/compare_models/compare_models.py spd/scripts/compare_models/compare_models_config.yaml
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import fire
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import Field
from torch import Tensor

matplotlib.use("Agg")

from spd.base_config import BaseConfig
from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data, get_obj_device
from spd.utils.run_utils import save_file

METRIC_PREFIXES = [
    ("rank1", "Rank-1 (V@U)"),
    ("u", "U vectors"),
    ("v", "V vectors"),
    ("ci", "CI profiles"),
]
STATS = ("mean", "std", "min", "max")

# Per-layer sim matrices keyed by metric prefix: {layer_name: {prefix: Tensor[C_curr_alive, C_ref]}}
SimMatrices = dict[str, dict[str, Tensor]]


def model_id_from_path(path: str) -> str:
    return path.rstrip("/").split("/")[-1]


def max_match_stats(
    sim_matrix: Float[Tensor, "C_curr C_ref"],
) -> tuple[float, float, float, float]:
    """Compute mean/std/min/max of per-row max similarities."""
    max_sim = sim_matrix.max(dim=1).values
    return (
        max_sim.mean().item(),
        max_sim.std().item(),
        max_sim.min().item(),
        max_sim.max().item(),
    )


def collect_batches(eval_iterator: Iterator[Any], n_steps: int) -> list[Any]:
    batches: list[Any] = []
    for step in range(n_steps):
        try:
            batch = extract_batch_data(next(eval_iterator))
        except StopIteration:
            assert step > 0, "Evaluation iterator provided no batches"
            logger.warning(
                "Evaluation iterator exhausted after %s steps (requested %s).", step, n_steps
            )
            break
        batches.append(batch)
    return batches


class CompareModelsConfig(BaseConfig):
    current_model_path: str
    reference_model_path: str
    mean_ci_threshold: float = Field(..., ge=0.0, le=1.0)
    n_eval_steps: int
    eval_batch_size: int
    shuffle_data: bool
    output_dir: str | None = None


class ModelComparator:
    """Compare two SPD models for geometric similarity between subcomponents."""

    def __init__(self, config: CompareModelsConfig):
        self.config = config
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
        run_info = SPDRunInfo.from_path(model_path)
        model = ComponentModel.from_run_info(run_info)
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)
        return model, run_info.config

    def create_eval_data_loader(self) -> Iterator[Any]:
        task_name = self.current_config.task_config.task_name

        data_loader_fns: dict[str, Callable[[], Iterator[Any]]] = {
            "tms": self._create_tms_data_loader,
            "resid_mlp": self._create_resid_mlp_data_loader,
            "lm": self._create_lm_data_loader,
            "ih": self._create_ih_data_loader,
        }

        assert task_name in data_loader_fns, (
            f"Unsupported task type: {task_name}. Supported: {', '.join(data_loader_fns)}"
        )
        return data_loader_fns[task_name]()

    def _create_tms_data_loader(self) -> Iterator[Any]:
        from spd.configs import TMSTaskConfig
        from spd.experiments.tms.models import TMSTargetRunInfo
        from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset

        assert isinstance(self.current_config.task_config, TMSTaskConfig)
        task_config = self.current_config.task_config
        assert self.current_config.pretrained_model_path
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
                dataset, batch_size=self.config.eval_batch_size, shuffle=self.config.shuffle_data
            )
        )

    def _create_resid_mlp_data_loader(self) -> Iterator[Any]:
        from spd.configs import ResidMLPTaskConfig
        from spd.experiments.resid_mlp.models import ResidMLPTargetRunInfo
        from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
        from spd.utils.data_utils import DatasetGeneratedDataLoader

        assert isinstance(self.current_config.task_config, ResidMLPTaskConfig)
        task_config = self.current_config.task_config
        assert self.current_config.pretrained_model_path
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
                dataset, batch_size=self.config.eval_batch_size, shuffle=self.config.shuffle_data
            )
        )

    def _create_lm_data_loader(self) -> Iterator[Any]:
        from spd.configs import LMTaskConfig
        from spd.data import DatasetConfig, create_data_loader

        assert self.current_config.tokenizer_name
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
        from spd.configs import IHTaskConfig
        from spd.experiments.ih.model import InductionModelTargetRunInfo
        from spd.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset

        assert isinstance(self.current_config.task_config, IHTaskConfig)
        task_config = self.current_config.task_config
        assert self.current_config.pretrained_model_path
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
                dataset, batch_size=self.config.eval_batch_size, shuffle=self.config.shuffle_data
            )
        )

    def compute_ci_statistics(
        self, batches: list[Any]
    ) -> tuple[dict[str, Float[Tensor, " C"]], dict[str, Tensor]]:
        """Compute mean CI values and CI cosine similarity matrices between the two models.

        Returns:
            mean_cis: Mean CI per component in the current model (for alive filtering).
            ci_cosine_matrices: Per-module [C_curr, C_ref] cosine similarity of CI profiles.
        """
        assert batches, "No evaluation batches provided"
        device = get_obj_device(self.current_model)

        # Per-module accumulators for current model's mean CI
        ci_sums: dict[str, Float[Tensor, " C"]] = {}
        n_examples: dict[str, float] = {}

        # Per-module accumulators for cross-model CI cosine similarity
        cross_dots: dict[str, Tensor] = {}
        curr_sq_sums: dict[str, Float[Tensor, " C"]] = {}
        ref_sq_sums: dict[str, Tensor] = {}

        for module_name, current_module in self.current_model.components.items():
            c_curr = current_module.C
            ci_sums[module_name] = torch.zeros(c_curr, device=device)
            n_examples[module_name] = 0.0
            curr_sq_sums[module_name] = torch.zeros(c_curr, device=device)

            ref_module = self.reference_model.components.get(module_name)
            if ref_module is not None:
                cross_dots[module_name] = torch.zeros(c_curr, ref_module.C, device=device)
                ref_sq_sums[module_name] = torch.zeros(ref_module.C, device=device)

        self.current_model.eval()
        self.reference_model.eval()

        with torch.no_grad():
            for batch in batches:
                batch = batch.to(self.device)

                ci_current = self.current_model.calc_causal_importances(
                    self.current_model(batch, cache_type="input").cache,
                    sampling=self.current_config.sampling,
                ).lower_leaky

                ci_reference = self.reference_model.calc_causal_importances(
                    self.reference_model(batch, cache_type="input").cache,
                    sampling=self.reference_config.sampling,
                ).lower_leaky

                for module_name, ci_curr in ci_current.items():
                    ci_curr_fp32 = ci_curr.to(device=device, dtype=torch.float32)
                    batch_dims = tuple(range(ci_curr_fp32.ndim - 1))

                    ci_sums[module_name] += ci_curr_fp32.sum(dim=batch_dims)
                    n_examples[module_name] += float(ci_curr_fp32.shape[:-1].numel())

                    if module_name not in cross_dots or module_name not in ci_reference:
                        continue

                    ci_ref = ci_reference[module_name]
                    assert ci_curr.shape == ci_ref.shape, (
                        f"Shape mismatch for {module_name}: {ci_curr.shape} vs {ci_ref.shape}"
                    )
                    ci_ref_fp32 = ci_ref.to(device=device, dtype=torch.float32)

                    # Flatten batch dims for dot product accumulation
                    curr_flat = ci_curr_fp32.reshape(-1, ci_curr_fp32.shape[-1])
                    ref_flat = ci_ref_fp32.reshape(-1, ci_ref_fp32.shape[-1])

                    cross_dots[module_name] += curr_flat.T @ ref_flat
                    curr_sq_sums[module_name] += curr_flat.square().sum(dim=0)
                    ref_sq_sums[module_name] += ref_flat.square().sum(dim=0)

        mean_cis = {name: ci_sums[name] / max(n_examples[name], 1.0) for name in ci_sums}

        eps = 1e-12
        ci_cosine_matrices: dict[str, Tensor] = {}
        for module_name, dots in cross_dots.items():
            curr_norm = torch.sqrt(curr_sq_sums[module_name]).clamp_min(eps)
            ref_norm = torch.sqrt(ref_sq_sums[module_name]).clamp_min(eps)
            ci_cosine_matrices[module_name] = dots / torch.outer(curr_norm, ref_norm)

        return mean_cis, ci_cosine_matrices

    def compute_geometric_similarities(
        self,
        mean_cis: dict[str, Float[Tensor, " C"]],
        ci_cosine_matrices: dict[str, Tensor],
    ) -> tuple[dict[str, float], SimMatrices]:
        """Compute per-layer similarity metrics between the two models' components.

        Returns:
            similarities: Scalar summary stats keyed by '{prefix}_cosine_{stat}/{layer}'.
            matrices: Raw sim matrices keyed by layer_name -> prefix -> Tensor[C_alive, C_ref].
        """
        similarities: dict[str, float] = {}
        matrices: SimMatrices = {}

        for layer_name in self.current_model.components:
            assert layer_name in self.reference_model.components, (
                f"Layer {layer_name} not in reference model"
            )

            current = self.current_model.components[layer_name]
            reference = self.reference_model.components[layer_name]

            alive_mask = mean_cis[layer_name] > self.config.mean_ci_threshold
            n_alive = int(alive_mask.sum().item())
            logger.info(
                f"Layer {layer_name}: {n_alive} components above "
                f"mean CI threshold {self.config.mean_ci_threshold}"
            )
            if n_alive == 0:
                logger.warning(f"No alive components in {layer_name}. Skipping.")
                continue

            # Parameter cosine similarities (factored rank-1 decomposition)
            curr_U_norm = F.normalize(current.U[alive_mask], p=2, dim=1)
            curr_V_norm = F.normalize(current.V[:, alive_mask], p=2, dim=0)
            ref_U_norm = F.normalize(reference.U, p=2, dim=1)
            ref_V_norm = F.normalize(reference.V, p=2, dim=0)

            u_sim = curr_U_norm @ ref_U_norm.T
            v_sim = curr_V_norm.T @ ref_V_norm
            rank1_sim = (u_sim * v_sim).abs()

            # CI cosine similarities
            assert layer_name in ci_cosine_matrices
            ci_cos_matrix = ci_cosine_matrices[layer_name]
            assert ci_cos_matrix.shape[0] == alive_mask.shape[0]
            ci_cos_alive = ci_cos_matrix[alive_mask]

            layer_matrices = {
                "rank1": rank1_sim,
                "u": u_sim.abs(),
                "v": v_sim.abs(),
                "ci": ci_cos_alive,
            }
            matrices[layer_name] = layer_matrices

            for prefix, matrix in layer_matrices.items():
                mean, std, min_val, max_val = max_match_stats(matrix)
                similarities[f"{prefix}_cosine_mean/{layer_name}"] = mean
                similarities[f"{prefix}_cosine_std/{layer_name}"] = std
                similarities[f"{prefix}_cosine_min/{layer_name}"] = min_val
                similarities[f"{prefix}_cosine_max/{layer_name}"] = max_val

        # Aggregate across layers
        all_metric_names = [
            f"{prefix}_cosine_{stat}" for prefix, _ in METRIC_PREFIXES for stat in STATS
        ]
        for metric_name in all_metric_names:
            values = [
                similarities[f"{metric_name}/{layer_name}"]
                for layer_name in self.current_model.components
                if f"{metric_name}/{layer_name}" in similarities
            ]
            if values:
                similarities[f"{metric_name}/all_layers"] = sum(values) / len(values)

        return similarities, matrices

    def run_comparison(self, eval_iterator: Iterator[Any]) -> tuple[dict[str, float], SimMatrices]:
        batches = collect_batches(eval_iterator, self.config.n_eval_steps)

        logger.info("Computing causal importance statistics for current and reference models...")
        mean_cis, ci_cosine_matrices = self.compute_ci_statistics(batches)

        logger.info("Computing geometric similarities...")
        return self.compute_geometric_similarities(mean_cis, ci_cosine_matrices)


def format_results_markdown(similarities: dict[str, float], config: CompareModelsConfig) -> str:
    lines: list[str] = []
    lines.append("# Model Comparison Results\n")
    lines.append(f"- **Current model**: `{config.current_model_path}`")
    lines.append(f"- **Reference model**: `{config.reference_model_path}`")
    lines.append(f"- **Mean CI threshold**: {config.mean_ci_threshold}")
    lines.append(f"- **Eval steps**: {config.n_eval_steps}")
    lines.append(f"- **Batch size**: {config.eval_batch_size}\n")

    layer_names = _extract_layer_names(similarities)

    lines.append("## Summary (all layers)\n")
    lines.extend(_metric_summary_table(similarities, "all_layers"))
    lines.append("")

    lines.append("## Per-layer breakdown\n")
    for prefix, label in METRIC_PREFIXES:
        lines.append(f"### {label}\n")
        lines.extend(_per_layer_table(similarities, prefix, layer_names))
        lines.append("")

    return "\n".join(lines)


def _extract_layer_names(results: dict[str, float]) -> list[str]:
    layer_names: list[str] = []
    for key in results:
        if "/" not in key:
            continue
        layer = key.split("/", 1)[1]
        if layer != "all_layers" and layer not in layer_names:
            layer_names.append(layer)
    return layer_names


def _metric_summary_table(results: dict[str, float], scope: str) -> list[str]:
    lines = [
        "| Metric | Mean | Std | Min | Max |",
        "|--------|-----:|----:|----:|----:|",
    ]
    for prefix, label in METRIC_PREFIXES:
        vals = [results.get(f"{prefix}_cosine_{s}/{scope}") for s in STATS]
        if vals[0] is not None:
            lines.append(
                f"| {label} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} |"
            )
    return lines


def _per_layer_table(results: dict[str, float], prefix: str, layer_names: list[str]) -> list[str]:
    lines = [
        "| Layer | Mean | Std | Min | Max |",
        "|-------|-----:|----:|----:|----:|",
    ]
    for layer in layer_names:
        vals = [results.get(f"{prefix}_cosine_{s}/{layer}") for s in STATS]
        if vals[0] is not None:
            lines.append(
                f"| {layer} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} |"
            )
    return lines


def save_heatmaps(matrices: SimMatrices, output_dir: Path) -> None:
    """Save cosine similarity matrices as heatmap images."""
    heatmap_dir = output_dir / "heatmaps"
    for layer_name, layer_matrices in matrices.items():
        # Dots in layer names (e.g. h.0.mlp.c_fc) are fine in filenames
        for prefix, matrix in layer_matrices.items():
            prefix_dir = heatmap_dir / prefix
            prefix_dir.mkdir(parents=True, exist_ok=True)

            data = matrix.detach().cpu().float().numpy()

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax)

            _, label = next((pfx, lbl) for pfx, lbl in METRIC_PREFIXES if pfx == prefix)
            ax.set_title(f"{label} — {layer_name}")
            ax.set_xlabel("Reference component")
            ax.set_ylabel("Current component (alive)")

            fig.tight_layout()
            fig.savefig(prefix_dir / f"{layer_name}.png", dpi=100)
            plt.close(fig)

    logger.info(f"Saved heatmaps to {heatmap_dir}")


def resolve_output_dir(config_output_dir: str | None) -> Path:
    if config_output_dir is None:
        return Path(__file__).parent / "out"
    return Path(config_output_dir)


def run_and_save_pair(
    config: CompareModelsConfig, pair_dir: Path
) -> tuple[dict[str, float], SimMatrices]:
    """Run a pairwise comparison, save results/heatmaps to pair_dir, return results."""
    pair_dir.mkdir(parents=True, exist_ok=True)

    comparator = ModelComparator(config)
    eval_iterator = comparator.create_eval_data_loader()
    similarities, matrices = comparator.run_comparison(eval_iterator)

    save_file(similarities, pair_dir / "results.json")
    (pair_dir / "results.md").write_text(format_results_markdown(similarities, config))
    save_heatmaps(matrices, pair_dir)

    return similarities, matrices


def main(config_path: Path | str) -> None:
    config = CompareModelsConfig.from_file(config_path)
    output_dir = resolve_output_dir(config.output_dir)

    current_id = model_id_from_path(config.current_model_path)
    reference_id = model_id_from_path(config.reference_model_path)
    pair_dir = output_dir / f"{current_id}_vs_{reference_id}"

    similarities, _ = run_and_save_pair(config, pair_dir)

    logger.info(f"Comparison complete! Results saved to {pair_dir}")
    for key, value in similarities.items():
        logger.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
