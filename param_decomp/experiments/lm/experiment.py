"""Language-model PD experiment.

This file is the runtime definition for the LM experiment: serializable config,
target loading, and driver registration.
"""

from pathlib import Path
from typing import Any, Self, override

from pydantic import Field, model_validator
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.experiments.driver import ExperimentConfig, ExperimentDriver, PreparedExperiment
from param_decomp.experiments.lm.data import LMDataConfig, build_lm_dataloaders
from param_decomp.models.batch_and_loss_fns import PDTarget, make_run_batch, recon_loss_kl
from param_decomp.param_decomp_types import ModelPath
from param_decomp.utils.distributed_utils import DistributedState, ensure_cached_and_call
from param_decomp.utils.general_utils import resolve_class


class LMTargetConfig(BaseConfig):
    """How to load the target language model."""

    model_class: str = Field(
        ...,
        description=(
            "Fully-qualified target class name. E.g. `transformers.LlamaForCausalLM` for HF "
            "models, or `param_decomp.pretrain.models.llama_simple_mlp.LlamaSimpleMLP` for "
            "an in-repo pretrain run."
        ),
    )
    model_name: str | None = Field(
        default=None,
        description=(
            "HF model id (e.g. `openai-community/gpt2`) or an in-repo pretrain run path "
            "(e.g. `goodfire/spd/runs/<id>`). Required for HF / pretrain models; mutually "
            "exclusive with `model_path`."
        ),
    )
    model_path: ModelPath | None = Field(
        default=None,
        description=(
            "Local or wandb path for `LoadableModule` subclasses. Mutually exclusive with "
            "`model_name`."
        ),
    )
    output_extract: int | str | None = Field(
        default="logits",
        description=(
            "How to extract the prediction tensor from the model output. None = raw output, "
            "int = index into output tuple, str = attribute name."
        ),
    )

    @model_validator(mode="after")
    def validate_location(self) -> Self:
        assert (self.model_name is None) != (self.model_path is None), (
            "Specify exactly one of `model_name` or `model_path`."
        )
        return self


class LMExperimentConfig(ExperimentConfig):
    target: LMTargetConfig
    data: LMDataConfig


def load_lm_target(target_cfg: LMTargetConfig) -> PDTarget:
    """Load the target language model and wrap it in a KL reconstruction target."""
    assert (target_cfg.model_name is None) != (target_cfg.model_path is None), (
        "Specify exactly one of `model_name` or `model_path` on `LMTargetConfig`."
    )

    model_class = resolve_class(target_cfg.model_class)
    assert hasattr(model_class, "from_pretrained"), (
        f"Model class {model_class} should have a `from_pretrained` method"
    )

    if target_cfg.model_class.startswith("param_decomp.pretrain.models."):
        assert target_cfg.model_name is not None, (
            "param_decomp.pretrain.* targets must use `model_name` (a wandb path)"
        )
        from param_decomp.pretrain.run_info import PretrainRunInfo

        run_info = ensure_cached_and_call(PretrainRunInfo.from_path, target_cfg.model_name)
        if "model_type" not in run_info.model_config_dict:
            run_info.model_config_dict["model_type"] = target_cfg.model_class.rsplit(".", 1)[-1]
        assert hasattr(model_class, "from_run_info")
        target_model = model_class.from_run_info(run_info)  # pyright: ignore[reportAttributeAccessIssue]
    elif target_cfg.model_name is not None:
        target_model = ensure_cached_and_call(
            model_class.from_pretrained,  # pyright: ignore[reportAttributeAccessIssue]
            target_cfg.model_name,
        )
    else:
        assert target_cfg.model_path is not None
        target_model = model_class.from_pretrained(target_cfg.model_path)  # pyright: ignore[reportAttributeAccessIssue]

    target_model.eval()
    return PDTarget(
        model=target_model,
        run_batch=make_run_batch(target_cfg.output_extract),
        reconstruction_loss=recon_loss_kl,
    )


class Driver(ExperimentDriver[LMExperimentConfig]):
    @property
    @override
    def kind(self) -> str:
        return "lm"

    @property
    @override
    def config_model(self) -> type[LMExperimentConfig]:
        return LMExperimentConfig

    @override
    def prepare(
        self,
        experiment_config: LMExperimentConfig,
        *,
        device: str,
        dist_state: DistributedState | None = None,
    ) -> PreparedExperiment:
        target = self.load_target(experiment_config)
        train_loader, eval_loader = self.build_dataloaders(
            experiment_config,
            train_batch_size=experiment_config.pd.batch_size,
            eval_batch_size=experiment_config.pd.eval_batch_size,
            dist_state=dist_state,
            device=device,
        )
        return PreparedExperiment(
            pd=experiment_config.pd,
            target=target,
            train_loader=train_loader,
            eval_loader=eval_loader,
        )

    @override
    def load_target(
        self, experiment_config: LMExperimentConfig, *, run_dir: Path | None = None
    ) -> PDTarget:
        _ = run_dir
        return load_lm_target(experiment_config.target)

    @override
    def build_dataloaders(
        self,
        experiment_config: LMExperimentConfig,
        *,
        train_batch_size: int,
        eval_batch_size: int,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
        run_dir: Path | None = None,
    ) -> tuple[DataLoader[Any], DataLoader[Any]]:
        _ = device, run_dir
        return build_lm_dataloaders(
            experiment_config.data,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            dist_state=dist_state,
        )

    @override
    def display_name(self, experiment_config: LMExperimentConfig) -> str:
        return (
            f"LM: {experiment_config.target.model_class.rsplit('.', 1)[-1]} "
            f"on {experiment_config.data.dataset_name}"
        )
