"""Language-model PD experiment: serializable config, target loading, and driver."""

from typing import Any, ClassVar, Self, override

from pydantic import Field, model_validator

from param_decomp import ExperimentDriver
from param_decomp.base_config import BaseConfig
from param_decomp.experiments.lm.data import (
    LMDataConfig,
    build_lm_eval_loader,
    build_lm_train_loader,
)
from param_decomp.models.batch_and_loss_fns import PDTarget, make_run_batch, recon_loss_kl
from param_decomp.run import RunConfig
from param_decomp.types import ModelPath
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


class LMRunConfig(RunConfig):
    target: LMTargetConfig
    data: LMDataConfig


def _load_target_model(target_cfg: LMTargetConfig) -> Any:
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
        return model_class.from_run_info(run_info)  # pyright: ignore[reportAttributeAccessIssue]
    if target_cfg.model_name is not None:
        return ensure_cached_and_call(
            model_class.from_pretrained,  # pyright: ignore[reportAttributeAccessIssue]
            target_cfg.model_name,
        )
    assert target_cfg.model_path is not None
    return model_class.from_pretrained(target_cfg.model_path)  # pyright: ignore[reportAttributeAccessIssue]


class Driver(ExperimentDriver[LMRunConfig]):
    name: ClassVar[str] = "lm"

    @property
    @override
    def config_type(self) -> type[LMRunConfig]:
        return LMRunConfig

    @override
    def build_target(self, run_cfg: LMRunConfig) -> PDTarget:
        target_model = _load_target_model(run_cfg.target)
        target_model.eval()
        return PDTarget(
            model=target_model,
            run_batch=make_run_batch(run_cfg.target.output_extract),
            reconstruction_loss=recon_loss_kl,
        )

    @override
    def build_train_loader(
        self,
        run_cfg: LMRunConfig,
        *,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> Any:
        _ = device
        return build_lm_train_loader(
            data_cfg=run_cfg.data,
            batch_size=batch_size_override or run_cfg.pd.batch_size,
            seed=run_cfg.pd.seed,
            dist_state=dist_state,
        )

    @override
    def build_eval_loader(
        self,
        run_cfg: LMRunConfig,
        *,
        batch_size_override: int | None = None,
        dist_state: DistributedState | None = None,
        device: str = "cpu",
    ) -> Any:
        _ = device
        return build_lm_eval_loader(
            data_cfg=run_cfg.data,
            batch_size=batch_size_override or run_cfg.logging.eval_batch_size,
            seed=run_cfg.pd.seed,
            dist_state=dist_state,
        )
