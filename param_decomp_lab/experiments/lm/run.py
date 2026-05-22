"""Language-model PD experiment: YAML -> `optimize()` glue.

`SavedRun` rebuilds saved LM runs by accessing this module's dispatch interface
(`TargetConfig`, `DataConfig`, `build_target`, `build_train_loader`, `build_eval_loader`,
`make_run_batch`). Run via ``pd-lm path/to/config.yaml``; multi-process (DDP) entry via
``torchrun`` of the same module.
"""

import importlib
from pathlib import Path
from typing import Any, Self

import fire
from pydantic import Field, model_validator
from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.configs import PDConfig, RuntimeConfig
from param_decomp.distributed import DistributedState, is_main_process
from param_decomp.log import logger
from param_decomp.optimize import optimize
from param_decomp_lab.batch_and_loss_fns import make_run_batch as _make_run_batch
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.distributed import (
    ensure_cached_and_call,
    get_device,
    init_distributed,
    with_distributed_cleanup,
)
from param_decomp_lab.experiments.lm.data import (
    LMDataConfig,
    collate_fn_for,
    create_lm_data_loader,
    rank_batch_size,
)
from param_decomp_lab.experiments.utils import (
    build_eval_metrics,
    load_yaml,
    run_sink_from_logging_block,
    save_run_meta,
)
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.run_files import generate_run_id
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.seed import set_seed


def _resolve_class(path: str) -> type:
    """Load a class from a string, e.g. 'transformers.LlamaForCausalLM'."""
    module_path, _, class_name = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class LMTargetConfig(BaseConfig):
    """How to load the target language model."""

    model_class: str = Field(
        ...,
        description=(
            "Fully-qualified target class name. E.g. `transformers.LlamaForCausalLM` for HF "
            "models, or `param_decomp_lab.pretrain.models.llama_simple_mlp.LlamaSimpleMLP` for "
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


TargetConfig = LMTargetConfig
DataConfig = LMDataConfig


def build_target(target_cfg: LMTargetConfig) -> Any:
    """Load the target LM from HuggingFace, a `param_decomp_lab.pretrain.*` wandb run, or a
    local/wandb checkpoint path."""
    model_class = _resolve_class(target_cfg.model_class)
    assert hasattr(model_class, "from_pretrained"), (
        f"Model class {model_class} should have a `from_pretrained` method"
    )
    if target_cfg.model_class.startswith("param_decomp_lab.pretrain.models."):
        assert target_cfg.model_name is not None, (
            "param_decomp_lab.pretrain.* targets must use `model_name` (a wandb path)"
        )
        from param_decomp_lab.pretrain.run_info import PretrainRunInfo

        run_info = ensure_cached_and_call(PretrainRunInfo.from_path, target_cfg.model_name)
        if "model_type" not in run_info.model_config_dict:
            run_info.model_config_dict["model_type"] = target_cfg.model_class.rsplit(".", 1)[-1]
        assert hasattr(model_class, "from_run_info")
        target_model = model_class.from_run_info(run_info)
    elif target_cfg.model_name is not None:
        target_model = ensure_cached_and_call(
            model_class.from_pretrained,
            target_cfg.model_name,
        )
    else:
        assert target_cfg.model_path is not None
        target_model = model_class.from_pretrained(target_cfg.model_path)
    target_model.eval()
    return target_model


def build_train_loader(
    target_cfg: LMTargetConfig,
    data_cfg: LMDataConfig,
    *,
    batch_size: int,
    device: str,
    dist_state: DistributedState | None = None,
    seed: int = 0,
) -> DataLoader[Any]:
    del target_cfg, device
    loader, _ = create_lm_data_loader(
        data_cfg,
        split=data_cfg.train_split,
        batch_size=rank_batch_size(batch_size, dist_state, label="train_batch_size"),
        seed=seed,
        dist_state=dist_state,
        collate_fn=collate_fn_for(data_cfg),
    )
    return loader


def build_eval_loader(
    target_cfg: LMTargetConfig,
    data_cfg: LMDataConfig,
    *,
    batch_size: int,
    device: str,
    dist_state: DistributedState | None = None,
    seed: int = 0,
) -> DataLoader[Any]:
    """Seed is offset by 1 so the eval split shuffles differently from the train split
    when both are constructed from the same ``pd_config.seed``."""
    del target_cfg, device
    loader, _ = create_lm_data_loader(
        data_cfg,
        split=data_cfg.eval_split,
        batch_size=rank_batch_size(batch_size, dist_state, label="eval_batch_size"),
        seed=seed + 1,
        dist_state=dist_state,
        collate_fn=collate_fn_for(data_cfg),
    )
    return loader


def make_run_batch(target_cfg: LMTargetConfig) -> RunBatch:
    return _make_run_batch(target_cfg.output_extract)


@with_distributed_cleanup
def main(config_path: str | Path) -> None:
    raw = load_yaml(config_path)
    pd_config = PDConfig.model_validate(raw["pd"])
    runtime_config = RuntimeConfig.model_validate(raw["runtime"])
    target_cfg = LMTargetConfig.model_validate(raw["target"])
    data_cfg = LMDataConfig.model_validate(raw["data"])
    logging_block = raw["logging"]

    dist_state = init_distributed()
    if is_main_process():
        logger.info(f"Distributed state: {dist_state}")
    set_seed(pd_config.seed)
    device = get_device()
    runtime_config = RuntimeConfig.model_validate({**runtime_config.model_dump(), "device": device})

    target_model = build_target(target_cfg)

    train_loader = build_train_loader(
        target_cfg,
        data_cfg,
        batch_size=pd_config.batch_size,
        device=device,
        dist_state=dist_state,
        seed=pd_config.seed,
    )
    eval_loader = build_eval_loader(
        target_cfg,
        data_cfg,
        batch_size=logging_block["eval_batch_size"],
        device=device,
        dist_state=dist_state,
        seed=pd_config.seed,
    )

    eval_metrics = build_eval_metrics(logging_block.get("eval_metrics"))

    run_id = generate_run_id("param_decomp")
    out_dir = PARAM_DECOMP_OUT_DIR / "decompositions" / run_id if is_main_process() else None
    sink = run_sink_from_logging_block(out_dir, logging_block)
    save_run_meta(
        out_dir,
        experiment_name="lm",
        pd_config=pd_config,
        runtime_config=runtime_config,
        target_dict=target_cfg.model_dump(mode="json"),
        data_dict=data_cfg.model_dump(mode="json"),
    )

    try:
        optimize(
            target_model=target_model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            run_batch=make_run_batch(target_cfg),
            reconstruction_loss=recon_loss_kl,
            pd_config=pd_config,
            runtime_config=runtime_config,
            sink=sink,
            eval_metrics=eval_metrics,
        )
    finally:
        sink.finish()


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
