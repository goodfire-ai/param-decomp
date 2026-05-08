"""Build a `PDTarget` for an LM experiment."""

from param_decomp.experiments.lm.configs import LMTargetConfig
from param_decomp.models.batch_and_loss_fns import PDTarget, make_run_batch, recon_loss_kl
from param_decomp.utils.distributed_utils import ensure_cached_and_call
from param_decomp.utils.general_utils import resolve_class


def load_lm_target(target_cfg: LMTargetConfig) -> PDTarget:
    """Load the target language model and wrap it in a `PDTarget` (KL reconstruction)."""
    assert (target_cfg.model_name is None) != (target_cfg.model_path is None), (
        "Specify exactly one of `model_name` or `model_path` on `LMTargetConfig`."
    )

    model_class = resolve_class(target_cfg.model_class)
    assert hasattr(model_class, "from_pretrained"), (
        f"Model class {model_class} should have a `from_pretrained` method"
    )

    if target_cfg.model_class.startswith("param_decomp.pretrain.models."):
        # In-repo pretrain run: cache via PretrainRunInfo, then build via from_run_info.
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
        name="lm",
    )
