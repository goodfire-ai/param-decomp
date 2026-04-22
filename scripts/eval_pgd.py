"""Standalone eval script to run PGDReconLoss on an existing SPD run.

Uses PGDMultiBatch with gradient accumulation for large models that don't fit
in memory with a full batch.
"""

import gc

import torch
from torch import Tensor

from spd.configs import LMTaskConfig, PGDMultiBatchReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.metrics.pgd_utils import calc_multibatch_pgd_masked_recon_loss
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import GlobalSharedTransformerCiFn
from spd.routing import AllLayersRouter
from spd.utils.general_utils import bf16_autocast


def patch_global_shared_transformer_ci_fn(training_commit: str) -> None:
    """Patch GlobalSharedTransformerCiFn to match behavior at the given training commit.

    The rms_norm on CI inputs was added after commit 28f64360. Runs trained before
    that need the original forward pass without rms_norm.
    """
    RMSNORM_ADDED_COMMIT = "23a13bf2"  # "Misc improvements: remove bf16_autocast, ..."

    def _forward_no_rmsnorm(
        self: GlobalSharedTransformerCiFn,
        input_acts: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        inputs_list = [input_acts[name] for name in self.layer_order]
        concatenated = torch.cat(inputs_list, dim=-1)
        projected = self._input_projector(concatenated)
        added_seq_dim = False
        if projected.ndim < 3:
            projected = projected.unsqueeze(-2)
            added_seq_dim = True
        x = projected
        for block in self._blocks:
            x = block(x)
        output = self._output_head(x)
        if added_seq_dim:
            output = output.squeeze(-2)
        split_outputs = torch.split(output, self.split_sizes, dim=-1)
        return {name: split_outputs[i] for i, name in enumerate(self.layer_order)}

    logger.info(
        f"Patching GlobalSharedTransformerCiFn (removing rms_norm for pre-{RMSNORM_ADDED_COMMIT} run)"
    )
    GlobalSharedTransformerCiFn.forward = _forward_no_rmsnorm  # type: ignore[assignment]


def main() -> None:
    wandb_path = "wandb:goodfire/spd/runs/s-55ea3f9b"
    configs = [
        (40, 0.1),
        (80, 0.1),
        (160, 0.1),
        (320, 0.1),
    ]

    logger.info(f"Loading SPD run from {wandb_path}")
    run_info = SPDRunInfo.from_path(wandb_path)
    config = run_info.config

    device = "cuda"
    model = ComponentModel.from_run_info(run_info)
    model.to(device)
    model.eval()

    assert isinstance(config.task_config, LMTaskConfig)

    microbatch_size = 16
    gradient_accumulation_steps = 8  # 16 * 8 = 128 effective batch size

    eval_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=config.task_config.eval_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=config.task_config.is_tokenized,
        streaming=config.task_config.streaming,
        column_name=config.task_config.column_name,
        shuffle_each_epoch=config.task_config.shuffle_each_epoch,
        seed=None,
    )

    eval_loader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=microbatch_size,
        buffer_size=config.task_config.buffer_size,
        global_seed=config.seed + 1,
    )

    def create_data_iter():
        eval_loader.generator.manual_seed(config.seed + 1)
        return iter(eval_loader)

    batch_dims = (microbatch_size, config.task_config.max_seq_len)

    with bf16_autocast(enabled=config.autocast_bf16):
        weight_deltas = model.calc_weight_deltas() if config.use_delta_component else None

        for n_steps, step_size in configs:
            pgd_config = PGDMultiBatchReconLossConfig(
                coeff=None,
                init="random",
                step_size=step_size,
                n_steps=n_steps,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

            logger.info(
                f"Running PGDMultiBatchReconLoss: n_steps={n_steps}, "
                f"step_size={step_size}, "
                f"effective_batch_size={microbatch_size * gradient_accumulation_steps}"
            )

            result = calc_multibatch_pgd_masked_recon_loss(
                pgd_config=pgd_config,
                model=model,
                weight_deltas=weight_deltas,
                create_data_iter=create_data_iter,
                output_loss_type=config.output_loss_type,
                router=AllLayersRouter(),
                sampling=config.sampling,
                use_delta_component=config.use_delta_component,
                batch_dims=batch_dims,
                device=device,
            )

            logger.info(f"n_steps={n_steps}, step_size={step_size} -> {result.item():.8f}")
            del result
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
