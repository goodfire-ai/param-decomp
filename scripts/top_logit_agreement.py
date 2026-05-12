"""Compute top-logit agreement between CI-masked SPD model and target model.

Loads an SPD run, feeds training data through both the target model (no masks) and
the SPD model with CI masks on MLP layers only (attention layers get all-ones masks
with the delta component included), and reports how often their top predicted token
agrees.
"""

import torch
from tqdm import tqdm

from spd.data import DatasetConfig, create_data_loader
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.models.components import ComponentsMaskInfo
from spd.utils.general_utils import extract_batch_data

WANDB_PATH = "wandb:goodfire/spd/s-275c8f21"
N_SEQUENCES = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    run_info = SPDRunInfo.from_path(WANDB_PATH)
    config = run_info.config
    print(f"Config: {config.pretrained_model_class}")
    print(f"Sampling type: {config.sampling}")

    model = ComponentModel.from_run_info(run_info)
    model.to(DEVICE)
    model.eval()

    task_config = config.task_config
    train_data_config = DatasetConfig(
        name=task_config.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=task_config.train_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=task_config.is_tokenized,
        streaming=task_config.streaming,
        column_name=task_config.column_name,
        shuffle_each_epoch=False,
        seed=0,
    )

    batch_size = min(N_SEQUENCES, 16)
    n_batches = (N_SEQUENCES + batch_size - 1) // batch_size

    train_loader, _tokenizer = create_data_loader(
        dataset_config=train_data_config,
        batch_size=batch_size,
        buffer_size=10_000,
        global_seed=0,
    )

    # Precompute weight deltas for attention layers
    weight_deltas = model.calc_weight_deltas()

    attn_modules = [n for n in model.target_module_paths if ".attn." in n]
    mlp_modules = [n for n in model.target_module_paths if ".mlp." in n]
    print(f"MLP modules ({len(mlp_modules)}): {mlp_modules}")
    print(f"Attn modules ({len(attn_modules)}): {attn_modules}")

    total_positions = 0
    total_agree = 0

    data_iter = iter(train_loader)
    for batch_idx in tqdm(range(n_batches), desc="Batches"):
        batch_raw = next(data_iter)
        batch = extract_batch_data(batch_raw).to(DEVICE)

        with torch.no_grad():
            target_output: OutputWithCache = model(batch, cache_type="input")
            target_logits = target_output.output

            ci = model.calc_causal_importances(
                pre_weight_acts=target_output.cache,
                sampling="uniform",
                detach_inputs=False,
            )

            # Build mask_infos: CI masks on MLP, all-ones + delta on attn
            mask_infos: dict[str, ComponentsMaskInfo] = {}

            for name in mlp_modules:
                mask_infos[name] = ComponentsMaskInfo(
                    component_mask=ci.lower_leaky[name],
                )

            for name in attn_modules:
                ci_shape = ci.lower_leaky[name].shape
                ones_mask = torch.ones(ci_shape, device=DEVICE, dtype=ci.lower_leaky[name].dtype)
                batch_dims = ci_shape[:-1]  # (batch, seq)
                delta_mask = torch.ones(batch_dims, device=DEVICE, dtype=ones_mask.dtype)
                mask_infos[name] = ComponentsMaskInfo(
                    component_mask=ones_mask,
                    weight_delta_and_mask=(weight_deltas[name], delta_mask),
                )

            masked_logits = model(batch, mask_infos=mask_infos)

        target_top = target_logits.argmax(dim=-1)
        masked_top = masked_logits.argmax(dim=-1)

        agree = (target_top == masked_top).sum().item()
        n_pos = target_top.numel()

        total_agree += agree
        total_positions += n_pos

        print(f"  Batch {batch_idx}: {agree}/{n_pos} positions agree ({agree / n_pos:.2%})")

    overall_rate = total_agree / total_positions
    print(f"\nOverall top-logit agreement: {total_agree}/{total_positions} ({overall_rate:.2%})")


if __name__ == "__main__":
    main()
