"""Pairwise KL between three forward passes:
  A) Base target model (no hooks)
  B) Ones-masked + weight delta, under bf16_autocast
  C) Ones-masked + weight delta, pure f32

Model weights are f32 throughout."""

import torch
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.general_utils import bf16_autocast

RUN = "wandb:goodfire/spd/s-55ea3f9b"
DEVICE = "cuda"
N_TOKENS = 128
N_BATCHES = 10
SEED = 42


def get_test_tokens(model: ComponentModel, n_tokens: int, n_batches: int) -> list[Tensor]:
    vocab_size = model.target_model.get_submodule("lm_head").weight.shape[0]
    rng = torch.Generator().manual_seed(SEED)
    return [torch.randint(0, vocab_size, (1, n_tokens), generator=rng) for _ in range(n_batches)]


def mean_kl(p_logits: Tensor, q_logits: Tensor) -> float:
    p = torch.softmax(p_logits.float(), dim=-1)
    q = torch.softmax(q_logits.float(), dim=-1)
    return (p * (p.log() - q.log())).sum(dim=-1).mean().item()


def ones_masked_forward(model: ComponentModel, tokens: Tensor) -> Tensor:
    weight_deltas = model.calc_weight_deltas()
    weight_deltas_and_masks = {
        k: (v.to(tokens.device), torch.ones(tokens.shape, device=tokens.device))
        for k, v in weight_deltas.items()
    }
    component_masks = {
        name: torch.ones((*tokens.shape, comp.C), device=tokens.device)
        for name, comp in model.components.items()
    }
    mask_infos = make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
    )
    out = model(tokens, mask_infos=mask_infos)
    assert isinstance(out, Tensor)
    return out


def main() -> None:
    model = ComponentModel.from_pretrained(RUN)
    model = model.to(DEVICE).to(torch.float32)
    model.eval()
    tokens_list = get_test_tokens(model, N_TOKENS, N_BATCHES)

    kls: dict[str, list[float]] = {
        "A↔B (base vs autocast ones+delta)": [],
        "A↔C (base vs f32 ones+delta)": [],
        "B↔C (autocast ones+delta vs f32 ones+delta)": [],
    }

    print(f"{'Batch':<6} {'A↔B':>10} {'A↔C':>10} {'B↔C':>10}")
    print("-" * 40)

    for i, tokens in enumerate(tokens_list):
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            a = model(tokens)  # base target, no hooks

            with bf16_autocast():
                b = ones_masked_forward(model, tokens)  # ones+delta, autocast

            c = ones_masked_forward(model, tokens)  # ones+delta, f32

        ab = mean_kl(a, b)
        ac = mean_kl(a, c)
        bc = mean_kl(b, c)

        kls["A↔B (base vs autocast ones+delta)"].append(ab)
        kls["A↔C (base vs f32 ones+delta)"].append(ac)
        kls["B↔C (autocast ones+delta vs f32 ones+delta)"].append(bc)

        print(f"{i:<6} {ab:>10.2e} {ac:>10.2e} {bc:>10.2e}")

    print(f"\n{'=' * 40}")
    print("Mean over batches:")
    for label, vals in kls.items():
        print(f"  {label}: {sum(vals) / len(vals):.2e}")


if __name__ == "__main__":
    main()
