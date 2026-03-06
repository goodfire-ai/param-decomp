"""Measure top-1 agreement between CI-masked model and target model.

For each token position, checks whether the CI-masked model's top-1 prediction
matches the target model's top-1 prediction. Also reports the mean probability
the CI-masked model assigns to the target's top-1 token.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from spd.adapters.spd import SPDAdapter
from spd.models.components import make_mask_infos
from spd.utils.general_utils import extract_batch_data

JOSE_RUN_ID = "s-55ea3f9b"
N_BATCHES = 100
BATCH_SIZE = 64


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = SPDAdapter(JOSE_RUN_ID)
    model = adapter.component_model.to(device).eval()
    dataloader = adapter.dataloader(BATCH_SIZE)

    total_tokens = 0
    total_agree = 0
    total_prob = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=N_BATCHES, desc="Evaluating")):
            if i >= N_BATCHES:
                break

            batch = extract_batch_data(batch).to(device)

            target_output = model(batch, cache_type="input")
            target_logits = target_output.output
            target_top1 = target_logits.argmax(dim=-1)

            ci = model.calc_causal_importances(
                target_output.cache, sampling="none", detach_inputs=True
            )
            mask_infos = make_mask_infos(ci.lower_leaky)
            ci_masked_logits = model(batch, mask_infos=mask_infos)

            ci_masked_top1 = ci_masked_logits.argmax(dim=-1)
            agree = (ci_masked_top1 == target_top1).sum().item()

            ci_masked_probs = F.softmax(ci_masked_logits, dim=-1)
            prob_of_target_top1 = ci_masked_probs.gather(-1, target_top1.unsqueeze(-1)).squeeze(-1)

            n_tokens = target_top1.numel()
            total_tokens += n_tokens
            total_agree += agree
            total_prob += prob_of_target_top1.sum().item()

            if (i + 1) % 10 == 0:
                running_agreement = total_agree / total_tokens
                running_prob = total_prob / total_tokens
                tqdm.write(
                    f"  Batch {i + 1}: "
                    f"agreement={running_agreement:.4f}, "
                    f"mean_prob={running_prob:.4f}"
                )

    agreement = total_agree / total_tokens
    mean_prob = total_prob / total_tokens
    print(f"\nResults over {N_BATCHES} batches ({total_tokens:,} tokens):")
    print(f"  Top-1 agreement: {agreement:.4f} ({agreement * 100:.2f}%)")
    print(f"  Mean prob of target top-1: {mean_prob:.4f}")


if __name__ == "__main__":
    main()
