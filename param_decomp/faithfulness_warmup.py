"""Pre-training phase that optimizes only the faithfulness loss.

Run once before the main optimization loop when `pd_config.faithfulness_warmup_steps > 0`.
Drives the component weights toward a faithful decomposition of the target weights before
the full loss objective takes over.
"""

import gc

import torch
from torch import optim

from param_decomp.component_model import ComponentModel
from param_decomp.configs import PDConfig
from param_decomp.log import logger
from param_decomp.metrics.faithfulness import faithfulness_loss


def run_faithfulness_warmup(
    component_model: ComponentModel,
    component_params: list[torch.nn.Parameter],
    config: PDConfig,
) -> None:
    """Pre-train `component_params` to faithfully approximate the target weights.

    Runs `config.faithfulness_warmup_steps` of AdamW minimising only the faithfulness loss
    over `component_model.calc_weight_deltas()`, so the sum of the components reconstructs
    the frozen target weights before the full PD loss takes over. The optimizer is
    discarded and CUDA caches cleared on exit.
    """
    logger.info("Starting faithfulness warmup phase...")
    assert component_params, "component_params is empty"

    faithfulness_warmup_optimizer = optim.AdamW(
        component_params,
        lr=config.faithfulness_warmup_lr,
        weight_decay=config.faithfulness_warmup_weight_decay,
    )

    for warmup_step in range(config.faithfulness_warmup_steps):
        faithfulness_warmup_optimizer.zero_grad()
        loss = faithfulness_loss(component_model.calc_weight_deltas())
        loss.backward()
        faithfulness_warmup_optimizer.step()

        if warmup_step % 100 == 0 or warmup_step == config.faithfulness_warmup_steps - 1:
            logger.info(
                f"Faithfulness warmup step {warmup_step + 1} / {config.faithfulness_warmup_steps}; "
                f"Faithfulness loss: {loss.item():.9f}"
            )
    del faithfulness_warmup_optimizer
    torch.cuda.empty_cache()
    gc.collect()
