"""Build a prompts JSON file from harvest activation examples for target components.

For each component listed (e.g. 'h.1.attn.q_proj:308'), pulls the reservoir-sampled
activation examples from the harvest DB. Each example contains a token window where
the component fired (CI > threshold) at the centre. Decodes those windows to text
and writes a flat JSON list of prompts that can be passed to compute_data.py via
--prompts_file.

By construction, every prompt is guaranteed to have at least one of the listed
components causally important somewhere in its tokens.

Usage:
    python -m param_decomp.scripts.interactive_qk_contributions.build_targeted_prompts \
        wandb:goodfire/spd/runs/<run_id> \
        --components 'h.1.attn.q_proj:308,h.1.attn.k_proj:218,h.1.attn.k_proj:485' \
        --output_path param_decomp/scripts/interactive_qk_contributions/targeted_prompts.json
"""

import json
from collections.abc import Sequence
from pathlib import Path

import fire
from transformers import AutoTokenizer

from param_decomp.harvest.repo import HarvestRepo
from param_decomp.log import logger
from param_decomp.models.component_model import ParamDecompRunInfo
from param_decomp.param_decomp_types import ModelPath
from param_decomp.utils.wandb_utils import parse_wandb_run_path

SCRIPT_DIR = Path(__file__).parent


def build_targeted_prompts(
    wandb_path: ModelPath,
    components: str,
    output_path: str | None = None,
    max_per_component: int | None = None,
) -> None:
    """Decode harvest activation examples for the listed components into prompt strings.

    Args:
        wandb_path: WandB run path.
        components: Comma-separated component keys (e.g.
            'h.1.attn.q_proj:308,h.1.attn.k_proj:485').
        output_path: Where to write the JSON list. Defaults to
            param_decomp/scripts/interactive_qk_contributions/targeted_prompts.json.
        max_per_component: Cap on examples to take per component.
    """
    component_keys: Sequence[str] = [c.strip() for c in components.split(",") if c.strip()]
    assert component_keys, "No components specified"

    _entity, _project, run_id = parse_wandb_run_path(str(wandb_path))
    run_info = ParamDecompRunInfo.from_path(wandb_path)

    repo = HarvestRepo.open_most_recent(run_id)
    assert repo is not None, f"No harvest data found for {run_id}"

    tokenizer = AutoTokenizer.from_pretrained(run_info.config.tokenizer_name)

    # Dedupe by exact token-id tuple to avoid identical reservoir picks across components
    seen: set[tuple[int, ...]] = set()
    prompts: list[str] = []
    per_component_kept: dict[str, int] = {}

    for comp_key in component_keys:
        comp = repo.get_component(comp_key)
        assert comp is not None, f"Component {comp_key} not found in harvest"
        examples = comp.activation_examples
        if max_per_component is not None:
            examples = examples[:max_per_component]

        kept = 0
        for ex in examples:
            tok_tuple = tuple(ex.token_ids)
            if tok_tuple in seen:
                continue
            seen.add(tok_tuple)
            text = tokenizer.decode(ex.token_ids, skip_special_tokens=False)  # pyright: ignore[reportAttributeAccessIssue]
            prompts.append(text)
            kept += 1
        per_component_kept[comp_key] = kept
        logger.info(f"{comp_key}: {len(examples)} examples in reservoir, {kept} unique kept")

    out = Path(output_path) if output_path else SCRIPT_DIR / "targeted_prompts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(prompts, f, indent=2)

    logger.info(f"Wrote {len(prompts)} unique prompts to {out}")
    logger.info(f"Per-component contribution: {per_component_kept}")


if __name__ == "__main__":
    fire.Fire(build_targeted_prompts)
