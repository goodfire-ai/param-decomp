"""Export manual prompt activation examples for a single component.

Produces a compact JSON file that the VPD blog frontend can render using the
same activation-example widget used for component cards and prompt viewers.

The output shape is intentionally compatible with the blog's existing
activation renderer:

    {
      "key": "1.attn.q:308",
      "label": "copula verbs",
      "reasoning": "...",
      "max_act": 3.4217,
      "examples": [
        {
          "prompt": "The princess lost her crown",
          "t": ["The", " princess", ...],
          "c": [0.0, 0.12, ...],
          "a": [0.0, -0.42, ...]
        }
      ]
    }

Run from ~/param-decomp:
  uv run python -m scripts.blog.export_manual_prompts \
      --component 1.attn.q:308 \
      --prompts-file ../vpd-blog-replit/my-prompts.json \
      --out-file ../vpd-blog-replit/data/manual-prompts/copula.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from param_decomp.app.backend.app_tokenizer import AppTokenizer
from param_decomp.app.backend.compute import compute_ci_only
from param_decomp.autointerp.repo import InterpRepo
from param_decomp.configs import LMTaskConfig
from param_decomp.models.component_model import ComponentModel, ParamDecompRunInfo
from param_decomp.scripts.prompt_utils import load_prompts
from param_decomp.topology import TransformerTopology
from param_decomp.utils.distributed_utils import get_device
from param_decomp.utils.wandb_utils import parse_wandb_run_path
from scripts.blog.constants import WANDB_PATH

ROUND_DIGITS = 4


def build_manual_prompt_example(
    prompt: str,
    tokens: Sequence[str],
    ci_values: Sequence[float],
    component_acts: Sequence[float],
    *,
    truncated: bool = False,
) -> tuple[dict[str, Any], float]:
    """Build one prompt-viewer-compatible example row.

    Returns the exported example dict and the max absolute activation in the
    unrounded activations for global normalization.
    """
    n_tokens = len(tokens)
    assert len(ci_values) == n_tokens, f"CI/token length mismatch: {len(ci_values)} != {n_tokens}"
    assert len(component_acts) == n_tokens, (
        f"Activation/token length mismatch: {len(component_acts)} != {n_tokens}"
    )

    max_abs_act = max((abs(float(act)) for act in component_acts), default=0.0)
    example: dict[str, Any] = {
        "prompt": prompt,
        "t": list(tokens),
        "c": [round(float(ci), ROUND_DIGITS) for ci in ci_values],
        "a": [round(float(act), ROUND_DIGITS) for act in component_acts],
    }
    if truncated:
        example["truncated"] = True
    return example, max_abs_act


def build_export_payload(
    component_key: str,
    run_path: str,
    run_id: str,
    context_length: int,
    examples: Sequence[dict[str, Any]],
    max_abs_act: float,
    *,
    label: str | None = None,
    reasoning: str | None = None,
) -> dict[str, Any]:
    """Assemble the final JSON payload."""
    layer, component_idx_str = component_key.rsplit(":", 1)

    payload: dict[str, Any] = {
        "key": component_key,
        "layer": layer,
        "component_idx": int(component_idx_str),
        "run_id": run_id,
        "run_path": run_path,
        "context_length": context_length,
        "n_prompts": len(examples),
        "max_act": round(max_abs_act, ROUND_DIGITS),
        "examples": list(examples),
    }
    if label:
        payload["label"] = label
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def _resolve_component_metadata(
    run_id: str, concrete_component_key: str
) -> tuple[str | None, str | None]:
    interp_repo = InterpRepo.open(run_id)
    if interp_repo is None:
        return None, None

    interp = interp_repo.get_interpretation(concrete_component_key)
    if interp is None:
        return None, None

    label = interp.label if interp.label and interp.label != "unclear" else None
    reasoning = interp.reasoning or None
    return label, reasoning


def _resolve_run_id(run_path: str, checkpoint_path: Path) -> str:
    """Recover the wandb run id for cached wandb runs and local checkpoints."""
    try:
        _entity, _project, run_id = parse_wandb_run_path(run_path)
        return run_id
    except ValueError:
        parent_name = checkpoint_path.parent.name
        if "-" in parent_name:
            return parent_name.split("-", 1)[1]
        return parent_name


def export_manual_prompts(
    run_path: str,
    component_key: str,
    prompts: Sequence[str],
) -> dict[str, Any]:
    """Compute CI/activation values for one component across manual prompts."""
    run_info = ParamDecompRunInfo.from_path(run_path)
    model = ComponentModel.from_run_info(run_info).to(get_device())
    model.eval()

    tokenizer_name = run_info.config.tokenizer_name
    assert tokenizer_name is not None, "run config missing tokenizer_name"
    tokenizer = AppTokenizer.from_pretrained(tokenizer_name)
    topology = TransformerTopology(model.target_model)

    task_config = run_info.config.task_config
    assert isinstance(task_config, LMTaskConfig), "manual prompt export only supports LM task runs"
    context_length = task_config.max_seq_len

    canonical_layer, component_idx_str = component_key.rsplit(":", 1)
    component_idx = int(component_idx_str)
    concrete_layer = topology.canon_to_target(canonical_layer)
    concrete_component_key = f"{concrete_layer}:{component_idx}"
    run_id = _resolve_run_id(run_path=run_path, checkpoint_path=run_info.checkpoint_path)

    label, reasoning = _resolve_component_metadata(
        run_id=run_id,
        concrete_component_key=concrete_component_key,
    )

    examples: list[dict[str, Any]] = []
    global_max_abs_act = 0.0
    device = next(model.parameters()).device

    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        assert token_ids, f"Prompt produced no tokens: {prompt!r}"
        truncated = len(token_ids) > context_length
        token_ids = token_ids[:context_length]

        tokens_tensor = torch.tensor([token_ids], device=device)
        result = compute_ci_only(
            model=model, tokens=tokens_tensor, sampling=run_info.config.sampling
        )

        ci_tensor = result.ci_lower_leaky[concrete_layer]
        act_tensor = result.component_acts[concrete_layer]
        spans = tokenizer.get_spans(token_ids)

        example, local_max_abs_act = build_manual_prompt_example(
            prompt=prompt,
            tokens=spans,
            ci_values=ci_tensor[0, :, component_idx].tolist(),
            component_acts=act_tensor[0, :, component_idx].tolist(),
            truncated=truncated,
        )
        examples.append(example)
        global_max_abs_act = max(global_max_abs_act, local_max_abs_act)

    return build_export_payload(
        component_key=component_key,
        run_path=run_path,
        run_id=run_id,
        context_length=context_length,
        examples=examples,
        max_abs_act=global_max_abs_act,
        label=label,
        reasoning=reasoning,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-path",
        default=WANDB_PATH,
        help="Wandb run path or local checkpoint path. Defaults to the blog run.",
    )
    parser.add_argument(
        "--component",
        required=True,
        help="Canonical component key, e.g. 1.attn.q:308",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        required=True,
        help="JSON file containing a list of prompt strings",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        required=True,
        help="Destination JSON file",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_file)
    assert prompts, f"No prompts found in {args.prompts_file}"

    payload = export_manual_prompts(
        run_path=args.run_path,
        component_key=args.component,
        prompts=prompts,
    )

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out_file}")


if __name__ == "__main__":
    main()
