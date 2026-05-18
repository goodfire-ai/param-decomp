import json
from pathlib import Path
from typing import Any

import torch
import wandb
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from param_decomp.models.component_model import ComponentModel


def local_log(data: dict[str, Any], step: int, out_dir: Path) -> None:
    metrics_file = out_dir / "metrics.jsonl"
    metrics_file.touch(exist_ok=True)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    metrics_without_images = {}
    for k, v in data.items():
        if isinstance(v, Image.Image):
            filename = f"{k.replace('/', '_')}_{step}.png"
            v.save(fig_dir / filename)
            tqdm.write(f"Saved figure {k} to {fig_dir / filename}")
        elif isinstance(v, wandb.plot.CustomChart):
            json_path = fig_dir / f"{k.replace('/', '_')}_{step}.json"
            payload = {"columns": list(v.table.columns), "data": list(v.table.data), "step": step}
            with open(json_path, "w") as f:
                json.dump(payload, f, default=str)
            tqdm.write(f"Saved custom chart data {k} to {json_path}")
        else:
            metrics_without_images[k] = v

    with open(metrics_file, "a") as f:
        f.write(json.dumps({"step": step, **metrics_without_images}) + "\n")


def _grad_sum_sq(param: torch.nn.Parameter, device: torch.device | str) -> Float[Tensor, ""]:
    if param.grad is None:
        return torch.zeros((), device=device)
    return param.grad.detach().pow(2).sum()


def get_grad_norms_dict(
    component_model: ComponentModel, device: torch.device | str
) -> dict[str, float]:
    """Create a dictionary of gradient norms for the parameters of a component model."""

    out: dict[str, float] = {}

    comp_grad_norm_sq_sum: Float[Tensor, ""] = torch.zeros((), device=device)
    for target_module_path, component in component_model.components.items():
        for local_param_name, local_param in component.named_parameters():
            param_grad_sum_sq = _grad_sum_sq(local_param, device)
            key = f"components/{target_module_path}.{local_param_name}"
            out[key] = param_grad_sum_sq.sqrt().item()
            comp_grad_norm_sq_sum += param_grad_sum_sq

    ci_fn_grad_norm_sq_sum: Float[Tensor, ""] = torch.zeros((), device=device)
    for local_param_name, local_param in component_model.ci_fn.named_parameters():
        ci_fn_grad_sum_sq = _grad_sum_sq(local_param, device)
        key = f"ci_fns/{local_param_name}"
        assert key not in out, f"Key {key} already exists in grad norms log"
        out[key] = ci_fn_grad_sum_sq.sqrt().item()
        ci_fn_grad_norm_sq_sum += ci_fn_grad_sum_sq

    out["summary/components"] = comp_grad_norm_sq_sum.sqrt().item()
    out["summary/ci_fns"] = ci_fn_grad_norm_sq_sum.sqrt().item()

    total_grad_norm_sq_sum = comp_grad_norm_sq_sum + ci_fn_grad_norm_sq_sum
    out["summary/total"] = total_grad_norm_sq_sum.sqrt().item()

    return out
