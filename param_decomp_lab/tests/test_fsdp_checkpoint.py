"""Single-process smoke test for the FSDP DCP checkpoint round-trip.

DCP works single-process (no distributed init), so we can verify the
trainable-only save/load contract without a sharded model: a tiny module with
both a trainable and a frozen param round-trips its trainable state + optimizer
moments + step + loss-metric states, and the frozen param is excluded from the
checkpoint (its on-disk value is never written, so corrupting it in the live
model and loading does NOT restore it).
"""

from pathlib import Path
from typing import override

import torch
from torch import nn

from param_decomp_lab.fsdp.checkpoint import latest_dcp_step, load_dcp, save_dcp


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = nn.Parameter(torch.zeros(4))
        self.frozen = nn.Parameter(torch.zeros(4), requires_grad=False)

    @override
    def forward(self) -> torch.Tensor:
        return self.trainable.sum()


def test_save_load_roundtrip_excludes_frozen(tmp_path: Path) -> None:
    model = _TinyModel()
    opt = torch.optim.AdamW([model.trainable], lr=1e-2)

    with torch.no_grad():
        model.trainable.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        model.frozen.copy_(torch.tensor([9.0, 9.0, 9.0, 9.0]))
    model.forward().backward()
    opt.step()

    saved_trainable = model.trainable.detach().clone()
    saved_exp_avg = opt.state[model.trainable]["exp_avg"].clone()
    loss_metric_states = {"FaithfulnessLoss": {"running": torch.tensor([0.5, 0.25])}}

    optimizers: dict[str, torch.optim.Optimizer] = {"components": opt}
    save_dcp(
        model,
        optimizers,
        step=10,
        loss_metric_states=loss_metric_states,
        out_dir=tmp_path,
    )

    assert latest_dcp_step(tmp_path) == 10

    with torch.no_grad():
        model.trainable.zero_()
        model.frozen.copy_(torch.tensor([-1.0, -1.0, -1.0, -1.0]))
    opt.state[model.trainable]["exp_avg"].zero_()

    returned = load_dcp(
        model,
        optimizers,
        step=10,
        in_dir=tmp_path,
        loss_metric_states={"FaithfulnessLoss": {"running": torch.zeros(2)}},
    )

    assert torch.equal(model.trainable, saved_trainable)
    assert torch.equal(opt.state[model.trainable]["exp_avg"], saved_exp_avg)
    assert torch.equal(returned["FaithfulnessLoss"]["running"], torch.tensor([0.5, 0.25]))
    # frozen param was never saved → load leaves the corrupted live value untouched
    assert torch.equal(model.frozen, torch.tensor([-1.0, -1.0, -1.0, -1.0]))
