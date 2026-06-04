import einops
import torch

from param_decomp.components import LinearComponents
from param_decomp.metrics.masked_weight_norm import (
    _masked_weight_norm_sum,
    masked_weight_norm_loss,
)


def _make_components(specs: dict[str, tuple[int, int, int]]) -> dict[str, LinearComponents]:
    """Build `LinearComponents` keyed by name from `{name: (d_in, C, d_out)}`."""
    comps: dict[str, LinearComponents] = {}
    for name, (d_in, C, d_out) in specs.items():
        comps[name] = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=None)
    return comps


def _naive_sum(ci: dict[str, torch.Tensor], comps: dict[str, LinearComponents]) -> torch.Tensor:
    """Materialise the per-datapoint masked weight and sum its squared entries."""
    total = torch.zeros((), dtype=next(iter(ci.values())).dtype)
    for name, layer_ci in ci.items():
        c = comps[name]
        weight = einops.einsum(layer_ci, c.V, c.U, "... k, din k, k dout -> ... dout din")
        total = total + (weight**2).sum()
    return total


class TestMaskedWeightNormLoss:
    def test_matches_naive_materialised_form(self: object) -> None:
        torch.manual_seed(0)
        comps = _make_components({"l0": (7, 3, 5), "l1": (4, 6, 9)})
        for c in comps.values():
            c.V.data = c.V.data.double()
            c.U.data = c.U.data.double()
        ci = {
            "l0": torch.rand(2, 8, 3, dtype=torch.float64),
            "l1": torch.rand(2, 8, 6, dtype=torch.float64),
        }
        gram_sum, n = _masked_weight_norm_sum(ci, comps)  # pyright: ignore[reportArgumentType]
        assert n == 16
        assert torch.allclose(gram_sum, _naive_sum(ci, comps))

    def test_normalisation_and_hand_value(self: object) -> None:
        # One layer, one datapoint: V=[[1],[0]] (d_in=2, C=1), U=[[1, 2]] (C=1, d_out=2).
        # masked weight = ci * V outer U, with ci=3 -> entries [[3*1*1, 3*1*2],[0,0]]
        # squared sum = 9 + 36 = 45. total weights = d_in*d_out = 4, n_examples = 1.
        comps = _make_components({"l0": (2, 1, 2)})
        comps["l0"].V.data = torch.tensor([[1.0], [0.0]])
        comps["l0"].U.data = torch.tensor([[1.0, 2.0]])
        ci = {"l0": torch.tensor([[3.0]])}
        loss = masked_weight_norm_loss(ci, comps)  # pyright: ignore[reportArgumentType]
        assert torch.allclose(loss, torch.tensor(45.0 / 4.0))

    def test_grad_flows_to_components_and_ci(self: object) -> None:
        torch.manual_seed(1)
        comps = _make_components({"l0": (5, 3, 4)})
        ci = {"l0": torch.rand(2, 6, 3, requires_grad=True)}
        masked_weight_norm_loss(ci, comps).backward()  # pyright: ignore[reportArgumentType]
        assert comps["l0"].V.grad is not None and comps["l0"].V.grad.abs().sum() > 0
        assert comps["l0"].U.grad is not None and comps["l0"].U.grad.abs().sum() > 0
        assert ci["l0"].grad is not None and ci["l0"].grad.abs().sum() > 0
