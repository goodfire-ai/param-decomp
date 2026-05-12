import torch

from param_decomp.metrics import faithfulness_loss
from param_decomp.models.component_model import ComponentModel
from tests.metrics.fixtures import make_one_layer_component_model


def zero_out_components(model: ComponentModel) -> None:
    with torch.no_grad():
        for cm in model.components.values():
            cm.V.zero_()
            cm.U.zero_()


class TestCalcFaithfulnessTerms:
    def test_components_zeroed(self: object) -> None:
        # Components zeroed → delta equals the target weight, so sum_sq = ||W||_F^2.
        fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)
        zero_out_components(model)

        sum_sq, numel = model.calc_faithfulness_terms()

        assert numel == fc_weight.numel()
        assert torch.allclose(sum_sq, fc_weight.square().sum())

    def test_components_nonzero(self: object) -> None:
        fc_weight = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        component = model.components["fc"]
        expected_delta = model.target_weight("fc") - component.component_weight
        sum_sq, numel = model.calc_faithfulness_terms()

        assert numel == expected_delta.numel()
        assert torch.allclose(sum_sq, expected_delta.square().sum())


class TestCalcFaithfulnessLoss:
    def test_zeroed_components_yields_mean_squared_target(self: object) -> None:
        fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)
        zero_out_components(model)

        expected = fc_weight.square().sum() / fc_weight.numel()
        result = faithfulness_loss(model=model)
        assert torch.allclose(result, expected)
