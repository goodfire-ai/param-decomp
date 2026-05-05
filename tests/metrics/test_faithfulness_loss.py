import torch

from param_decomp.metrics import faithfulness_loss
from param_decomp.models.component_model import ComponentModel
from tests.metrics.fixtures import make_one_layer_component_model


def zero_out_components(model: ComponentModel) -> None:
    with torch.no_grad():
        for cm in model.components.values():
            cm.V.zero_()
            cm.U.zero_()


class TestCalcWeightDeltas:
    def test_components_and_identity(self: object) -> None:
        # fc weight 2x3 with known values
        fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)
        zero_out_components(model)
        deltas = model.calc_weight_deltas()

        assert set(deltas.keys()) == {"fc"}

        # components were zeroed, so delta equals original weight
        expected_fc = fc_weight
        assert torch.allclose(deltas["fc"], expected_fc)

    def test_components_nonzero(self: object) -> None:
        fc_weight = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)

        deltas = model.calc_weight_deltas()
        assert set(deltas.keys()) == {"fc"}

        component = model.components["fc"]
        assert component is not None
        expected_fc = model.target_weight("fc") - component.weight
        assert torch.allclose(deltas["fc"], expected_fc)


class TestCalcFaithfulnessLoss:
    def test_manual_weight_deltas_normalization(self: object) -> None:
        weight_deltas = {
            "a": torch.tensor([[1.0, -1.0], [2.0, 0.0]], dtype=torch.float32),  # sum sq = 6
            "b": torch.tensor([[2.0, -2.0, 1.0]], dtype=torch.float32),  # sum sq = 9
        }
        # total sum sq = 15, total params = 4 + 3 = 7
        expected = torch.tensor(15.0 / 7.0)
        result = faithfulness_loss(weight_deltas=weight_deltas)
        assert torch.allclose(result, expected)

    def test_with_model_weight_deltas(self: object) -> None:
        fc_weight = torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, -4.0]], dtype=torch.float32)
        model = make_one_layer_component_model(weight=fc_weight)
        zero_out_components(model)
        deltas = model.calc_weight_deltas()

        # Expected: mean of squared entries across both matrices
        expected = fc_weight.square().sum() / fc_weight.numel()

        result = faithfulness_loss(weight_deltas=deltas)
        assert torch.allclose(result, expected)
