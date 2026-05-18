import torch

from param_decomp.utils.logging_utils import get_grad_norms_dict
from tests.metrics.fixtures import make_two_layer_component_model


def test_get_grad_norms_dict_handles_missing_grads() -> None:
    model = make_two_layer_component_model(weight1=torch.randn(3, 2), weight2=torch.randn(2, 3))

    loss = model.components["fc1"].V.sum()
    loss.backward()

    grad_norms = get_grad_norms_dict(model, "cpu")

    assert grad_norms["components/fc1.V"] > 0.0
    assert grad_norms["components/fc1.U"] == 0.0
    assert grad_norms["components/fc2.V"] == 0.0
    assert grad_norms["components/fc2.U"] == 0.0
    assert grad_norms["summary/components"] == grad_norms["components/fc1.V"]
    assert grad_norms["summary/ci_fns"] == 0.0
    assert grad_norms["summary/total"] == grad_norms["components/fc1.V"]
