import torch

from param_decomp.ci_fns import (
    CiBottleneckConfig,
    GlobalSharedTransformerCiFn,
    TargetLayerConfig,
)
from param_decomp.ci_nn_blocks import DoubleSidedJumpReLU, SparseBottleneck


class TestDoubleSidedJumpReLU:
    def test_gate_with_tiny_theta_is_identity(self) -> None:
        # theta = exp(log_theta), so theta_init=1e-30 gives effectively-zero theta
        gate = DoubleSidedJumpReLU(dim=4, theta_init=1e-30, bandwidth=1e-3)
        u = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
        assert torch.allclose(gate(u), u)

    def test_gate_with_large_theta_is_zero(self) -> None:
        gate = DoubleSidedJumpReLU(dim=4, theta_init=10.0, bandwidth=1e-3)
        u = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
        assert torch.all(gate(u) == 0)

    def test_sign_preserved_above_threshold(self) -> None:
        gate = DoubleSidedJumpReLU(dim=4, theta_init=1.0, bandwidth=1e-3)
        u = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
        assert torch.allclose(gate(u), torch.tensor([[-2.0, 0.0, 0.0, 2.0]]))

    def test_ste_grad_through_u(self) -> None:
        gate = DoubleSidedJumpReLU(dim=2, theta_init=1.0, bandwidth=1e-3)
        u = torch.tensor([[2.0, 0.5]], requires_grad=True)
        gate(u).sum().backward()
        assert u.grad is not None
        # col 0: |u| > theta => grad 1; col 1: gated off => grad 0
        assert torch.allclose(u.grad, torch.tensor([[1.0, 0.0]]))

    def test_log_theta_grad_value_in_window(self) -> None:
        # theta=0.5, bandwidth=0.2 => kernel support |u| in (0.4, 0.6)
        gate = DoubleSidedJumpReLU(dim=2, theta_init=0.5, bandwidth=0.2)
        u = torch.tensor([[0.55, -0.55]])
        gate(u).sum().backward()
        assert gate.log_theta.grad is not None
        # dz/dtheta = -u / bandwidth inside the window; chain through theta = exp(log_theta)
        theta = 0.5
        expected = torch.tensor([-0.55 / 0.2 * theta, 0.55 / 0.2 * theta])
        assert torch.allclose(gate.log_theta.grad, expected)

    def test_log_theta_grad_zero_outside_window(self) -> None:
        gate = DoubleSidedJumpReLU(dim=2, theta_init=0.5, bandwidth=0.2)
        # |u| = 2.0 and 0.1 are both outside the (0.4, 0.6) kernel support
        u = torch.tensor([[2.0, 0.1]])
        gate(u).sum().backward()
        assert gate.log_theta.grad is not None
        assert torch.all(gate.log_theta.grad == 0)

    def test_theta_positive_by_construction(self) -> None:
        # exp underflows to exactly 0 below ~-87 in fp32; -50 is extreme but representable
        gate = DoubleSidedJumpReLU(dim=3, theta_init=0.05, bandwidth=0.05)
        with torch.no_grad():
            gate.log_theta.fill_(-50.0)
        assert torch.all(gate.theta > 0)


class TestSparseBottleneck:
    def test_output_shape_and_exact_zeros(self) -> None:
        bottleneck = SparseBottleneck(input_dim=16, bottleneck_dim=8, theta_init=0.5, bandwidth=0.1)
        x = torch.randn(4, 7, 16)
        codes = bottleneck(x)
        assert codes.shape == (4, 7, 8)
        gated_off = codes == 0
        gated_on = codes.abs() > bottleneck.gate.theta
        assert torch.all(gated_off | gated_on)


def _make_transformer_ci_fn(
    bottleneck_config: CiBottleneckConfig | None,
) -> GlobalSharedTransformerCiFn:
    return GlobalSharedTransformerCiFn(
        target_model_layer_configs={
            "layer_a": TargetLayerConfig(input_dim=8, C=5),
            "layer_b": TargetLayerConfig(input_dim=4, C=3),
        },
        d_model=16,
        n_layers=1,
        n_heads=2,
        max_len=32,
        bottleneck_config=bottleneck_config,
    )


class TestGlobalSharedTransformerCiFnBottleneck:
    def test_no_bottleneck_codes_none(self) -> None:
        ci_fn = _make_transformer_ci_fn(bottleneck_config=None)
        input_acts = {"layer_a": torch.randn(2, 7, 8), "layer_b": torch.randn(2, 7, 4)}
        out = ci_fn(input_acts)
        assert out.bottleneck_codes is None
        assert out.pre_sigmoid["layer_a"].shape == (2, 7, 5)
        assert out.pre_sigmoid["layer_b"].shape == (2, 7, 3)

    def test_bottleneck_codes_shape(self) -> None:
        cfg = CiBottleneckConfig(bottleneck_dim=6, decoder_hidden_dims=[12])
        ci_fn = _make_transformer_ci_fn(bottleneck_config=cfg)
        input_acts = {"layer_a": torch.randn(2, 7, 8), "layer_b": torch.randn(2, 7, 4)}
        out = ci_fn(input_acts)
        assert out.bottleneck_codes is not None
        assert out.bottleneck_codes.shape == (2, 7, 6)
        assert out.pre_sigmoid["layer_a"].shape == (2, 7, 5)
        assert out.pre_sigmoid["layer_b"].shape == (2, 7, 3)

    def test_bottleneck_codes_squeezed_for_2d_inputs(self) -> None:
        cfg = CiBottleneckConfig(bottleneck_dim=6, decoder_hidden_dims=[])
        ci_fn = _make_transformer_ci_fn(bottleneck_config=cfg)
        input_acts = {"layer_a": torch.randn(2, 8), "layer_b": torch.randn(2, 4)}
        out = ci_fn(input_acts)
        assert out.bottleneck_codes is not None
        assert out.bottleneck_codes.shape == (2, 6)
        assert out.pre_sigmoid["layer_a"].shape == (2, 5)

    def test_grads_flow_through_bottleneck(self) -> None:
        cfg = CiBottleneckConfig(bottleneck_dim=6, decoder_hidden_dims=[12], theta_init=1e-30)
        ci_fn = _make_transformer_ci_fn(bottleneck_config=cfg)
        input_acts = {"layer_a": torch.randn(2, 7, 8), "layer_b": torch.randn(2, 7, 4)}
        out = ci_fn(input_acts)
        loss = sum(v.sum() for v in out.pre_sigmoid.values())
        assert isinstance(loss, torch.Tensor)
        loss.backward()
        assert ci_fn._bottleneck is not None
        proj_grad = ci_fn._bottleneck._proj.W.grad
        assert proj_grad is not None
        assert proj_grad.abs().sum() > 0
