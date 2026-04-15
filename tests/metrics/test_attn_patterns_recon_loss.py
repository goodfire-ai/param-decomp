import torch

from spd.configs import LayerwiseCiConfig
from spd.metrics.attn_patterns_recon_loss import (
    CIMaskedAttnPatternsReconLoss,
    StochasticAttnPatternsReconLoss,
    _compute_attn_patterns,
)
from spd.models.component_model import ComponentModel
from spd.pretrain.models.gpt2 import GPT2, GPT2Config
from spd.pretrain.models.gpt2_simple import GPT2Simple, GPT2SimpleConfig
from spd.pretrain.models.llama_simple import LlamaSimple, LlamaSimpleConfig
from spd.utils.module_utils import ModulePathInfo


def _make_gpt2_component_model(n_embd: int = 16, n_head: int = 2) -> ComponentModel:
    """Create a 1-layer GPT2Simple wrapped in ComponentModel with q_proj/k_proj decomposed."""
    config = GPT2SimpleConfig(
        model_type="GPT2Simple",
        block_size=32,
        vocab_size=64,
        n_layer=1,
        n_head=n_head,
        n_embd=n_embd,
        flash_attention=False,
    )
    target = GPT2Simple(config)
    target.requires_grad_(False)

    module_path_info = [
        ModulePathInfo(module_path="h.0.attn.q_proj", C=n_embd),
        ModulePathInfo(module_path="h.0.attn.k_proj", C=n_embd),
    ]

    comp_model = ComponentModel(
        target_model=target,
        module_path_info=module_path_info,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[8]),
        pretrained_model_output_attr="idx_0",
        sigmoid_type="leaky_hard",
    )
    return comp_model


def _make_gpt2_c_attn_component_model(n_embd: int = 16, n_head: int = 2) -> ComponentModel:
    """Create a 1-layer GPT2 wrapped in ComponentModel with combined c_attn decomposed."""
    config = GPT2Config(
        model_type="GPT2",
        block_size=32,
        vocab_size=64,
        n_layer=1,
        n_head=n_head,
        n_embd=n_embd,
        flash_attention=False,
    )
    target = GPT2(config)
    target.requires_grad_(False)

    module_path_info = [
        ModulePathInfo(module_path="h_torch.0.attn.c_attn", C=n_embd),
    ]

    comp_model = ComponentModel(
        target_model=target,
        module_path_info=module_path_info,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[8]),
        pretrained_model_output_attr="idx_0",
        sigmoid_type="leaky_hard",
    )
    return comp_model


class TestAttnPatternsReconLoss:
    def test_identity_decomposition_kl_near_zero(self) -> None:
        """With V=weight.T and U=eye, the component exactly reproduces the original weight,
        so attention patterns should match and KL divergence should be ~0."""
        torch.manual_seed(42)
        n_embd = 16
        n_head = 2
        model = _make_gpt2_component_model(n_embd=n_embd, n_head=n_head)

        for path in ["h.0.attn.q_proj", "h.0.attn.k_proj"]:
            target_weight = model.target_weight(path)
            with torch.no_grad():
                model.components[path].V.copy_(target_weight.T)
                model.components[path].U.copy_(torch.eye(n_embd))

        batch = torch.randint(0, 64, (2, 8))
        target_output = model(batch, cache_type="input")
        pre_weight_acts = target_output.cache
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, detach_inputs=False, sampling="continuous"
        )

        metric = CIMaskedAttnPatternsReconLoss(
            model=model,
            device="cpu",
            n_heads=n_head,
            q_proj_path="h.*.attn.q_proj",
            k_proj_path="h.*.attn.k_proj",
            c_attn_path=None,
        )
        metric.update(batch=batch, pre_weight_acts=pre_weight_acts, ci=ci)
        loss = metric.compute()

        assert loss.item() < 1e-4, f"Expected KL ≈ 0 with identity decomposition, got {loss.item()}"

    def test_random_init_kl_positive(self) -> None:
        """With random V/U init, attention patterns should differ and KL should be > 0."""
        torch.manual_seed(42)
        n_embd = 16
        n_head = 2
        model = _make_gpt2_component_model(n_embd=n_embd, n_head=n_head)

        batch = torch.randint(0, 64, (2, 8))
        target_output = model(batch, cache_type="input")
        pre_weight_acts = target_output.cache
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, detach_inputs=False, sampling="continuous"
        )

        metric = CIMaskedAttnPatternsReconLoss(
            model=model,
            device="cpu",
            n_heads=n_head,
            q_proj_path="h.*.attn.q_proj",
            k_proj_path="h.*.attn.k_proj",
            c_attn_path=None,
        )
        metric.update(batch=batch, pre_weight_acts=pre_weight_acts, ci=ci)
        loss = metric.compute()

        assert loss.item() > 0.01, f"Expected KL > 0 with random init, got {loss.item()}"

    def test_stochastic_identity_decomposition_kl_near_zero(self) -> None:
        """Stochastic variant with identity decomposition should give KL ≈ 0."""
        torch.manual_seed(42)
        n_embd = 16
        n_head = 2
        model = _make_gpt2_component_model(n_embd=n_embd, n_head=n_head)

        for path in ["h.0.attn.q_proj", "h.0.attn.k_proj"]:
            target_weight = model.target_weight(path)
            with torch.no_grad():
                model.components[path].V.copy_(target_weight.T)
                model.components[path].U.copy_(torch.eye(n_embd))

        batch = torch.randint(0, 64, (2, 8))
        target_output = model(batch, cache_type="input")
        pre_weight_acts = target_output.cache
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, detach_inputs=False, sampling="continuous"
        )

        metric = StochasticAttnPatternsReconLoss(
            model=model,
            device="cpu",
            sampling="continuous",
            use_delta_component=False,
            n_mask_samples=2,
            n_heads=n_head,
            q_proj_path="h.*.attn.q_proj",
            k_proj_path="h.*.attn.k_proj",
            c_attn_path=None,
        )
        weight_deltas = model.calc_weight_deltas()
        metric.update(
            batch=batch, pre_weight_acts=pre_weight_acts, ci=ci, weight_deltas=weight_deltas
        )
        loss = metric.compute()

        assert loss.item() < 1e-4, f"Expected KL ≈ 0 with identity decomposition, got {loss.item()}"

    def test_stochastic_random_init_kl_positive(self) -> None:
        """Stochastic variant with random init should give KL > 0."""
        torch.manual_seed(42)
        n_embd = 16
        n_head = 2
        model = _make_gpt2_component_model(n_embd=n_embd, n_head=n_head)

        batch = torch.randint(0, 64, (2, 8))
        target_output = model(batch, cache_type="input")
        pre_weight_acts = target_output.cache
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, detach_inputs=False, sampling="continuous"
        )

        metric = StochasticAttnPatternsReconLoss(
            model=model,
            device="cpu",
            sampling="continuous",
            use_delta_component=False,
            n_mask_samples=2,
            n_heads=n_head,
            q_proj_path="h.*.attn.q_proj",
            k_proj_path="h.*.attn.k_proj",
            c_attn_path=None,
        )
        weight_deltas = model.calc_weight_deltas()
        metric.update(
            batch=batch, pre_weight_acts=pre_weight_acts, ci=ci, weight_deltas=weight_deltas
        )
        loss = metric.compute()

        assert loss.item() > 0.01, f"Expected KL > 0 with random init, got {loss.item()}"


class TestCAttnPatternsReconLoss:
    def test_c_attn_identity_decomposition_kl_near_zero(self) -> None:
        """Combined c_attn path with identity decomposition should give KL ≈ 0."""
        torch.manual_seed(42)
        n_embd = 16
        n_head = 2
        model = _make_gpt2_c_attn_component_model(n_embd=n_embd, n_head=n_head)

        path = "h_torch.0.attn.c_attn"
        target_weight = model.target_weight(path)  # (3*n_embd, n_embd)
        with torch.no_grad():
            model.components[path].V.copy_(torch.eye(n_embd))  # (n_embd, C=n_embd)
            model.components[path].U.copy_(target_weight.T)  # (C=n_embd, 3*n_embd)

        batch = torch.randint(0, 64, (2, 8))
        target_output = model(batch, cache_type="input")
        pre_weight_acts = target_output.cache
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, detach_inputs=False, sampling="continuous"
        )

        metric = CIMaskedAttnPatternsReconLoss(
            model=model,
            device="cpu",
            n_heads=n_head,
            q_proj_path=None,
            k_proj_path=None,
            c_attn_path="h_torch.*.attn.c_attn",
        )
        metric.update(batch=batch, pre_weight_acts=pre_weight_acts, ci=ci)
        loss = metric.compute()

        assert loss.item() < 1e-4, f"Expected KL ≈ 0 with identity decomposition, got {loss.item()}"

    def test_c_attn_random_init_kl_positive(self) -> None:
        """Combined c_attn path with random init should give KL > 0."""
        torch.manual_seed(42)
        n_embd = 16
        n_head = 2
        model = _make_gpt2_c_attn_component_model(n_embd=n_embd, n_head=n_head)

        batch = torch.randint(0, 64, (2, 8))
        target_output = model(batch, cache_type="input")
        pre_weight_acts = target_output.cache
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, detach_inputs=False, sampling="continuous"
        )

        metric = CIMaskedAttnPatternsReconLoss(
            model=model,
            device="cpu",
            n_heads=n_head,
            q_proj_path=None,
            k_proj_path=None,
            c_attn_path="h_torch.*.attn.c_attn",
        )
        metric.update(batch=batch, pre_weight_acts=pre_weight_acts, ci=ci)
        loss = metric.compute()

        assert loss.item() > 0.01, f"Expected KL > 0 with random init, got {loss.item()}"


def _make_llama_component_model(n_embd: int = 16, n_head: int = 2) -> ComponentModel:
    """Create a 1-layer LlamaSimple with RoPE, wrapped in ComponentModel with q_proj/k_proj."""
    config = LlamaSimpleConfig(
        model_type="LlamaSimple",
        block_size=32,
        vocab_size=64,
        n_layer=1,
        n_head=n_head,
        n_embd=n_embd,
        n_intermediate=n_embd * 4 * 2 // 3,
        use_grouped_query_attention=True,
        n_key_value_heads=n_head,
        flash_attention=False,
        n_ctx=32,
        rotary_dim=n_embd // n_head,
    )
    target = LlamaSimple(config)
    target.requires_grad_(False)

    module_path_info = [
        ModulePathInfo(module_path="h.0.attn.q_proj", C=n_embd),
        ModulePathInfo(module_path="h.0.attn.k_proj", C=n_embd),
    ]

    return ComponentModel(
        target_model=target,
        module_path_info=module_path_info,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[8]),
        pretrained_model_output_attr="idx_0",
        sigmoid_type="leaky_hard",
    )


class TestRoPEAttnPatternsReconLoss:
    def test_rope_identity_decomposition_kl_near_zero(self) -> None:
        """Identity decomposition with auto-detected RoPE should give KL ~ 0."""
        torch.manual_seed(42)
        n_embd = 16
        n_head = 2
        model = _make_llama_component_model(n_embd=n_embd, n_head=n_head)

        for path in ["h.0.attn.q_proj", "h.0.attn.k_proj"]:
            target_weight = model.target_weight(path)
            with torch.no_grad():
                model.components[path].V.copy_(target_weight.T)
                model.components[path].U.copy_(torch.eye(n_embd))

        batch = torch.randint(0, 64, (2, 8))
        target_output = model(batch, cache_type="input")
        pre_weight_acts = target_output.cache
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, detach_inputs=False, sampling="continuous"
        )

        metric = CIMaskedAttnPatternsReconLoss(
            model=model,
            device="cpu",
            n_heads=n_head,
            q_proj_path="h.*.attn.q_proj",
            k_proj_path="h.*.attn.k_proj",
            c_attn_path=None,
        )
        metric.update(batch=batch, pre_weight_acts=pre_weight_acts, ci=ci)
        loss = metric.compute()

        assert loss.item() < 1e-4, f"Expected KL ≈ 0 with identity decomposition, got {loss.item()}"

    def test_rope_random_init_kl_positive(self) -> None:
        """Random init with RoPE should give KL > 0."""
        torch.manual_seed(42)
        n_embd = 16
        n_head = 2
        model = _make_llama_component_model(n_embd=n_embd, n_head=n_head)

        batch = torch.randint(0, 64, (2, 8))
        target_output = model(batch, cache_type="input")
        pre_weight_acts = target_output.cache
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts, detach_inputs=False, sampling="continuous"
        )

        metric = CIMaskedAttnPatternsReconLoss(
            model=model,
            device="cpu",
            n_heads=n_head,
            q_proj_path="h.*.attn.q_proj",
            k_proj_path="h.*.attn.k_proj",
            c_attn_path=None,
        )
        metric.update(batch=batch, pre_weight_acts=pre_weight_acts, ci=ci)
        loss = metric.compute()

        assert loss.item() > 0.01, f"Expected KL > 0 with random init, got {loss.item()}"

    def test_rope_changes_patterns(self) -> None:
        """Applying RoPE should produce different attention patterns than without."""
        torch.manual_seed(42)
        n_embd = 16
        n_head = 2
        model = _make_llama_component_model(n_embd=n_embd, n_head=n_head)
        attn_module = model.target_model.get_submodule("h.0.attn")

        q = torch.randn(2, 8, n_embd)
        k = torch.randn(2, 8, n_embd)

        patterns_without_rope = _compute_attn_patterns(q, k, n_head, attn_module=None)
        patterns_with_rope = _compute_attn_patterns(q, k, n_head, attn_module=attn_module)

        assert not torch.allclose(patterns_without_rope, patterns_with_rope, atol=1e-6), (
            "RoPE should change attention patterns"
        )
