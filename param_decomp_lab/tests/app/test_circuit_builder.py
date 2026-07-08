"""Tests for the circuit-builder core: j-vectors, LoRA assembly, comparison."""

import itertools

import pytest
import torch

from param_decomp_lab.app.backend.circuit_builder import (
    LoraSpec,
    SubcomponentRef,
    WriteTerm,
    apply_loras,
    build_lora,
    compare_models,
    compute_j_vectors,
    downstream_sites,
    read_vector,
    site_rank,
)
from param_decomp_lab.app.backend.mock_run import load_mock_context


@pytest.fixture(scope="module")
def ctx():
    return load_mock_context(seed=0)


def test_site_rank_ordering(ctx):
    # q/k/v are parallel; o downstream of them; mlp downstream of attn; blocks ordered.
    assert site_rank("h.0.attn.q_proj") == site_rank("h.0.attn.v_proj")
    assert site_rank("h.0.attn.o_proj") > site_rank("h.0.attn.v_proj")
    assert site_rank("h.0.mlp.c_fc") > site_rank("h.0.attn.o_proj")
    assert site_rank("h.1.attn.q_proj") > site_rank("h.0.mlp.down_proj")

    ds = downstream_sites(ctx.model, "h.1.attn.v_proj")
    assert "h.1.attn.q_proj" not in ds  # parallel, not downstream
    assert "h.1.attn.o_proj" in ds
    assert "h.1.mlp.c_fc" in ds
    assert "h.0.mlp.down_proj" not in ds
    assert "h.2.attn.q_proj" in ds
    assert ds == sorted(ds, key=site_rank)


def test_read_vector_normalized(ctx):
    v = read_vector(ctx.model, SubcomponentRef("h.0.attn.v_proj", 3))
    assert torch.allclose(v.norm(), torch.tensor(1.0), atol=1e-5)


def test_j_vector_matches_finite_difference(ctx):
    """j (averaged autograd derivative) must match a finite-difference probe."""
    model = ctx.model
    read_site = "h.1.attn.v_proj"
    target = SubcomponentRef("h.2.mlp.c_fc", 5)
    token_ids = next(ctx.token_provider.batches(batch_size=2, seq_len=8))

    (res,) = compute_j_vectors(
        model, read_site, [target], iter([token_ids]), n_prompts=2
    )
    j = res.j

    # Finite difference: perturb y at the read site along a random direction eps*d,
    # measure the change in sum of the target activation, compare with j . (eps*d)
    # summed over positions = (B*T) * mean-grad . eps*d ... we compare per-token sums.
    module = model.target_model.get_submodule(read_site)
    direction = torch.randn(module.out_features)
    direction /= direction.norm()
    eps = 1e-3

    comps = model.components[target.site]
    v_col = comps.V[:, target.idx].detach().float()
    u_norm = comps.U[target.idx, :].detach().float().norm()

    def total_activation(delta: torch.Tensor | None) -> float:
        captured = {}

        def read_hook(_m, _a, out):
            return out + delta if delta is not None else out

        def input_hook(_m, args, _o):
            captured["x"] = args[0]

        h1 = module.register_forward_hook(read_hook)
        h2 = model.target_model.get_submodule(target.site).register_forward_hook(input_hook)
        try:
            with torch.no_grad():
                model(token_ids)
        finally:
            h1.remove()
            h2.remove()
        return float((captured["x"].float() @ v_col * u_norm).sum())

    base = total_activation(None)
    plus = total_activation(eps * direction)
    minus = total_activation(-eps * direction)
    fd = (plus - minus) / (2 * eps)  # d(sum act)/d(uniform shift along direction)

    B, T = token_ids.shape
    # j is the per-token average of dsum/dy[t]; a uniform shift hits every position:
    analytic = float(j @ direction) * B * T
    assert base == base  # sanity: finite
    assert abs(fd - analytic) / (abs(fd) + 1e-6) < 5e-2, (fd, analytic)


def test_j_vector_upstream_rejected(ctx):
    with pytest.raises(AssertionError, match="not downstream"):
        compute_j_vectors(
            ctx.model,
            "h.2.mlp.c_fc",
            [SubcomponentRef("h.1.attn.v_proj", 0)],
            ctx.token_provider.batches(2, 8),
            n_prompts=2,
        )


def test_lora_build_and_apply(ctx):
    model = ctx.model
    read = SubcomponentRef("h.0.mlp.down_proj", 7)
    targets = [SubcomponentRef("h.1.attn.v_proj", 2), SubcomponentRef("h.2.mlp.c_fc", 3)]
    j_results = compute_j_vectors(
        model, read.site, targets, ctx.token_provider.batches(2, 8), n_prompts=2
    )
    spec = LoraSpec(
        name="test",
        read_site=read.site,
        read_idx=read.idx,
        writes=[WriteTerm(site=t.site, idx=t.idx, weight=w) for t, w in zip(targets, [2.0, -1.0])],
        scale=0.5,
    )
    lora = build_lora(model, spec, j_results)

    module = model.target_model.get_submodule(read.site)
    d_out, d_in = module.weight.shape
    assert lora.delta_w.shape == (d_out, d_in)
    svs = torch.linalg.svdvals(lora.delta_w.double())
    assert svs[1] / svs[0] < 1e-6  # rank 1 up to fp32 noise

    # (W + dW) x == Wx + scale * (v_hat . x) * w_total
    x = torch.randn(d_in)
    v_hat = read_vector(model, read)
    w_total = sum(w * r.j_hat for r, w in zip(j_results, [2.0, -1.0]))
    expected_extra = 0.5 * float(v_hat @ x) * w_total

    original = module.weight.data.clone()
    with apply_loras(model, [lora]):
        edited_out = module.weight @ x
    assert torch.allclose(module.weight.data, original)  # restored
    base_out = original @ x
    assert torch.allclose(edited_out - base_out, expected_extra, atol=1e-4)


def test_lora_disabled_is_noop(ctx):
    model = ctx.model
    read = SubcomponentRef("h.0.attn.v_proj", 0)
    j_results = compute_j_vectors(
        model, read.site, [SubcomponentRef("h.0.attn.o_proj", 1)],
        ctx.token_provider.batches(2, 8), n_prompts=2,
    )
    spec = LoraSpec(
        name="off", read_site=read.site, read_idx=read.idx,
        writes=[WriteTerm(site="h.0.attn.o_proj", idx=1, weight=1.0)], enabled=False,
    )
    lora = build_lora(model, spec, j_results)
    module = model.target_model.get_submodule(read.site)
    original = module.weight.data.clone()
    with apply_loras(model, [lora]):
        assert torch.equal(module.weight.data, original)


def test_compare_models_end_to_end(ctx):
    model = ctx.model
    read = SubcomponentRef("h.1.attn.v_proj", 2)
    target = SubcomponentRef("h.2.mlp.c_fc", 5)
    j_results = compute_j_vectors(
        model, read.site, [target], ctx.token_provider.batches(2, 8), n_prompts=2
    )
    spec = LoraSpec(
        name="probe", read_site=read.site, read_idx=read.idx,
        writes=[WriteTerm(site=target.site, idx=target.idx, weight=50.0)],
    )
    lora = build_lora(model, spec, j_results)

    result = compare_models(
        model, [lora], ctx.tokenizer, "hello world", top_k=5, max_new_tokens=4
    )
    assert len(result.prompt_tokens) == len("hello world".encode())
    assert len(result.positions) == len(result.prompt_tokens)
    assert all(len(p.top_base) == 5 and len(p.top_edited) == 5 for p in result.positions)
    assert result.mean_kl > 0  # a 50x write should move the logits
    assert len(result.base.greedy) > 0 and len(result.edited.greedy) > 0

    # base model must be untouched after comparison: rerun without loras
    result2 = compare_models(model, [], ctx.tokenizer, "hello world", top_k=5, max_new_tokens=4)
    assert result2.mean_kl == pytest.approx(0.0, abs=1e-9)


def test_write_term_default_weight_is_raw_norm(ctx):
    """weight=None must contribute the raw (un-normalized) j-vector."""
    model = ctx.model
    read = SubcomponentRef("h.0.attn.v_proj", 1)
    target = SubcomponentRef("h.1.mlp.c_fc", 4)
    j_results = compute_j_vectors(
        model, read.site, [target], ctx.token_provider.batches(2, 8), n_prompts=2
    )
    spec_default = LoraSpec(
        name="d", read_site=read.site, read_idx=read.idx,
        writes=[WriteTerm(site=target.site, idx=target.idx)],  # weight=None
    )
    spec_explicit = LoraSpec(
        name="e", read_site=read.site, read_idx=read.idx,
        writes=[WriteTerm(site=target.site, idx=target.idx, weight=j_results[0].raw_norm)],
    )
    dw_default = build_lora(model, spec_default, j_results).delta_w
    dw_explicit = build_lora(model, spec_explicit, j_results).delta_w
    assert torch.allclose(dw_default, dw_explicit)
    # and it equals the raw j outer v_hat
    manual = torch.outer(j_results[0].j, read_vector(model, read))
    assert torch.allclose(dw_default, manual, atol=1e-6)
