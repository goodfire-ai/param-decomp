"""The authored LM CI schema (`experiments.lm.config`): the `decomposition.ci` union and
the chunkwise arm's `attention` / `ffn` discriminated unions, validated from dicts — the
shape a run yaml actually arrives as, so these exercise the discriminators the way a
config file hits them, not the typed constructor."""

import pytest
from pydantic import ValidationError

from param_decomp.experiments.lm.config import (
    ChunkwiseTransformerCIFnConfig,
    GeluCIFnFfnConfig,
    GlobalMlpCIFnConfig,
    GQACIFnAttentionConfig,
    LMDecompositionConfig,
    MHACIFnAttentionConfig,
    SwigluCIFnFfnConfig,
    _resolve_ci_fn_attention,
)


def _cfg(
    attention: dict[str, object] | None = None, ffn: dict[str, object] | None = None
) -> ChunkwiseTransformerCIFnConfig:
    return ChunkwiseTransformerCIFnConfig.model_validate(
        {
            "blocks_per_chunk": 1,
            "d_model": 16,
            "n_blocks": 1,
            "attention": attention
            if attention is not None
            else {"mask": "bidirectional", "kind": "mha", "implementation": "xla", "n_heads": 4},
            "ffn": ffn if ffn is not None else {"kind": "gelu", "hidden": 32},
        }
    )


# ----------------------------- attention -----------------------------


def test_mha_arm_parses():
    cfg = _cfg(
        attention={"mask": "bidirectional", "kind": "mha", "implementation": "xla", "n_heads": 4}
    )
    assert isinstance(cfg.attention, MHACIFnAttentionConfig)
    assert cfg.attention.n_heads == 4


def test_gqa_arm_parses():
    cfg = _cfg(
        attention={
            "mask": "bidirectional",
            "kind": "gqa",
            "implementation": "xla",
            "n_heads": 4,
            "n_kv_heads": 2,
        }
    )
    assert isinstance(cfg.attention, GQACIFnAttentionConfig)
    assert (cfg.attention.n_heads, cfg.attention.n_kv_heads) == (4, 2)


@pytest.mark.parametrize(
    "attention", [{"kind": "mha", "n_heads": 4}, {"kind": "gqa", "n_heads": 4, "n_kv_heads": 2}]
)
@pytest.mark.parametrize("mask", ["bidirectional", "causal"])
def test_attention_mask_resolves_its_authored_choice(attention: dict[str, object], mask: str):
    authored = {**attention, "implementation": "xla", "mask": mask}
    resolved = _resolve_ci_fn_attention(_cfg(attention=authored).attention)
    assert resolved.mask == mask
    assert resolved.implementation == "xla"


@pytest.mark.parametrize(
    "attention", [{"kind": "mha", "n_heads": 4}, {"kind": "gqa", "n_heads": 4, "n_kv_heads": 2}]
)
@pytest.mark.parametrize("mask", [None, "automatic"])
def test_attention_mask_requires_an_explicit_choice(attention: dict[str, object], mask: str | None):
    authored = {**attention, "implementation": "xla"}
    if mask is not None:
        authored["mask"] = mask
    with pytest.raises(ValidationError, match="mask"):
        _cfg(attention=authored)


def test_mha_arm_cannot_carry_n_kv_heads():
    """The point of the union: a K/V head count can't exist where it has no meaning. Under
    the old flat `n_kv_heads: int | None` this parsed fine and was silently ignored."""
    with pytest.raises(ValidationError, match="n_kv_heads"):
        _cfg(
            attention={
                "mask": "bidirectional",
                "kind": "mha",
                "implementation": "xla",
                "n_heads": 4,
                "n_kv_heads": 2,
            }
        )


def test_gqa_arm_requires_n_kv_heads():
    with pytest.raises(ValidationError, match="n_kv_heads"):
        _cfg(
            attention={
                "mask": "bidirectional",
                "kind": "gqa",
                "implementation": "xla",
                "n_heads": 4,
            }
        )


def test_gqa_refuses_indivisible_kv_heads():
    with pytest.raises(ValidationError, match="divisible"):
        _cfg(
            attention={
                "mask": "bidirectional",
                "kind": "gqa",
                "implementation": "xla",
                "n_heads": 4,
                "n_kv_heads": 3,
            }
        )


def test_gqa_refuses_degenerate_mha():
    """`n_kv_heads == n_heads` IS mha; two spellings of one arch is exactly the ambiguity
    the union removes."""
    with pytest.raises(ValidationError, match="is MHA"):
        _cfg(
            attention={
                "mask": "bidirectional",
                "kind": "gqa",
                "implementation": "xla",
                "n_heads": 4,
                "n_kv_heads": 4,
            }
        )


def test_unknown_attention_kind_refuses():
    with pytest.raises(ValidationError):
        _cfg(attention={"kind": "mqa", "n_heads": 4})


# ----------------------------- ffn -----------------------------


def test_ffn_arms_parse():
    assert isinstance(_cfg(ffn={"kind": "gelu", "hidden": 32}).ffn, GeluCIFnFfnConfig)
    assert isinstance(_cfg(ffn={"kind": "swiglu", "hidden": 22}).ffn, SwigluCIFnFfnConfig)


def test_ffn_requires_a_hidden_width():
    """`hidden` is the FFN's, not the transformer's — it can't be omitted or left floating."""
    with pytest.raises(ValidationError, match="hidden"):
        _cfg(ffn={"kind": "swiglu"})


def test_unknown_ffn_kind_refuses():
    with pytest.raises(ValidationError):
        _cfg(ffn={"kind": "geglu", "hidden": 32})


# ----------------------------- the decomposition.ci union -----------------------------


def _decomposition(ci: dict[str, object]) -> LMDecompositionConfig:
    return LMDecompositionConfig.model_validate(
        {
            "sites": {
                "kind": "glu_transformer",
                "layers": {"kind": "list", "indices": [18]},
                "cs": {"gate": 8},
            },
            "ci": ci,
        }
    )


def test_ci_union_parses_the_chunkwise_arm():
    cfg = _decomposition(
        {
            "type": "chunkwise_transformer",
            "blocks_per_chunk": 1,
            "d_model": 16,
            "n_blocks": 1,
            "attention": {
                "mask": "bidirectional",
                "kind": "mha",
                "implementation": "xla",
                "n_heads": 4,
            },
            "ffn": {"kind": "gelu", "hidden": 32},
        }
    )
    assert isinstance(cfg.ci, ChunkwiseTransformerCIFnConfig)


def test_ci_union_parses_the_global_mlp_arm():
    cfg = _decomposition(
        {"type": "global_mlp", "hidden_dims": [512], "input_tap": "all_block_taps"}
    )
    assert isinstance(cfg.ci, GlobalMlpCIFnConfig)
    assert cfg.ci.hidden_dims == (512,)
    assert cfg.ci.input_tap == "all_block_taps"


def test_ci_union_refuses_an_unknown_type():
    with pytest.raises(ValidationError):
        _decomposition({"type": "layerwise_mlp", "hidden_dims": [512]})


def test_global_mlp_arm_refuses_chunkwise_fields():
    with pytest.raises(ValidationError, match="d_model"):
        _decomposition(
            {
                "type": "global_mlp",
                "hidden_dims": [512],
                "input_tap": "all_block_taps",
                "d_model": 16,
            }
        )


def test_global_mlp_arm_requires_a_hidden_layer():
    with pytest.raises(ValidationError, match="hidden_dims"):
        _decomposition({"type": "global_mlp", "hidden_dims": [], "input_tap": "all_block_taps"})


def test_global_mlp_arm_requires_an_input_tap():
    with pytest.raises(ValidationError, match="input_tap"):
        _decomposition({"type": "global_mlp", "hidden_dims": [512]})


@pytest.mark.parametrize(
    "attention",
    [
        {"mask": "bidirectional", "kind": "mha", "n_heads": 4},
        {"mask": "bidirectional", "kind": "gqa", "n_heads": 4, "n_kv_heads": 2},
    ],
)
@pytest.mark.parametrize("implementation", [None, "auto", "cudnn"])
def test_ci_attention_requires_an_explicit_backend(
    attention: dict[str, object], implementation: str | None
):
    authored = (
        attention if implementation is None else {**attention, "implementation": implementation}
    )
    with pytest.raises(ValidationError, match="implementation"):
        _cfg(attention=authored)


@pytest.mark.parametrize(
    "attention",
    [
        {"mask": "bidirectional", "kind": "mha", "n_heads": 4},
        {"mask": "bidirectional", "kind": "gqa", "n_heads": 4, "n_kv_heads": 2},
    ],
)
@pytest.mark.parametrize("implementation", ["flash", "xla"])
def test_ci_attention_resolves_its_authored_backend(
    attention: dict[str, object], implementation: str
):
    authored = _cfg(attention={**attention, "implementation": implementation})
    resolved = _resolve_ci_fn_attention(authored.attention)
    assert resolved.implementation == implementation
    assert resolved.n_heads == 4
