"""Concrete-path <-> canonical-address round trips per model family."""

import pytest

from param_decomp.targets.qwen36_moe import (
    KIND_ORDER,
    Qwen36MoeConfig,
    full_site_cs,
    layers_of_kind,
    parse_site_name,
    qwen36_35b_a3b_config,
    sublayer_of,
)
from param_decomp.targets.testing import awkward_qwen36_cfg
from param_decomp.topology.canonical import (
    CanonicalWeight,
    DeltaNetWeight,
    LayerWeight,
    MoEExpertsWeight,
    MoESharedWeight,
    SeparateAttnWeight,
)
from param_decomp.topology.path_schemas import path_schema_for_model_type

QWEN36_MOE_CASES = {
    "layers.0.linear_attn.in_proj_qkv.q": "0.deltanet.q",
    "layers.5.linear_attn.in_proj_qkv.v": "5.deltanet.v",
    "layers.5.linear_attn.in_proj_z": "5.deltanet.z",
    "layers.5.linear_attn.in_proj_b": "5.deltanet.b",
    "layers.38.linear_attn.out_proj": "38.deltanet.out",
    "layers.3.self_attn.o_proj": "3.attn.o",
    "layers.3.mlp.experts.gate_proj": "3.moe.gate",
    "layers.0.mlp.experts.up_proj": "0.moe.up",
    "layers.39.mlp.experts.down_proj": "39.moe.down",
    "layers.11.mlp.shared_expert.up_proj": "11.moe_shared.up",
    "layers.11.mlp.shared_expert.gate_proj": "11.moe_shared.gate",
    "layers.2.self_attn.q_proj": "2.attn.q",
    "embed_tokens": "embed",
    "lm_head": "output",
}


@pytest.mark.parametrize(("path", "canonical"), sorted(QWEN36_MOE_CASES.items()))
def test_qwen36_moe_round_trip(path: str, canonical: str) -> None:
    schema = path_schema_for_model_type("Qwen3_5Moe")
    weight = schema.parse_target_path(path)
    assert weight.canonical_str() == canonical
    assert schema.render_canonical_weight(weight) == path
    assert CanonicalWeight.parse(canonical) == weight


EXISTING_FAMILY_CASES = {
    ("Qwen3", "layers.5.mlp.gate_proj"): "5.glu.gate",
    ("Llama", "layers.5.self_attn.o_proj"): "5.attn.o",
    ("LlamaSimple", "h.2.mlp.down_proj"): "2.glu.down",
    ("LlamaSimpleMLP", "h.2.mlp.c_fc"): "2.mlp.up",
    ("GPT2", "h_torch.1.attn.c_attn"): "1.attn_fused.qkv",
    ("GPT2Simple", "h.0.mlp.down_proj"): "0.mlp.down",
}


@pytest.mark.parametrize(
    ("model_type", "path", "canonical"),
    sorted((model, path, canonical) for (model, path), canonical in EXISTING_FAMILY_CASES.items()),
)
def test_existing_families_round_trip(model_type: str, path: str, canonical: str) -> None:
    schema = path_schema_for_model_type(model_type)
    weight = schema.parse_target_path(path)
    assert weight.canonical_str() == canonical
    assert schema.render_canonical_weight(weight) == path


@pytest.mark.parametrize(
    "cfg", [qwen36_35b_a3b_config(), awkward_qwen36_cfg()], ids=["qwen36_35b", "qwanito_e42"]
)
def test_qwen36_moe_schema_accepts_every_site_the_target_emits(cfg: Qwen36MoeConfig) -> None:
    """Every site name the target can emit — each kind on every layer that has it — parses
    through the schema into its kind's sublayer, round-trips, and re-parses canonically."""
    schema = path_schema_for_model_type("Qwen3_5Moe")
    sites = full_site_cs(cfg, {kind: 1 for kind in KIND_ORDER})
    assert len(sites) == sum(len(layers_of_kind(cfg, kind)) for kind in KIND_ORDER)
    for site in sites:
        layer, kind = parse_site_name(site.name)
        weight = schema.parse_target_path(site.name)
        assert isinstance(weight, LayerWeight) and weight.layer_idx == layer, site.name
        assert schema.render_canonical_weight(weight) == site.name
        assert CanonicalWeight.parse(weight.canonical_str()) == weight
        match sublayer_of(kind):
            case "deltanet":
                assert isinstance(weight.name, DeltaNetWeight), site.name
            case "attn":
                assert isinstance(weight.name, SeparateAttnWeight), site.name
            case "moe":
                assert isinstance(weight.name, MoEExpertsWeight | MoESharedWeight), site.name


def test_unknown_block_path_refuses() -> None:
    schema = path_schema_for_model_type("Qwen3_5Moe")
    for path in (
        "layers.3.mlp.gate_proj",
        "layers.3.linear_attn.in_proj_qkv",
        "layers.3.linear_attn.q_proj",
        "layers.3.self_attn.in_proj_z",
    ):
        with pytest.raises(AssertionError):
            schema.parse_target_path(path)
    with pytest.raises(AssertionError):
        path_schema_for_model_type("Qwen3").parse_target_path("layers.3.linear_attn.in_proj_z")


@pytest.mark.parametrize(
    ("model_type", "canonical"),
    [
        ("GPT2", "0.attn.q"),
        ("Llama", "0.attn_fused.qkv"),
        ("Qwen3", "0.deltanet.q"),
        ("Qwen3_5Moe", "0.glu.up"),
        ("LlamaSimpleMLP", "0.glu.gate"),
    ],
)
def test_render_refuses_weights_outside_the_model_family(model_type: str, canonical: str) -> None:
    schema = path_schema_for_model_type(model_type)
    with pytest.raises(AssertionError):
        schema.render_canonical_weight(CanonicalWeight.parse(canonical))
