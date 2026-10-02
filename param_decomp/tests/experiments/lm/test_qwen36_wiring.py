"""The qwen36_moe composition wiring: authored config → resolved decomposition, for both
weight sources (the 35B HF snapshot; a lab-pretrained toy's pretrain-cache entry).

Resolution-level only — no weights are loaded. The tiny-model forward and the
expert-blocked engine behavior are pinned by the target and core suites."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import yaml
from safetensors.numpy import save_file

from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.chunkwise.arch import ChunkwiseTransformerCIFnArch
from param_decomp.core.components import BlockedFactorization, DenseFactorization
from param_decomp.experiments.lm.config import (
    QWEN36_MOE_MODEL_CLASS,
    QWEN36_MOE_MODEL_NAME,
    LMDecompositionConfig,
    LMTargetConfig,
    MoEChunkwiseTransformerCIFnConfig,
    PretrainedQwen35MoeTarget,
    resolve_decomposition,
    resolve_lm_ci_fn_arch,
)
from param_decomp.experiments.lm.resolved import (
    HFSnapshotWeights,
    PretrainCacheWeights,
    Qwen36MoeTargetConfig,
)
from param_decomp.infra import pretrain_cache
from param_decomp.targets.lm_output import MaterializedOutputEdge
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeConfig,
    qwen36_35b_a3b_config,
    qwen36_moe_config_from_pretrain_cache,
)
from param_decomp.targets.testing import TINY_QWEN36_CS, tiny_qwen36_cfg

N_LAYER, N_EXPERTS, D_RESID = 40, 256, 2048
FULL_ATTENTION_INTERVAL = 4


def _target_config(model_class: str = QWEN36_MOE_MODEL_CLASS) -> LMTargetConfig:
    return LMTargetConfig.model_validate(
        {
            "spec": {
                "kind": "hf",
                "model_class": model_class,
                "model_name": QWEN36_MOE_MODEL_NAME,
            },
            "attention_implementation": "xla",
            "weights_dtype": "bfloat16",
            "output_edge": {"kind": "materialized"},
        }
    )


def _decomposition(
    cs: dict[str, int],
    layers: dict[str, object] | None = None,
    input_tap: str = "first_block_resid",
    initialization: str = "random",
    blocks_per_chunk: int = FULL_ATTENTION_INTERVAL,
) -> LMDecompositionConfig:
    return LMDecompositionConfig.model_validate(
        {
            "sites": {
                "kind": "qwen36_moe",
                "layers": layers if layers is not None else {"kind": "all"},
                "cs": cs,
                "initialization": initialization,
            },
            "ci": {
                "type": "chunkwise_transformer",
                "blocks_per_chunk": blocks_per_chunk,
                "input_tap": input_tap,
                "d_model": 8,
                "n_blocks": 1,
                "attention": {
                    "mask": "bidirectional",
                    "kind": "mha",
                    "implementation": "xla",
                    "n_heads": 2,
                },
                "ffn": {"kind": "gelu", "hidden": 16},
            },
        }
    )


def test_qwen36_resolution_builds_expert_blocked_specs(tmp_path: Path):
    cs = {"experts_gate": 512, "experts_down": 256, "shared_gate": 4}
    resolved = resolve_decomposition(_target_config(), _decomposition(cs), tmp_path)

    target = resolved.target
    assert isinstance(target, Qwen36MoeTargetConfig)
    assert target.weights == HFSnapshotWeights(QWEN36_MOE_MODEL_NAME)
    assert target.arch == qwen36_35b_a3b_config()
    # The authored attention implementation must reach the resolved target unchanged.
    assert target.attention_implementation == "xla"
    assert len(target.sites) == N_LAYER * len(cs)
    assert len(resolved.tree.blocks) == N_LAYER

    by_group = {spec.group: spec.factorization for spec in resolved.site_specs}
    assert by_group["experts_gate"] == BlockedFactorization(
        n_blocks=N_EXPERTS, d_in=D_RESID, d_out=512, c_per_block=2
    )
    assert by_group["experts_down"] == BlockedFactorization(
        n_blocks=N_EXPERTS, d_in=512, d_out=D_RESID, c_per_block=1
    )
    assert isinstance(by_group["shared_gate"], DenseFactorization)

    ci_fn_arch = resolve_lm_ci_fn_arch(resolved, _decomposition(cs).ci)
    assert isinstance(ci_fn_arch, ChunkwiseTransformerCIFnArch)
    assert len(ci_fn_arch.chunks) == N_LAYER // 4
    assert ci_fn_arch.input_dim == D_RESID


def _write_pretrain_cache(cache_dir: Path, arch: Qwen36MoeConfig) -> None:
    """A pretrain-cache entry in the pretrainer's layout: `model_config.yaml` speaks the
    pretrainer's field names (`n_experts_per_tok`, `n_ctx`, plus its training-only facts)
    beside one `model_step_*.safetensors`."""
    renamed = {"n_experts_per_token": "n_experts_per_tok", "max_position_embeddings": "n_ctx"}
    pretrainer_config = {
        "model_type": "Qwen35Moe",
        "block_size": arch.max_position_embeddings,
        "router_aux_loss_coef": 0.01,
        "router_z_loss_coef": 0.001,
        "tied_head": False,
        **{renamed.get(f.name, f.name): getattr(arch, f.name) for f in fields(arch)},
    }
    cache_dir.mkdir(parents=True)
    (cache_dir / "model_config.yaml").write_text(yaml.safe_dump(pretrainer_config))
    save_file(
        {"lm_head.weight": np.zeros((arch.vocab_size, arch.n_embd), np.float32)},
        cache_dir / "model_step_100.safetensors",
    )


def test_pretrain_cache_arch_reads_the_pretrainer_vocabulary(tmp_path: Path):
    """The toy's arch comes from its cache entry's yaml under the pretrainer's spellings;
    the training-only facts are dropped; a foreign `model_type` refuses."""
    cache_dir = tmp_path / "param-decomp-t-tiny"
    _write_pretrain_cache(cache_dir, tiny_qwen36_cfg())
    assert qwen36_moe_config_from_pretrain_cache(cache_dir) == tiny_qwen36_cfg()

    foreign = yaml.safe_load((cache_dir / "model_config.yaml").read_text())
    foreign["model_type"] = "LlamaSimpleMLP"
    (cache_dir / "model_config.yaml").write_text(yaml.safe_dump(foreign))
    with pytest.raises(AssertionError, match="LlamaSimpleMLP"):
        qwen36_moe_config_from_pretrain_cache(cache_dir)


@pytest.mark.parametrize(
    "implementation", ["dense_masked", "ragged_dot", "tokamax", "tokamax_split_vjp"]
)
@pytest.mark.parametrize("ci_implementation", ["dense_masked", "ragged_dot"])
def test_pretrained_target_resolves_against_its_cache(
    tmp_path: Path, implementation: str, ci_implementation: str
):
    """The cached architecture determines target sites and the MoE CI function's shape."""
    arch = tiny_qwen36_cfg()
    spec = PretrainedQwen35MoeTarget(run_path="test/pretrain/t-00000000")
    target_config = LMTargetConfig.model_validate(
        {
            "spec": spec,
            "attention_implementation": "xla",
            "expert_implementation": implementation,
            "weights_dtype": "bfloat16",
            "output_edge": {"kind": "materialized"},
        }
    )
    decomposition = LMDecompositionConfig.model_validate(
        {
            "sites": {"kind": "qwen36_moe", "layers": {"kind": "all"}, "cs": TINY_QWEN36_CS},
            "ci": {
                "type": "moe_chunkwise_transformer",
                "expert_implementation": ci_implementation,
                "blocks_per_chunk": arch.full_attention_interval,
                "input_tap": "first_block_resid",
                "d_model": 8,
                "n_blocks": 2,
                "attention": {
                    "kind": "mha",
                    "implementation": "xla",
                    "mask": "bidirectional",
                    "n_heads": 2,
                },
                "expert_ffn_hidden": arch.moe_intermediate,
                "shared_ffn_hidden": arch.shared_expert_intermediate,
            },
        }
    )
    _write_pretrain_cache(pretrain_cache.cache_dir_for_run(tmp_path, spec.run_path), arch)

    resolved = resolve_decomposition(target_config, decomposition, tmp_path)
    target = resolved.target
    assert isinstance(target, Qwen36MoeTargetConfig)
    assert target.arch == arch
    assert target.weights == PretrainCacheWeights(spec.run_path)
    assert target.output_edge == MaterializedOutputEdge()
    assert target.expert_implementation == target_config.expert_implementation
    assert len(target.sites) == arch.n_layer * len(TINY_QWEN36_CS)

    ci_fn_arch = resolve_lm_ci_fn_arch(resolved, decomposition.ci)
    assert isinstance(ci_fn_arch, BlockSelectedChunkwiseTransformerCIFnArch)
    assert [chunk.layers for chunk in ci_fn_arch.chunks] == [(0, 1), (2, 3)]
    assert ci_fn_arch.table_size == arch.n_experts
    assert ci_fn_arch.input_dim == arch.n_embd
    assert isinstance(decomposition.ci, MoEChunkwiseTransformerCIFnConfig)
    assert ci_fn_arch.expert_implementation == decomposition.ci.expert_implementation


def test_qwen36_nonlinearity_aligned_initialization_resolves(tmp_path: Path):
    resolved = resolve_decomposition(
        _target_config(),
        _decomposition(
            {"experts_gate": 512, "experts_down": 256, "shared_gate": 4},
            initialization="nonlinearity_aligned",
        ),
        tmp_path,
    )

    assert isinstance(resolved.target, Qwen36MoeTargetConfig)
    assert resolved.target.component_initialization == "nonlinearity_aligned"


def _is_full_attention(layer: int) -> bool:
    return (layer + 1) % FULL_ATTENTION_INTERVAL == 0


def test_qwen36_mixer_kinds_resolve_per_block_slots(tmp_path: Path):
    """A block carries the authored kinds its layer HAS: DeltaNet kinds on the
    non-full-attention layers, attention kinds on the full-attention layers, MoE kinds
    everywhere. Chunks aligned to the stage are homogeneous; a chunk length off the
    stage period mixes block shapes and refuses at resolve."""
    cs = {"gdn_q": 8, "gdn_out": 8, "attn_k": 8, "attn_o": 8, "experts_gate": 512, "shared_down": 4}
    resolved = resolve_decomposition(_target_config(), _decomposition(cs), tmp_path)

    assert [block.layer_idx for block in resolved.tree.blocks] == list(range(N_LAYER))
    for block in resolved.tree.blocks:
        mixer = (
            (("attn_k", 8), ("attn_o", 8))
            if _is_full_attention(block.layer_idx)
            else (("gdn_q", 8), ("gdn_out", 8))
        )
        assert block.slots == (*mixer, ("experts_gate", 512), ("shared_down", 4)), block
    assert len(resolved.target.sites) == N_LAYER * 4
    alignments = {spec.group: spec.alignment for spec in resolved.site_specs}
    for group, unit_kind in {
        "attn_k": "attention_head",
        "gdn_q": "deltanet_head",
        "gdn_out": "deltanet_head",
        "attn_o": "attention_head",
        "shared_down": "neuron",
    }.items():
        alignment = alignments[group]
        assert alignment is not None and alignment.partition.unit_kind == unit_kind

    stage_aligned = resolve_lm_ci_fn_arch(resolved, _decomposition(cs).ci)
    assert isinstance(stage_aligned, ChunkwiseTransformerCIFnArch)
    assert len(stage_aligned.chunks) == N_LAYER // FULL_ATTENTION_INTERVAL
    assert all(len(chunk.output_sites) == 16 for chunk in stage_aligned.chunks)
    with pytest.raises(AssertionError, match="structurally identical"):
        resolve_lm_ci_fn_arch(resolved, _decomposition(cs, blocks_per_chunk=2).ci)


def test_qwen36_one_mixer_alone_leaves_the_other_mixer_blocks_empty(tmp_path: Path):
    gdn_only = resolve_decomposition(_target_config(), _decomposition({"gdn_v": 8}), tmp_path)
    for block in gdn_only.tree.blocks:
        assert block.slots == (() if _is_full_attention(block.layer_idx) else (("gdn_v", 8),))
    assert len(gdn_only.target.sites) == N_LAYER - N_LAYER // FULL_ATTENTION_INTERVAL
    ci_fn_arch = resolve_lm_ci_fn_arch(gdn_only, _decomposition({"gdn_v": 8}).ci)
    assert isinstance(ci_fn_arch, ChunkwiseTransformerCIFnArch)
    assert all(len(chunk.output_sites) == 3 for chunk in ci_fn_arch.chunks)
    with pytest.raises(AssertionError, match="structurally identical"):
        resolve_lm_ci_fn_arch(gdn_only, _decomposition({"gdn_v": 8}, blocks_per_chunk=1).ci)


def test_qwen36_all_block_taps_reads_the_site_inputs_each_layer_computes(tmp_path: Path):
    """No dense MLP hidden exists here, so a chunk reads each layer's mixer input, mixer
    core, and MoE input; the mixer core's width follows the layer's mixer."""
    decomposition = _decomposition({"experts_gate": 512}, input_tap="all_block_taps")
    resolved = resolve_decomposition(_target_config(), decomposition, tmp_path)
    ci_fn_arch = resolve_lm_ci_fn_arch(resolved, decomposition.ci)
    assert isinstance(ci_fn_arch, ChunkwiseTransformerCIFnArch)
    arch = qwen36_35b_a3b_config()
    first_stage = range(FULL_ATTENTION_INTERVAL)
    assert ci_fn_arch.chunks[0].input_taps == tuple(
        f"{tap}.{layer}" for layer in first_stage for tap in ("attn_in", "attn_out", "mlp_in")
    )
    mixer_core_widths = [
        arch.n_head * arch.head_dim if _is_full_attention(layer) else arch.linear_value_dim
        for layer in first_stage
    ]
    assert ci_fn_arch.input_dim == 2 * D_RESID * FULL_ATTENTION_INTERVAL + sum(mixer_core_widths)


def test_qwen36_resolution_refusals(tmp_path: Path):
    cs = {"experts_gate": 512}
    with pytest.raises(AssertionError, match="every layer that has it"):
        resolve_decomposition(
            _target_config(),
            _decomposition(cs, layers={"kind": "range", "start": 0, "end": 8}),
            tmp_path,
        )
    with pytest.raises(AssertionError):
        resolve_decomposition(
            _target_config(model_class="transformers.Qwen3ForCausalLM"),
            _decomposition(cs),
            tmp_path,
        )
    with pytest.raises(AssertionError, match="multiple of n_experts"):
        resolve_decomposition(_target_config(), _decomposition({"experts_gate": 300}), tmp_path)


def test_qwen36_output_edge_resolves_and_gates(tmp_path: Path):
    """`target.output_edge` is authored on every config and reaches the resolved target
    as the typed edge; an omitted edge refuses at parse and a non-dividing chunk
    count refuses at resolve."""
    from pydantic import ValidationError

    from param_decomp.targets.lm_output import MaterializedOutputEdge, StreamedOutputEdge

    cs = {"experts_gate": 512}
    materialized = resolve_decomposition(_target_config(), _decomposition(cs), tmp_path).target
    assert isinstance(materialized, Qwen36MoeTargetConfig)
    assert materialized.output_edge == MaterializedOutputEdge()

    unauthored = {
        key: value
        for key, value in _target_config().model_dump(mode="json").items()
        if key != "output_edge"
    }
    with pytest.raises(ValidationError, match="output_edge"):
        LMTargetConfig.model_validate(unauthored)

    streamed_config = LMTargetConfig.model_validate(
        {
            **_target_config().model_dump(mode="json"),
            "output_edge": {"kind": "streamed", "n_vocab_chunks": 32},
        }
    )
    streamed = resolve_decomposition(streamed_config, _decomposition(cs), tmp_path).target
    assert isinstance(streamed, Qwen36MoeTargetConfig)
    assert streamed.output_edge == StreamedOutputEdge(n_vocab_chunks=32)

    non_dividing = LMTargetConfig.model_validate(
        {
            **_target_config().model_dump(mode="json"),
            "output_edge": {"kind": "streamed", "n_vocab_chunks": 7},
        }
    )
    with pytest.raises(AssertionError, match="must divide the"):
        resolve_decomposition(non_dividing, _decomposition(cs), tmp_path)


@pytest.mark.parametrize(
    ("model_name", "model_class"),
    [
        ("meta-llama/Llama-3.1-8B", "transformers.LlamaForCausalLM"),
        ("Qwen/Qwen3-8B-Base", "transformers.Qwen3ForCausalLM"),
    ],
)
def test_glu_output_edge_resolves_and_checks_vocab(
    tmp_path: Path, model_name: str, model_class: str
) -> None:
    from param_decomp.experiments.lm.config import GluTransformerCSpec
    from param_decomp.experiments.lm.resolved import TargetConfig
    from param_decomp.targets.lm_output import MaterializedOutputEdge, StreamedOutputEdge

    glu_streamed = LMTargetConfig.model_validate(
        {
            "spec": {
                "kind": "hf",
                "model_class": model_class,
                "model_name": model_name,
            },
            "attention_implementation": "xla",
            "weights_dtype": "bfloat16",
            "output_edge": {"kind": "streamed", "n_vocab_chunks": 32},
        }
    )
    glu_decomposition = LMDecompositionConfig.model_validate(
        {
            "sites": {
                "kind": "glu_transformer",
                "layers": {"kind": "all"},
                "cs": {"q": 4},
            },
            "ci": {
                "type": "chunkwise_transformer",
                "blocks_per_chunk": 4,
                "input_tap": "first_block_resid",
                "d_model": 8,
                "n_blocks": 1,
                "attention": {
                    "mask": "bidirectional",
                    "kind": "mha",
                    "implementation": "xla",
                    "n_heads": 2,
                },
                "ffn": {"kind": "gelu", "hidden": 16},
            },
        }
    )
    assert isinstance(glu_decomposition.sites, GluTransformerCSpec)
    streamed = resolve_decomposition(glu_streamed, glu_decomposition, tmp_path).target
    assert isinstance(streamed, TargetConfig)
    assert streamed.output_edge == StreamedOutputEdge(n_vocab_chunks=32)
    for edge, expected in [
        ({"kind": "materialized"}, MaterializedOutputEdge()),
        ({"kind": "streamed", "n_vocab_chunks": 1}, StreamedOutputEdge(n_vocab_chunks=1)),
    ]:
        config = LMTargetConfig.model_validate(
            {**glu_streamed.model_dump(mode="json"), "output_edge": edge}
        )
        assert (
            resolve_decomposition(config, glu_decomposition, tmp_path).target.output_edge
            == expected
        )
    non_dividing = LMTargetConfig.model_validate(
        {
            **glu_streamed.model_dump(mode="json"),
            "output_edge": {"kind": "streamed", "n_vocab_chunks": 7},
        }
    )
    with pytest.raises(AssertionError, match="must divide the"):
        resolve_decomposition(non_dividing, glu_decomposition, tmp_path)
