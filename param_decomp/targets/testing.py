"""Tiny random targets — the standard CPU-test fixtures.

One tiny random target per LM family (the Llama-3.1-flavored GLU transformer and the
`LlamaSimpleMLP`), plus a one-chunk chunkwise CI fn over each. Engine tests use these as
the concrete target behind the `DecomposedModel` protocol; the per-target suites
(`param_decomp/tests/targets/`) use them as the system under test. Toy dims throughout —
no real weights, no GPU.
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace

import jax
import jax.numpy as jnp

from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunk,
    BlockSelectedChunkwiseTransformerCIFn,
    BlockSelectedChunkwiseTransformerCIFnArch,
    FullSlot,
    SelectedSlot,
)
from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerBackbone,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.backbone import BackboneCIFn
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.components import BlockSelection, SelectedCI, SiteC, SiteCI, SiteSpec
from param_decomp.core.model import (
    ComponentActivations,
    DecomposedModel,
    MaterializedMasking,
    SiteRoutes,
)
from param_decomp.target_ports.llama import LlamaConfig, llama3_inv_freq
from param_decomp.targets import llama_simple_mlp, qwen36_moe
from param_decomp.targets.llama_simple_mlp import (
    SIMPLE_MLP_KINDS,
    LlamaSimpleMLPConfig,
    build_decomposed_simple_mlp,
)
from param_decomp.targets.lm_output import LMOutput, MaterializedOutputEdge
from param_decomp.targets.qwen36_moe import (
    AttnSublayer,
    DeltaNetSublayer,
    FrozenMoE,
    GatedAttention,
    GatedDeltaNet,
    Qwen36MoeConfig,
    Qwen36MoeDecomposedModel,
    build_qwen36_moe_model,
    layer_is_full_attention,
)
from param_decomp.targets.transformer import (
    GLU_MLP_KINDS,
    FrozenAttn,
    GatedMLP,
    PlainMLP,
    TransformerDecomposedModel,
    TransformerLayer,
    build_decomposed_lm,
    parse_site_name,
)
from param_decomp.targets.transformer_taps import resid_tap_key


def _tiny_chunkwise_ci_fn_arch(
    model: TransformerDecomposedModel, first_block: int, input_dim: int, n_blocks: int
) -> ChunkwiseTransformerCIFnArch:
    """One chunk reading the residual entering the first decomposed block, emitting CI
    for every site. `input_dim` is the target residual width (`n_embd`)."""
    return ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=model.site_names),),
        input_dim=input_dim,
        d_model=16,
        n_blocks=n_blocks,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )


def _tiny_chunkwise_ci_fn(
    model: TransformerDecomposedModel,
    key: jax.Array,
    first_block: int,
    input_dim: int,
    n_blocks: int,
) -> BackboneCIFn:
    arch = _tiny_chunkwise_ci_fn_arch(model, first_block, input_dim, n_blocks)
    return arch.initialize(model.sites, None, key)


def chunkwise_transformer_backbone(ci_fn: object) -> ChunkwiseTransformerBackbone:
    """`ci_fn`'s backbone, asserted to be a chunkwise transformer's."""
    assert isinstance(ci_fn, BackboneCIFn), type(ci_fn)
    backbone = ci_fn.backbone
    assert isinstance(backbone, ChunkwiseTransformerBackbone), type(backbone)
    return backbone


def tiny_glu_cfg() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        n_layer=8,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        max_position_embeddings=512,
        rope_factor=8.0,
        rope_low_freq_factor=1.0,
        rope_high_freq_factor=4.0,
        rope_original_max_position_embeddings=128,
    )


def tiny_glu_decomposed_lm(
    cfg: LlamaConfig, sites: tuple[SiteSpec, ...], key: jax.Array
) -> TransformerDecomposedModel:
    """A tiny random `TransformerDecomposedModel` (random embedding + full frozen layer stack
    plus the decomposition `sites`) — the CPU-test analog of `load_decomposed_lm_from_hf`."""
    ks = iter(jax.random.split(key, 1024))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def n(shape: tuple[int, ...], s: float | None = None) -> jax.Array:
        return jax.random.normal(next(ks), shape) * (s or d**-0.5)

    def fattn():
        return FrozenAttn(
            n((qd, d)), n((kvd, d)), n((kvd, d)), n((d, qd)),
            cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.n_rep, "xla",
        )  # fmt: skip

    def layer():
        # attn drawn before the MLP: the key-consumption order the committed
        # fixture-derived tests (slow-eval histograms) were pinned against.
        attn = fattn()
        mlp = GatedMLP(kinds=GLU_MLP_KINDS, Wg=n((di, d)), Wu=n((di, d)), Wd=n((d, di)))
        return TransformerLayer(jnp.ones((d,)), jnp.ones((d,)), attn, mlp)

    return build_decomposed_lm(
        embed=n((cfg.vocab_size, d), 0.02),
        layers=[layer() for _ in range(cfg.n_layer)],
        norm=jnp.ones((d,)),
        lm_head=n((cfg.vocab_size, d), 0.02),
        inv_freq=llama3_inv_freq(cfg),
        cfg=cfg,
        sites=sites,
        output_edge=MaterializedOutputEdge(),
    )


def tiny_glu_chunkwise_ci_fn_arch(
    model: TransformerDecomposedModel, n_blocks: int
) -> ChunkwiseTransformerCIFnArch:
    first_block = min(parse_site_name(n)[0] for n in model.site_names)
    return _tiny_chunkwise_ci_fn_arch(model, first_block, tiny_glu_cfg().n_embd, n_blocks)


def tiny_glu_chunkwise_ci_fn(
    model: TransformerDecomposedModel, key: jax.Array, n_blocks: int
) -> BackboneCIFn:
    first_block = min(parse_site_name(n)[0] for n in model.site_names)
    return _tiny_chunkwise_ci_fn(model, key, first_block, tiny_glu_cfg().n_embd, n_blocks)


def tiny_simple_mlp_cfg() -> LlamaSimpleMLPConfig:
    return LlamaSimpleMLPConfig(
        vocab_size=64,
        n_layer=6,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        n_intermediate=64,
        rotary_base=10000.0,
        rms_norm_eps=1e-6,
        n_ctx=64,
    )


def _tiny_simple_mlp_layers(
    cfg: LlamaSimpleMLPConfig, n: int, key: jax.Array
) -> list[TransformerLayer]:
    ks = iter(jax.random.split(key, 1024))
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def rand(shape: tuple[int, ...]) -> jax.Array:
        return jax.random.normal(next(ks), shape) * d**-0.5

    def layer() -> TransformerLayer:
        # attn drawn before the MLP: the key-consumption order the committed
        # fixture-derived tests (slow-eval histograms) were pinned against.
        attn = FrozenAttn(
            rand((qd, d)),
            rand((kvd, d)),
            rand((kvd, d)),
            rand((d, qd)),
            cfg.n_head,
            cfg.n_kv_head,
            cfg.head_dim,
            cfg.n_rep,
            "xla",
        )
        mlp = PlainMLP(kinds=SIMPLE_MLP_KINDS, Wfc=rand((di, d)), Wdown=rand((d, di)))
        return TransformerLayer(ln1=jnp.ones((d,)), ln2=jnp.ones((d,)), attn=attn, mlp=mlp)

    return [layer() for _ in range(n)]


def tiny_simple_mlp_decomposed_model(
    cfg: LlamaSimpleMLPConfig, sites: tuple[SiteSpec, ...], key: jax.Array
) -> TransformerDecomposedModel:
    """A tiny random engine-hosted SimpleMLP carrying a random (tied) embedding + full
    frozen layer stack plus the decomposition `sites`."""
    layers_key, embed_key = jax.random.split(key)
    layers = _tiny_simple_mlp_layers(cfg, cfg.n_layer, layers_key)
    embed = jax.random.normal(embed_key, (cfg.vocab_size, cfg.n_embd)) * 0.02
    return build_decomposed_simple_mlp(
        embed=embed, layers=layers, norm=jnp.ones((cfg.n_embd,)),
        cfg=cfg, sites=sites, output_edge=MaterializedOutputEdge(),
    )  # fmt: skip


SIMPLE_MLP_MIXED_SITE_CS = (
    SiteC("h.2.attn.q_proj", 8),
    SiteC("h.2.attn.v_proj", 12),
    SiteC("h.2.mlp.c_fc", 8),
    SiteC("h.2.mlp.down_proj", 16),
    SiteC("h.3.attn.q_proj", 8),
    SiteC("h.3.attn.v_proj", 12),
    SiteC("h.3.mlp.c_fc", 8),
    SiteC("h.3.mlp.down_proj", 16),
)
"""Mixed attention + MLP kinds with heterogeneous per-kind C, RECTANGULAR over the
contiguous layer range 2..3 — the engine's segmented masked forward requires every
decomposed kind on every decomposed layer."""


def tiny_simple_mlp_chunkwise_ci_fn(
    model: TransformerDecomposedModel, key: jax.Array
) -> BackboneCIFn:
    first_block = min(llama_simple_mlp.parse_site_name(n)[0] for n in model.site_names)
    return _tiny_chunkwise_ci_fn(model, key, first_block, tiny_simple_mlp_cfg().n_embd, 2)


# These projections deliberately take the RAW `DecomposedModel`, not the `PlacedModel`
# bundle: the per-target suites exercise the protocol surface itself — per-call
# `placement`, here the unplaced (None) CPU-test execution — below the engine's
# one-bundle assembly.


def run_clean[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT](
    model: DecomposedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    inputs: TargetIn,
) -> Out:
    """Test-only output projection of the capture-aware clean forward."""
    return model.clean_forward(inputs, placement=None).output


def materialized_logits(output: LMOutput) -> jax.Array:
    """The materialized edge's logits — the tests that do array arithmetic on an LM output
    narrow through this; a streamed package refuses."""
    assert isinstance(output, jax.Array), type(output).__name__
    return output


def capture_clean[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT](
    model: DecomposedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    inputs: TargetIn,
    keys: Iterable[str],
) -> dict[str, jax.Array]:
    return model.clean_forward(inputs, frozenset(keys), placement=None).captures


def run_masked[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT](
    model: DecomposedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    prepared_weights: PreparedT,
    conditioning: Conditioning,
    masking: MaterializedMasking,
    *,
    routes: SiteRoutes | None,
    remat: bool,
) -> Out:
    """Test-only output projection of a complete masked forward."""
    return model.masked_forward(
        prepared_weights,
        conditioning,
        masking=model.prepare_masking(masking),
        routes=routes,
        placement=None,
        remat=remat,
    ).output


def capture_site_outputs[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model: DecomposedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    prepared_weights: PreparedT,
    conditioning: Conditioning,
    masking: MaterializedMasking,
    *,
    routes: SiteRoutes | None,
) -> dict[str, jax.Array]:
    sites = tuple(masking.component_masks)
    site_output_keys = model.site_output_keys(sites)
    masked_forward_result = model.masked_forward(
        prepared_weights,
        conditioning,
        masking=model.prepare_masking(masking),
        routes=routes,
        placement=None,
        capture_keys=frozenset(site_output_keys),
        remat=False,
    )
    return {
        site: masked_forward_result.captures[key]
        for site, key in zip(sites, site_output_keys, strict=True)
    }


def tiny_qwen36_cfg() -> Qwen36MoeConfig:
    """Two stages of one DeltaNet + one attention layer — the fewest that exercise both
    mixers and a stage boundary. Every dimension is distinct so no two axes coincide;
    site `d_in`s (`n_embd`, the two hidden widths) tile the (data=4, tp=2) test mesh."""
    return Qwen36MoeConfig(
        vocab_size=40,
        n_layer=4,
        full_attention_interval=2,
        n_embd=12,
        n_head=2,
        n_kv_head=1,
        head_dim=16,
        partial_rotary_factor=0.25,
        rope_theta=10_000_000.0,
        linear_num_key_heads=2,
        linear_key_head_dim=6,
        linear_num_value_heads=4,
        linear_value_head_dim=5,
        linear_conv_kernel_dim=3,
        n_experts=4,
        n_experts_per_token=2,
        moe_intermediate=8,
        shared_expert_intermediate=20,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
    )


# Expert-site Cs must be multiples of n_experts=4: components are expert-local, so a
# site's flat C is n_experts * c_per_expert by construction.
TINY_QWEN36_CS: dict[str, int] = {
    "experts_gate": 8,
    "experts_up": 8,
    "experts_down": 12,
    "shared_gate": 4,
    "shared_up": 4,
    "shared_down": 5,
}


def exact_width_cs(cfg: Qwen36MoeConfig, kinds: Iterable[str]) -> dict[str, int]:
    """Each kind at the width of the axis its nonlinearity-aligned init aligns on — the input
    axis for the residual writers, the output axis otherwise — the C at which that init
    is the exact one-coordinate factorization."""

    def width(kind: str) -> int:
        dims = qwen36_moe.site_dims(cfg, kind)
        match qwen36_moe.nonlinearity_alignment(cfg, kind).side:
            case "input":
                return dims.d_in
            case "output":
                return dims.d_out

    return {kind: width(kind) for kind in kinds}


TINY_QWEN36_MIXER_CS: dict[str, int] = exact_width_cs(
    tiny_qwen36_cfg(),
    [kind for kind in qwen36_moe.KIND_ORDER if qwen36_moe.sublayer_of(kind) != "moe"],
)
"""The eleven token-mixer kinds at exact width."""

TINY_QWEN36_ALL_CS: dict[str, int] = {**TINY_QWEN36_MIXER_CS, **TINY_QWEN36_CS}
"""All seventeen kinds in `KIND_ORDER`: the mixers at exact width, the MoE at `TINY_QWEN36_CS`."""


def awkward_qwen36_cfg() -> Qwen36MoeConfig:
    """`tiny_qwen36_cfg` on the parameter-matched pretrained toy's expert grid — 42 routed
    experts, top-8 — a non-power-of-two expert count with k > 2, where slot bookkeeping
    that happens to hold at E=4/k=2 cannot coincide. Not for the (data=4, tp=2) mesh:
    42 experts split tp=2 but the fused axis 294 does not tile it with the tiny d_in."""
    return replace(tiny_qwen36_cfg(), n_experts=42, n_experts_per_token=8, moe_intermediate=7)


AWKWARD_QWEN36_CS: dict[str, int] = {
    "experts_gate": 84,
    "experts_up": 84,
    "experts_down": 126,
    "shared_gate": 4,
    "shared_up": 4,
    "shared_down": 5,
}


def tiny_qwen36_decomposed_model(
    cfg: Qwen36MoeConfig, sites: tuple[SiteSpec, ...], key: jax.Array
) -> Qwen36MoeDecomposedModel:
    """A tiny random qwen36_moe model — the CPU-test analog of
    `load_decomposed_qwen36_moe_from_hf`. Norm weights are drawn non-trivially in each
    convention (zero-centered near 0, the gated norm near 1) so a conflated convention
    shows."""
    ks = iter(jax.random.split(key, 4096))
    d = cfg.n_embd
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    fused = cfg.n_experts * cfg.moe_intermediate

    def n(shape: tuple[int, ...], scale: float | None = None) -> jax.Array:
        return jax.random.normal(next(ks), shape) * (scale or d**-0.5)

    def centered_norm(shape: tuple[int, ...]) -> jax.Array:
        return 0.1 * jax.random.normal(next(ks), shape)

    def deltanet_sublayer() -> DeltaNetSublayer:
        return DeltaNetSublayer(
            ln1=centered_norm((d,)),
            mixer=GatedDeltaNet(
                w_q=n((cfg.linear_key_dim, d)),
                w_k=n((cfg.linear_key_dim, d)),
                w_v=n((cfg.linear_value_dim, d)),
                w_z=n((cfg.linear_value_dim, d)),
                w_b=n((cfg.linear_num_value_heads, d)),
                w_a=n((cfg.linear_num_value_heads, d)),
                conv_q=n((cfg.linear_key_dim, cfg.linear_conv_kernel_dim), 0.5),
                conv_k=n((cfg.linear_key_dim, cfg.linear_conv_kernel_dim), 0.5),
                conv_v=n((cfg.linear_value_dim, cfg.linear_conv_kernel_dim), 0.5),
                a_log=0.3 * jax.random.normal(next(ks), (cfg.linear_num_value_heads,)),
                dt_bias=1.0 + 0.1 * jax.random.normal(next(ks), (cfg.linear_num_value_heads,)),
                norm_w=1.0 + 0.1 * jax.random.normal(next(ks), (cfg.linear_value_head_dim,)),
                w_out=n((d, cfg.linear_value_dim)),
                n_k_heads=cfg.linear_num_key_heads,
                n_v_heads=cfg.linear_num_value_heads,
                k_head_dim=cfg.linear_key_head_dim,
                v_head_dim=cfg.linear_value_head_dim,
                eps=cfg.rms_norm_eps,
            ),
        )

    def attn_sublayer() -> AttnSublayer:
        return AttnSublayer(
            ln1=centered_norm((d,)),
            attn=GatedAttention(
                wq=n((2 * qd, d)),
                wk=n((kvd, d)),
                wv=n((kvd, d)),
                wo=n((d, qd)),
                q_norm=centered_norm((cfg.head_dim,)),
                k_norm=centered_norm((cfg.head_dim,)),
                n_head=cfg.n_head,
                n_kv_head=cfg.n_kv_head,
                head_dim=cfg.head_dim,
                eps=cfg.rms_norm_eps,
                implementation="xla",
            ),
        )

    def moe_layer() -> FrozenMoE:
        return FrozenMoE(
            ln=centered_norm((d,)),
            router=n((cfg.n_experts, d)),
            experts_gate=n((fused, d)),
            experts_up=n((fused, d)),
            experts_down=n((d, fused)),
            shared_gate=n((cfg.shared_expert_intermediate, d)),
            shared_up=n((cfg.shared_expert_intermediate, d)),
            shared_down=n((d, cfg.shared_expert_intermediate)),
            shared_expert_gate=n((1, d)),
        )

    return build_qwen36_moe_model(
        cfg,
        sites,
        embed=n((cfg.vocab_size, d), 0.05),
        deltanet_sublayers=[
            deltanet_sublayer()
            for layer in range(cfg.n_layer)
            if not layer_is_full_attention(cfg, layer)
        ],
        attn_sublayers=[
            attn_sublayer() for layer in range(cfg.n_layer) if layer_is_full_attention(cfg, layer)
        ],
        moe_layers=[moe_layer() for _ in range(cfg.n_layer)],
        norm=centered_norm((d,)),
        lm_head=n((cfg.vocab_size, d), 0.05),
        # toy dims sit below the split arm's 64-multiple kernel tiling: the oracle arm
        # serves every engine-semantics fixture (kernel parity lives in
        # tests/routed/test_experts at 64-multiple shapes)
        expert_implementation="ragged_dot",
        output_edge=MaterializedOutputEdge(),
    )


def _mask_width(cfg: Qwen36MoeConfig, spec: SiteSpec) -> int:
    """The per-position width of one site's mask/CI values: the `k·c` routed slots of an
    expert kind, the full `C` of a shared kind."""
    _layer, kind = qwen36_moe.parse_site_name(spec.name)
    if qwen36_moe.is_expert_kind(kind):
        return cfg.n_experts_per_token * spec.C // cfg.n_experts
    return spec.C


def site_masks(
    model: Qwen36MoeDecomposedModel, routing: BlockSelection, values: Mapping[str, jax.Array]
) -> dict[str, SiteCI]:
    """Per-site masks (or CI) in each kind's emission from bare values: an expert kind's
    `[*lead, k·c]` values bundle with the pinned routing of their layer (slot m is expert
    `indices[layer, .., m]`), a shared kind's `[*lead, C]` values pass through."""
    masks: dict[str, SiteCI] = {}
    for name, value in values.items():
        layer, kind = qwen36_moe.parse_site_name(name)
        if not qwen36_moe.is_expert_kind(kind):
            masks[name] = value
            continue
        masks[name] = SelectedCI(value, routing.indices[layer], model.cfg.n_experts)
    return masks


def constant_mask_values(
    model: Qwen36MoeDecomposedModel, lead: tuple[int, ...], fill: float
) -> dict[str, jax.Array]:
    """Every site's mask/CI values at its emission width, filled with `fill`."""
    return {
        spec.name: jnp.full((*lead, _mask_width(model.cfg, spec)), fill) for spec in model.sites
    }


def random_mask_values(
    model: Qwen36MoeDecomposedModel, lead: tuple[int, ...], key: jax.Array
) -> dict[str, jax.Array]:
    """Every site's mask/CI values at its emission width, uniform on [0, 1)."""
    return {
        spec.name: jax.random.uniform(
            jax.random.fold_in(key, i), (*lead, _mask_width(model.cfg, spec))
        )
        for i, spec in enumerate(model.sites)
    }


def identity_masking(
    model: Qwen36MoeDecomposedModel, conditioning: BlockSelection
) -> MaterializedMasking:
    """The exact identity on `conditioning`: every site's mask ≡ 1 in its kind's emission and
    every weight-delta mask ≡ 1, so every site computes `x @ W` again and the masked
    forward reproduces the clean one up to fp32 reassociation."""
    lead = conditioning.indices.shape[1:-1]
    return MaterializedMasking(
        component_masks=site_masks(model, conditioning, constant_mask_values(model, lead, 1.0)),
        weight_delta_masks={spec.name: jnp.ones(lead) for spec in model.sites},
    )


def tiny_qwen36_moe_ci_fn_arch(
    model: Qwen36MoeDecomposedModel,
) -> BlockSelectedChunkwiseTransformerCIFnArch:
    """One chunk per target stage over the model's OWN sites: expert kinds narrow
    (router = in-stage position), shared kinds full — the resolver's shape at the tiny
    config."""
    cfg = model.cfg
    interval = cfg.full_attention_interval
    chunk_slots: list[list[FullSlot | SelectedSlot]] = [[] for _ in range(cfg.n_stages)]
    for spec in model.sites:
        layer, kind = qwen36_moe.parse_site_name(spec.name)
        chunk_slots[layer // interval].append(
            SelectedSlot(site=spec.name, selection=layer % interval)
            if qwen36_moe.is_expert_kind(kind)
            else FullSlot(site=spec.name)
        )
    chunks = tuple(
        BlockSelectedChunk(
            input_taps=(resid_tap_key(start),),
            layers=tuple(range(start, start + interval)),
            slots=tuple(chunk_slots[start // interval]),
        )
        for start in range(0, cfg.n_layer, interval)
    )
    # Dims sized to tile the placed suites' (data=4, tp=2) mesh (÷8 on the sharded axes).
    return BlockSelectedChunkwiseTransformerCIFnArch(
        chunks=chunks,
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=2,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        table_size=cfg.n_experts,
        selected_ffn_hidden=8,
        shared_ffn_hidden=8,
        learned_norm_scale=False,
        # toy dims sit below the split arm's 64-multiple kernel tiling: the oracle arm
        expert_implementation="ragged_dot",
    )


def tiny_qwen36_moe_ci_fn(
    model: Qwen36MoeDecomposedModel, key: jax.Array
) -> BlockSelectedChunkwiseTransformerCIFn:
    ci_fn = tiny_qwen36_moe_ci_fn_arch(model).initialize(model.sites, None, key)
    assert isinstance(ci_fn, BlockSelectedChunkwiseTransformerCIFn)
    return ci_fn
