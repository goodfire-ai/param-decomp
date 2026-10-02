"""Global-transformer tap resolution, the placement gate, and stored-run consumption."""

from pathlib import Path

import jax
import pytest
import yaml

from param_decomp.core.ci_fn.implementations.global_transformer.arch import (
    GlobalTransformerBackbone,
    GlobalTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.backbone import BackboneCIFn
from param_decomp.core.ci_fn.implementations.transformer.component_conditioning import (
    ConditionedCIFnArch,
    InputScaleCalibration,
    SiteInput,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.ci_fn.implementations.transformer.placement import PlacedTransformerCIFnRows
from param_decomp.core.ci_fn.interface import TapSpec
from param_decomp.core.components import SiteC
from param_decomp.core.configs import HsdpMeshShape
from param_decomp.experiments.lm.config import (
    CIFnInputTapSelection,
    LMDecompositionConfig,
    LMExperimentConfig,
    LMTargetConfig,
    ResidualBoundaryTapSelection,
    assert_placement_claims,
    resolve_decomposition,
    resolve_lm_ci_fn_arch,
)
from param_decomp.experiments.lm.load_run import ConsumerLayout
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer import glu_site_specs, site_name
from param_decomp.tests.experiments.lm.test_load_run import _open_checkpointed, assert_opened_on

QWEN3_1_7B = Path(__file__).parents[3] / "experiments/lm/configs/qwen3_1_7b.yaml"


@pytest.mark.parametrize(
    ("layers", "selection", "expected"),
    [
        ((18, 19, 20, 21), "first_block_resid", ("resid.18",)),
        ((18, 19, 20, 21), "all_block_resids", ("resid.18", "resid.19", "resid.20", "resid.21")),
        (
            (18, 19, 20, 21),
            ResidualBoundaryTapSelection(offsets=(0, 2, 4)),
            ("resid.18", "resid.20", "resid.22"),
        ),
        (
            (18, 20, 21),
            ResidualBoundaryTapSelection(offsets=(0, 1, 4)),
            ("resid.18", "resid.19", "resid.22"),
        ),
    ],
)
def test_global_transformer_resolves_taps_over_every_selected_block(
    tmp_path: Path,
    layers: tuple[int, ...],
    selection: CIFnInputTapSelection,
    expected: tuple[str, ...],
):
    target = LMTargetConfig.model_validate(
        {
            "spec": {
                "kind": "hf",
                "model_class": "transformers.LlamaForCausalLM",
                "model_name": "meta-llama/Llama-3.1-8B",
            },
            "attention_implementation": "xla",
            "weights_dtype": "bfloat16",
            "output_edge": {"kind": "materialized"},
        }
    )
    decomposition = LMDecompositionConfig.model_validate(
        {
            "sites": {
                "kind": "glu_transformer",
                "layers": {"kind": "list", "indices": layers},
                "cs": {"q": 4, "down": 8},
            },
            "ci": {
                "type": "global_transformer",
                "input_tap": selection,
                "d_model": 16,
                "n_blocks": 2,
                "attention": {
                    "kind": "mha",
                    "n_heads": 2,
                    "implementation": "xla",
                    "mask": "bidirectional",
                },
                "ffn": {"kind": "gelu", "hidden": 32},
            },
        }
    )
    resolved = resolve_decomposition(target, decomposition, tmp_path)
    arch = resolve_lm_ci_fn_arch(resolved, decomposition.ci)
    assert isinstance(arch, GlobalTransformerCIFnArch)
    assert arch.input_taps == tuple(TapSpec(key=key, width=4096) for key in expected)


def test_component_conditioning_reads_each_sites_clean_input(tmp_path: Path):
    target = LMTargetConfig.model_validate(
        {
            "spec": {
                "kind": "hf",
                "model_class": "transformers.LlamaForCausalLM",
                "model_name": "meta-llama/Llama-3.1-8B",
            },
            "attention_implementation": "xla",
            "weights_dtype": "bfloat16",
            "output_edge": {"kind": "materialized"},
        }
    )
    decomposition = LMDecompositionConfig.model_validate(
        {
            "sites": {
                "kind": "glu_transformer",
                "layers": {"kind": "list", "indices": (2, 3)},
                "cs": {"q": 4, "o": 4, "up": 4, "down": 8},
            },
            "ci": {
                "type": "global_transformer",
                "component_conditioning": {
                    "kind": "affine_clipped_component_activation",
                    "output_scale_init": 0.5,
                    "calibration": {"min_n_tokens": 4096},
                },
                "input_tap": "first_block_resid",
                "d_model": 16,
                "n_blocks": 2,
                "attention": {
                    "kind": "mha",
                    "n_heads": 2,
                    "implementation": "xla",
                    "mask": "causal",
                },
                "ffn": {"kind": "gelu", "hidden": 32},
            },
        }
    )
    arch = resolve_lm_ci_fn_arch(
        resolve_decomposition(target, decomposition, tmp_path), decomposition.ci
    )

    assert isinstance(arch, ConditionedCIFnArch)
    assert isinstance(arch.inner, GlobalTransformerCIFnArch)
    assert arch.output_scale_init == 0.5
    assert arch.calibration == InputScaleCalibration(min_n_tokens=4096)
    taps = (("q", "attn_in"), ("o", "attn_out"), ("up", "mlp_in"), ("down", "mlp_hidden"))
    assert arch.site_inputs == (
        tuple(
            SiteInput(site_name(layer, kind), f"{tap}.{layer}")
            for layer in (2, 3)
            for kind, tap in taps
        )
    )
    assert arch.capture_keys == {
        "resid.2",
        *(f"{tap}.{layer}" for layer in (2, 3) for _, tap in taps),
    }


def test_residual_boundaries_refuse_offsets_past_the_last_block(tmp_path: Path):
    raw = yaml.safe_load(QWEN3_1_7B.read_text())
    raw["decomposition"]["ci"]["input_tap"]["offsets"] = [0, 29]
    with pytest.raises(AssertionError, match=r"exceed the selected boundary range 0\.\.28"):
        assert_placement_claims(LMExperimentConfig.model_validate(raw), tmp_path)


def test_qwen3_1_7b_global_ci_places_on_its_authored_mesh(tmp_path: Path):
    """The maintained config's component-conditioned global CI passes the gate under
    its owner rows and refuses a preset the global transformer names no rows for."""
    raw = yaml.safe_load(QWEN3_1_7B.read_text())
    config = LMExperimentConfig.model_validate(raw)
    assert config.runtime.sharding == "owner"
    assert_placement_claims(config, tmp_path)

    raw["runtime"]["sharding"] = "ddp"
    with pytest.raises(NotImplementedError, match="no rows for placement preset 'ddp'"):
        assert_placement_claims(LMExperimentConfig.model_validate(raw), tmp_path)


def test_global_transformer_checkpoint_opens_on_a_consumer_owner_layout(tmp_path: Path):
    cfg = tiny_glu_cfg()
    replicas = jax.device_count()
    sites = glu_site_specs(
        cfg,
        tuple(
            SiteC(site_name(layer, kind), 8 * replicas)
            for layer in (2, 3)
            for kind in ("q", "down")
        ),
    )
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.key(0))
    arch = GlobalTransformerCIFnArch(
        input_taps=(TapSpec("resid.2", cfg.n_embd), TapSpec("resid.4", cfg.n_embd)),
        d_model=16 * replicas,
        n_blocks=1,
        attention=MHACIFnAttention(n_heads=2, implementation="xla", mask="bidirectional"),
        ffn_hidden=32 * replicas,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    layout = ConsumerLayout(mesh=HsdpMeshShape(replicate=1, fsdp=replicas, tp=1), sharding="owner")
    run = _open_checkpointed(tmp_path / "p-global-transformer", model, arch, layout)
    assert_opened_on(run, layout)
    prepared = run.prepared_ci_fn
    assert isinstance(prepared, BackboneCIFn)
    assert isinstance(prepared.backbone, GlobalTransformerBackbone)
    assert isinstance(prepared.backbone.placement, PlacedTransformerCIFnRows)
