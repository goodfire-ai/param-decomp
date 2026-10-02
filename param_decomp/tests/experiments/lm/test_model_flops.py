"""Config-to-MFU accounting without target weights, batches, or compilation."""

import json
from pathlib import Path

import pytest
import yaml

from param_decomp.experiments.lm.config import (
    LMExperimentConfig,
    LMTargetedExperimentConfig,
    PretrainedTarget,
    resolve_decomposition,
    resolve_lm_ci_fn_arch,
)
from param_decomp.experiments.lm.model_flops import (
    decomposition_compute,
    ordinary,
    performance_report,
    prepare_lm_step_flops,
    prepare_targeted_lm_step_flops,
    targeted,
    targeted_decomposition_compute,
)
from param_decomp.infra.pretrain_cache import cache_dir_for_run
from param_decomp.tests.experiments.lm.test_lm_targeted import _targeted_raw
from param_decomp.tests.lm_configs import l18_mlp_raw

CONFIGS = Path(__file__).parents[3] / "experiments/lm/configs"
HF_CONFIGS = [
    path
    for path in sorted(CONFIGS.glob("*.yaml"))
    if yaml.safe_load(path.read_text())["target"]["spec"]["kind"] == "hf"
]


@pytest.mark.parametrize("path", HF_CONFIGS, ids=lambda path: path.stem)
def test_shipped_hf_configs_have_complete_flop_breakdowns(path: Path, tmp_path: Path) -> None:
    raw = yaml.safe_load(path.read_text())
    if "nontarget" in raw:
        targeted = LMTargetedExperimentConfig.model_validate(raw)
        flops = targeted_decomposition_compute(targeted, 6, 512, tmp_path, step=0).model
        assert all(term.name.startswith(("target/", "nontarget/")) for term in flops.terms)
    else:
        ordinary = LMExperimentConfig.model_validate(raw)
        flops = decomposition_compute(ordinary, 512, tmp_path, step=0).model
        assert "faithfulness" in {term.name for term in flops.terms}
    assert flops.total == sum(term.total for term in flops.terms)
    assert flops.forward > 0
    assert flops.backward > 0
    assert not list(tmp_path.iterdir())


def test_global_batch_scales_token_work_but_not_weight_space_work(tmp_path: Path) -> None:
    config = LMExperimentConfig.model_validate(l18_mlp_raw())
    raw = config.model_dump(mode="json")
    raw["pd"]["batch_size"] *= 2
    doubled = LMExperimentConfig.model_validate(raw)
    before = decomposition_compute(config, 128, tmp_path, step=0).model
    after = decomposition_compute(doubled, 128, tmp_path, step=0).model
    for first, second in zip(before.terms, after.terms, strict=True):
        multiplier = 1 if first.name in {"faithfulness", "nonlinearity"} else 2
        assert second.total == multiplier * first.total


def test_rematerialization_does_not_change_model_flops(tmp_path: Path) -> None:
    config = LMExperimentConfig.model_validate(l18_mlp_raw())
    raw = config.model_dump(mode="json")
    raw["runtime"]["remat_ci_fn"] = not config.runtime.remat_ci_fn
    raw["runtime"]["remat_recon_forwards"] = not config.runtime.remat_recon_forwards
    changed = LMExperimentConfig.model_validate(raw)
    assert decomposition_compute(config, 128, tmp_path, step=0) == decomposition_compute(
        changed, 128, tmp_path, step=0
    )


def test_pretrained_simple_mlp_needs_only_architecture_metadata(tmp_path: Path) -> None:
    config = LMExperimentConfig.from_file(CONFIGS / "pile_llama_simple_mlp-4L.yaml")
    assert isinstance(config.target.spec, PretrainedTarget)
    run_path = config.target.spec.run_path
    cache = cache_dir_for_run(tmp_path, run_path)
    cache.mkdir(parents=True)
    (cache / "model_config.yaml").write_text(
        yaml.safe_dump(
            {
                "model_type": "LlamaSimpleMLP",
                "use_grouped_query_attention": True,
                "attn_bias": False,
                "mlp_bias": False,
                "rotary_adjacent_pairs": False,
                "vocab_size": 128,
                "n_layer": 4,
                "n_head": 4,
                "n_key_value_heads": 2,
                "n_embd": 64,
                "n_intermediate": 128,
                "rotary_base": 10000.0,
                "rms_norm_eps": 1e-5,
                "n_ctx": 512,
                "rotary_dim": 16,
                "block_size": 512,
            }
        )
    )
    flops = decomposition_compute(config, 64, tmp_path, step=0)
    assert flops.total > 0
    assert {file.name for file in cache.iterdir()} == {"model_config.yaml"}


def test_cli_uses_global_mesh_and_reports_a_fraction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = CONFIGS / "qwen3_6_35b_a3b.yaml"
    ordinary(str(path), str(tmp_path), 512, "NVIDIA B200", 2.0, step=0)
    report = json.loads(capsys.readouterr().out)
    assert report["n_devices"] == 64
    assert report["mfu_without_optimizer"] == pytest.approx(
        report["model_flops_per_step"] / (64 * 2.25e15 * 2)
    )
    assert report["ideal_step_time_without_optimizer_s"] == pytest.approx(
        report["mfu_without_optimizer"] * 2
    )
    assert report["mfu_with_optimizer"] == pytest.approx(
        report["total_flops_per_step"] / (64 * 2.25e15 * 2)
    )
    assert report["mfu_with_optimizer"] > report["mfu_without_optimizer"]
    assert report["optimizer_flops_per_step"] == sum(
        term["total_flops"] for term in report["optimizer_terms"]
    )


def test_cli_refuses_cpu_before_reading_config(tmp_path: Path) -> None:
    config = str(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="CPU has no declared BF16 peak"):
        ordinary(config, str(tmp_path), 128, "cpu", 1.0, step=0)
    with pytest.raises(ValueError, match="CPU has no declared BF16 peak"):
        targeted(config, str(tmp_path), 32, 128, "cpu", 1.0, step=0)


def test_report_preserves_each_term(tmp_path: Path) -> None:
    config = LMExperimentConfig.model_validate(l18_mlp_raw())
    flops = decomposition_compute(config, 128, tmp_path, step=0)
    report = performance_report(flops, "NVIDIA B200", 1, 1.0)
    assert report["forward_flops_per_step"] == flops.model.forward
    assert report["backward_flops_per_step"] == flops.model.backward


@pytest.mark.parametrize("length", [0, -1])
def test_invalid_sequence_length_is_refused(tmp_path: Path, length: int) -> None:
    config = LMExperimentConfig.model_validate(l18_mlp_raw())
    with pytest.raises(ValueError, match="positive"):
        decomposition_compute(config, length, tmp_path, step=0)


def test_distinct_ppgd_objective_adds_only_unshared_backward(tmp_path: Path) -> None:
    config = LMExperimentConfig.model_validate(l18_mlp_raw())
    raw = config.model_dump(mode="json")
    persistent = next(
        loss for loss in raw["pd"]["loss_metrics"] if loss["type"] == "PersistentPGDReconLoss"
    )
    persistent["adversary_objective"] = "e2e"
    persistent["auxiliaries"] = [
        {
            "name": "hidden",
            "coeff": 1.0,
            "comparisons": [{"capture": "resid.19", "distance": "relative_squared_error"}],
        }
    ]
    before = decomposition_compute(config, 128, tmp_path, step=0)
    after = decomposition_compute(LMExperimentConfig.model_validate(raw), 128, tmp_path, step=0)
    assert after.model.forward == before.model.forward
    assert after.optimizer == before.optimizer
    terms = {term.name: term for term in after.model.terms}
    retake = terms["reconstruction/PersistentPGDReconLoss/source_gradient"]
    ascent = terms["reconstruction/PersistentPGDReconLoss/ascent"]
    assert retake.flops.forward == 0
    assert 0 < retake.flops.backward < ascent.flops.backward
    assert after.model.backward - before.model.backward == retake.flops.backward


def test_optimizer_choice_changes_only_extended_mfu(tmp_path: Path) -> None:
    config = LMExperimentConfig.model_validate(l18_mlp_raw())
    raw = config.model_dump(mode="json")
    for name in ("components_optimizer", "ci_fn_optimizer"):
        raw["pd"][name] = {"type": "muon", "lr_schedule": 0.001, "ns_steps": 3}
    three_steps = decomposition_compute(
        LMExperimentConfig.model_validate(raw), 128, tmp_path, step=0
    )
    for name in ("components_optimizer", "ci_fn_optimizer"):
        raw["pd"][name]["ns_steps"] = 5
    five_steps = decomposition_compute(
        LMExperimentConfig.model_validate(raw), 128, tmp_path, step=0
    )
    assert three_steps.model == five_steps.model
    assert five_steps.optimizer.contraction_flops * 3 == three_steps.optimizer.contraction_flops * 5
    three_report = performance_report(three_steps, "NVIDIA B200", 8, 1.0)
    five_report = performance_report(five_steps, "NVIDIA B200", 8, 1.0)
    assert three_report["mfu_without_optimizer"] == five_report["mfu_without_optimizer"]
    assert three_report["mfu_with_optimizer"] != five_report["mfu_with_optimizer"]


def test_merged_final_backward_counts_only_adversarial_examples(tmp_path: Path) -> None:
    config = LMExperimentConfig.model_validate(l18_mlp_raw())
    raw = config.model_dump(mode="json")
    persistent = next(
        loss for loss in raw["pd"]["loss_metrics"] if loss["type"] == "PersistentPGDReconLoss"
    )
    persistent.update(
        {
            "type": "MergedStochasticSubsetPPGDReconLoss",
            "adv_fraction": 0.5,
            "adversary_objective": "e2e",
            "auxiliaries": [
                {
                    "name": "hidden",
                    "coeff": 1.0,
                    "comparisons": [{"capture": "resid.19", "distance": "relative_squared_error"}],
                }
            ],
        }
    )
    half = decomposition_compute(LMExperimentConfig.model_validate(raw), 128, tmp_path, step=1)
    persistent["adv_fraction"] = 1.0
    whole = decomposition_compute(LMExperimentConfig.model_validate(raw), 128, tmp_path, step=1)
    half_retake = next(term for term in half.model.terms if term.name.endswith("/source_gradient"))
    whole_retake = next(
        term for term in whole.model.terms if term.name.endswith("/source_gradient")
    )
    assert half_retake.n_repetitions == 0.5
    assert whole_retake.total == 2 * half_retake.total
    assert whole.model.forward == half.model.forward
    assert whole.model.backward - half.model.backward == half_retake.total


def test_lm_schedule_matches_saved_config_calculator(tmp_path: Path) -> None:
    raw = l18_mlp_raw()
    raw["pd"]["steps"] = 5
    config = LMExperimentConfig.model_validate(raw)
    resolved = resolve_decomposition(config.target, config.decomposition, tmp_path)
    ci = resolve_lm_ci_fn_arch(resolved, config.decomposition.ci)
    compute = prepare_lm_step_flops(
        config.pd, resolved.target, resolved.site_specs, ci, 16, tmp_path
    )
    for step in (0, 2, 4):
        standalone = decomposition_compute(config, 16, tmp_path, step)
        assert compute(step).model == pytest.approx(standalone.model.total)
        assert compute(step).optimizer == pytest.approx(standalone.optimizer.total)


def test_targeted_lm_schedule_preserves_independent_sequence_lengths(tmp_path: Path) -> None:
    raw = _targeted_raw()
    raw["pd"]["steps"] = 3
    config = LMTargetedExperimentConfig.model_validate(raw)
    resolved = resolve_decomposition(config.target, config.decomposition, tmp_path)
    ci = resolve_lm_ci_fn_arch(resolved, config.decomposition.ci)
    compute = prepare_targeted_lm_step_flops(
        config.pd, config.nontarget, resolved.target, resolved.site_specs, ci, 6, 32, tmp_path
    )
    for step in (0, 2):
        standalone = targeted_decomposition_compute(config, 6, 32, tmp_path, step)
        assert compute(step).model == pytest.approx(standalone.model.total)
        assert compute(step).optimizer == pytest.approx(standalone.optimizer.total)


@pytest.mark.parametrize("ffn_kind", ["gelu", "swiglu"])
@pytest.mark.parametrize("mask", ["causal", "bidirectional"])
def test_global_transformer_counts_as_one_chunk_over_every_block(
    tmp_path: Path, ffn_kind: str, mask: str
) -> None:
    raw = l18_mlp_raw()
    raw["decomposition"]["sites"]["layers"] = {"kind": "range", "start": 18, "end": 22}
    ci = raw["decomposition"]["ci"]
    ci.update(
        blocks_per_chunk=4,
        input_tap={"kind": "residual_boundaries", "offsets": [0, 2, 4]},
        d_model=16,
        n_blocks=2,
        attention={
            "kind": "gqa",
            "n_heads": 4,
            "n_kv_heads": 2,
            "implementation": "xla",
            "mask": mask,
        },
        ffn={"kind": ffn_kind, "hidden": 32},
        learned_norm_scale=True,
    )
    single_chunk = LMExperimentConfig.model_validate(raw)
    ci["type"] = "global_transformer"
    del ci["blocks_per_chunk"]
    global_transformer = LMExperimentConfig.model_validate(raw)

    expected = decomposition_compute(single_chunk, 16, tmp_path, step=0).model
    assert decomposition_compute(global_transformer, 16, tmp_path, step=0).model == expected
