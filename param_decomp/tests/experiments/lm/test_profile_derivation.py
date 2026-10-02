from pathlib import Path

import pytest
import yaml

from param_decomp.experiments.lm.config import LMExperimentConfig, assert_placement_claims
from param_decomp.experiments.lm.profile_config import ProfileShape, derive_profile


@pytest.mark.parametrize("hidden_reconstruction", [False, True])
def test_profile_derivation_uses_the_explicit_mesh_surface(hidden_reconstruction: bool):
    base = Path(__file__).parents[3] / "experiments" / "lm" / "configs" / "llama3_1_8b.yaml"
    raw = yaml.safe_load(base.read_text())
    if hidden_reconstruction:
        recon = next(loss for loss in raw["pd"]["loss_metrics"] if "Recon" in loss["type"])
        recon["auxiliaries"] = [
            {
                "name": "hidden_acts_reconstruction",
                "coeff": 0.2,
                "comparisons": [
                    {"capture": f"resid.{i}", "distance": "relative_squared_error"}
                    for i in range(1, 33)
                ],
            }
        ]
    derived = derive_profile(
        raw,
        ProfileShape(
            layers=16,
            batch_size=64,
            replicate=8,
            fsdp=2,
            tp=4,
            steps=5,
            profile_steps=3,
            sharding="owner",
        ),
    )

    assert_placement_claims(LMExperimentConfig.model_validate(derived), Path("out"))

    assert derived["run_name"] == "profile-16l-b64-r8-f2-t4-semantic-owner"
    runtime = derived["runtime"]
    assert runtime["mesh"] == {"replicate": 8, "fsdp": 2, "tp": 4}
    assert "dp" not in runtime and "replicate" not in runtime
    # The profile window is the typed `runtime.profiling` arm — never launch_env plumbing.
    assert runtime["profiling"] == {"kind": "ad_hoc", "steps": 3}
    assert "PD_AD_HOC_PROFILE_STEPS" not in yaml.safe_dump(derived)
    # The derived config carries the base config's authored compiler token verbatim.
    assert runtime["compiler_options"] == raw["runtime"]["compiler_options"] == "tuned-v2"
    assert derived["decomposition"]["sites"]["layers"]["end"] == 16
    # Profiling preserves the source recipe's objective/optimizer, not a second recipe.
    if hidden_reconstruction:
        recon = next(loss for loss in derived["pd"]["loss_metrics"] if "Recon" in loss["type"])
        (auxiliary,) = recon["auxiliaries"]
        assert [comparison["capture"] for comparison in auxiliary["comparisons"]] == [
            f"resid.{i}" for i in range(1, 17)
        ]
    else:
        assert derived["pd"]["loss_metrics"] == raw["pd"]["loss_metrics"]
    assert derived["pd"]["components_optimizer"] == raw["pd"]["components_optimizer"]
    assert derived["pd"]["ci_fn_optimizer"] == raw["pd"]["ci_fn_optimizer"]
