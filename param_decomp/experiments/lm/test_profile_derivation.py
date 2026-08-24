from pathlib import Path

import yaml

from param_decomp.experiments.lm.profile_config import ProfileShape, derive_profile


def test_profile_derivation_uses_the_explicit_mesh_surface():
    base = Path(__file__).parent / "configs" / "profile_llama8b_full32_adam.yaml"
    raw = yaml.safe_load(base.read_text())
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
            sharding="zero1",
        ),
    )

    assert derived["run_name"] == "profile-h100-16l-b64-r8-f2-t4-semantic-zero1-adam"
    runtime = derived["runtime"]
    assert (runtime["replicate"], runtime["fsdp"], runtime["tp"]) == (8, 2, 4)
    assert "dp" not in runtime
    # The profile window is the typed `runtime.profiling` arm — never launch_env plumbing.
    assert runtime["profiling"] == {"kind": "ad_hoc", "steps": 3}
    assert "PD_AD_HOC_PROFILE_STEPS" not in yaml.safe_dump(derived)
    # The derived config carries the base seat's authored compiler token verbatim.
    assert runtime["compiler_options"] == raw["runtime"]["compiler_options"] == "tuned-v1"
    assert derived["decomposition"]["sites"]["layers"]["end"] == 16
    reconstruction = next(
        loss["hidden_acts_reconstruction"]
        for loss in derived["pd"]["loss_metrics"]
        if "hidden_acts_reconstruction" in loss
    )
    assert reconstruction["points"] == [f"resid.{layer}" for layer in range(1, 17)]
