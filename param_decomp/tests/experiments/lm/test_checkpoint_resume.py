"""A fresh trainer reloads cached target weights and continues the checkpoint trajectory."""

import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Literal

import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest
import yaml
from safetensors.numpy import save_file

from param_decomp.experiments.lm.load_run import ConsumerLayout, load_target, open_run
from param_decomp.tests.experiments.lm.test_qwen36_wiring import (
    _write_pretrain_cache as _write_qwen_config,
)
from param_decomp.tests.experiments.lm.test_run_inline import (
    _write_pretrain_cache,
    _write_run_config,
    _write_token_shards,
)
from param_decomp.tests.targets.qwen36_moe_hf_parity.test_qwen36_moe_hf_parity import (
    HERE,
    _tiny_cfg_from_golden,
)


@pytest.mark.parametrize(
    ("tp", "family"),
    [
        (1, "llama"),
        pytest.param(2, "llama", marks=pytest.mark.multidevice),
        pytest.param(2, "qwen", marks=pytest.mark.multidevice),
    ],
)
def test_cached_target_checkpoint_resumes_in_fresh_process(
    tmp_path: Path, tp: int, family: Literal["llama", "qwen"]
) -> None:
    if jax.default_backend() != "gpu" or jax.device_count() < tp:
        pytest.skip(f"requires {tp} GPUs")
    data_root = tmp_path / "data"
    _write_token_shards(tmp_path / "shards")
    _write_token_shards(tmp_path / "eval_shards")
    config_path = tmp_path / "config.yaml"
    _write_run_config(config_path, tmp_path / "shards", dp=tp, tp=tp, weights_dtype="bfloat16")
    config = yaml.safe_load(config_path.read_text())
    match family:
        case "llama":
            # The read-only consumer spans all eight devices on the multi-device CI host.
            config["decomposition"]["sites"]["cs"] = {"c_fc": 8, "down_proj": 8}
            _write_pretrain_cache(data_root)
        case "qwen":
            with np.load(HERE / "qwen36_moe_tiny_hf_fixtures.npz") as golden:
                arch = replace(
                    _tiny_cfg_from_golden(str(golden["config_json"])),
                    n_layer=2,
                    full_attention_interval=2,
                )
                cache = data_root / "pretrain_cache" / "spd-t-00000000"
                _write_qwen_config(cache, arch)
                weights = {
                    name.removeprefix("sd::").replace("layers.3.", "layers.1."): golden[name]
                    for name in golden.files
                    if name.startswith("sd::")
                    and ("layers." not in name or "layers.0." in name or "layers.3." in name)
                }
                save_file(weights, cache / "model_step_100.safetensors")
            config["target"]["spec"] = {
                "kind": "pretrained_qwen35_moe",
                "run_path": "goodfire/spd/runs/t-00000000",
            }
            config["target"]["expert_implementation"] = "dense_masked"
            config["decomposition"]["sites"] = {
                "kind": "qwen36_moe",
                "layers": {"kind": "all"},
                "cs": {
                    # Four experts each need four components for the consumer data axis.
                    "experts_gate": 16,
                    "experts_up": 16,
                    "experts_down": 16,
                    "shared_gate": 8,
                    "gdn_v": 8,
                },
            }
            config["decomposition"]["ci"] = {
                "type": "moe_chunkwise_transformer",
                "blocks_per_chunk": 2,
                "d_model": 8,
                "n_blocks": 1,
                "attention": {
                    "kind": "mha",
                    "mask": "bidirectional",
                    "implementation": "xla",
                    "n_heads": 2,
                },
                "expert_ffn_hidden": 8,
                "shared_ffn_hidden": 8,
                "expert_implementation": "dense_masked",
            }
            config["runtime"]["mesh"] = {"data": 1, "tp": tp}
            config["runtime"]["sharding"] = "zero1-replicated-resident-moe"
    config["target"]["output_edge"] = {"kind": "streamed", "n_vocab_chunks": 4}
    config["cadence"]["checkpointing"] = {
        "kind": "periodic",
        "save_every": 1,
        "retention": {"kind": "keep_last", "n": 2},
    }
    config["pd"]["faithfulness_warmup_steps"] = 0
    config["pd"]["loss_metrics"][1]["frequency"] = {
        "coeff": 1e-3,
        "reference_datapoint_count": 64,
        "ema_halflife_steps": 2,
    }
    config["pd"]["loss_metrics"][2] = {
        "type": "MergedStochasticSubsetPooledPPGDReconLoss",
        "coeff": 1.0,
        "pool": {"size_per_batch_element": 4},
        "adv_fraction": 0.5,
        "n_warmup_steps": 1,
        "optimizer": {"type": "adam", "lr_schedule": 0.02},
    }
    config_path.write_text(yaml.safe_dump(config))
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["JAX_PLATFORMS"] = "cuda"
    visible = env.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(jax.device_count()))))
    env["CUDA_VISIBLE_DEVICES"] = ",".join(visible.split(",")[:tp])
    command = [
        sys.executable,
        "-m",
        "param_decomp.experiments.lm.run",
        str(config_path),
        "--data-root",
        str(data_root),
        "--local-device-count",
        str(tp),
        "--run-id",
        "p-0000abcd",
    ]
    first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    (tmp_path / "initial.stdout.log").write_text(first.stdout)
    (tmp_path / "initial.stderr.log").write_text(first.stderr)
    assert first.returncode == 0, first.stdout + first.stderr
    run_dir = data_root / "runs" / "p-0000abcd"
    checkpoint = run_dir / "ckpts" / "2"
    uninterrupted = tmp_path / "uninterrupted"
    shutil.copytree(checkpoint, uninterrupted)
    shutil.rmtree(checkpoint)

    resumed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    (tmp_path / "resumed.stdout.log").write_text(resumed.stdout)
    (tmp_path / "resumed.stderr.log").write_text(resumed.stderr)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "resumed from checkpoint step 1" in resumed.stdout
    assert "checkpoint saved @ step 2" in resumed.stdout
    with ocp.StandardCheckpointer() as reader:
        for item in ("decomposition", "training"):
            expected = reader.restore(uninterrupted / item)
            actual = reader.restore(checkpoint / item)
            if item == "training":
                assert int(actual["step"]) == 2
                estimate = actual["frequency"]["estimate"]
                assert estimate
                assert all(np.any(np.asarray(frequency) > 0) for frequency in estimate.values())
                (adversary,) = actual["adversaries"].values()
                assert float(adversary["opt_state"]["step_count"]) == 4
                assert any(
                    np.any(np.asarray(moment) != 0)
                    for moment in jax.tree.leaves(adversary["opt_state"]["m"])
                )
            assert jax.tree.structure(actual) == jax.tree.structure(expected)
            for saved, restored in zip(
                jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
            ):
                np.testing.assert_array_equal(restored, saved)
                assert restored.sharding == saved.sharding

    loaded = open_run(
        run_dir,
        None,
        data_root=data_root,
        layout=ConsumerLayout.model_validate(
            {
                "mesh": {"data": jax.device_count() // tp, "tp": tp},
                "sharding": "zero1-replicated-resident-moe"
                if family == "qwen"
                else "zero1-replicated-resident",
            }
        ),
    )
    assert loaded.step == 2
    assert all(
        leaf.devices() <= set(loaded.mesh.devices.flat) for leaf in jax.tree.leaves(loaded.model)
    )
    assert all(
        np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(loaded.prepared_weights)
    )

    cpu_target = load_target(loaded.deliverable.target, data_root)
    for source, placed in zip(
        jax.tree.leaves(cpu_target), jax.tree.leaves(loaded.model), strict=True
    ):
        assert all(device.platform == "cpu" for device in source.devices())
        np.testing.assert_array_equal(np.asarray(source), np.asarray(placed))
