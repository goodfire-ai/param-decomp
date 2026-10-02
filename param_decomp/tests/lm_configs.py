"""Test-specific variants derive from the single maintained Llama recipe."""

from pathlib import Path
from typing import Any

import yaml

CONFIGS = Path(__file__).parents[1] / "experiments/lm/configs"


def llama_raw() -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / "llama3_1_8b.yaml").read_text())


def l18_mlp_raw() -> dict[str, Any]:
    """Small-site build fixture exercising separate stochastic/persistent objectives."""
    raw = llama_raw()
    raw["run_name"] = "test-l18-mlp"
    raw["target"]["output_edge"] = {"kind": "materialized"}
    raw["runtime"].update(mesh={"replicate": 1, "fsdp": 1, "tp": 1}, sharding="owner")
    raw["pd"]["batch_size"] = 128
    raw["decomposition"] = {
        "sites": {
            "kind": "glu_transformer",
            "layers": {"kind": "list", "indices": [18]},
            "cs": {"gate": 24576, "up": 24576, "down": 24576},
        },
        "ci": {
            "type": "chunkwise_transformer",
            "blocks_per_chunk": 1,
            "d_model": 4096,
            "n_blocks": 4,
            "attention": {
                "kind": "mha",
                "implementation": "flash",
                "mask": "bidirectional",
                "n_heads": 64,
            },
            "ffn": {"kind": "gelu", "hidden": 16384},
        },
    }
    raw["pd"]["loss_metrics"] = [
        {"type": "FaithfulnessLoss", "coeff": 1000.0},
        {
            "type": "ImportanceMinimalityLoss",
            "coeff": 5e-06,
            "gamma": {
                "max_val": 1.0,
                "points": [{"at": 0.0, "frac": 1.0}, {"at": 1.0, "frac": 0.01}],
            },
            "frequency": {"coeff": 1e-06, "reference_datapoint_count": 262144},
        },
        {
            "type": "StochasticReconSubsetLoss",
            "coeff": 0.5,
            "routing": {"type": "uniform_k_subset"},
        },
        {
            "type": "PersistentPGDReconLoss",
            "coeff": 0.5,
            "n_warmup_steps": 2,
            "optimizer": {
                "type": "adam",
                "beta1": 0.5,
                "beta2": 0.99,
                "eps": 1e-08,
                "lr_schedule": {
                    "max_val": 0.01,
                    "points": [
                        {"at": 0.0, "frac": 0.0},
                        {"at": 0.025, "frac": 1.0},
                        {"at": 1.0, "frac": 1.0},
                    ],
                },
            },
            "source_shape": "bsc",
        },
    ]
    return raw


def arithmetic_metric() -> dict[str, Any]:
    return {
        "type": "ArithmeticCIGrid",
        "probe_metrics": {
            "ce_kl": {"rounding_threshold": 0.0},
            "ci_l0": {"ci_alive_threshold": 0.0, "groups": None},
            "fresh_pgd": {"n_steps": 20, "step_size": 0.1},
        },
    }
