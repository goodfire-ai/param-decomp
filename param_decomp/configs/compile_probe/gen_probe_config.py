"""Generate a small full-model-faithful compile-probe config.

Derived from llama8b_full32L_seq512_b128_dp128.yaml: SAME per-kind C structure
(q/k 2048, v/o 4096, gate/up 8192, down 10240) so the masked forward takes the
uniform-per-kind `lax.scan` path (the production compile path), SAME chunkwise-recon +
persistent-PGD + faith + imp-min machinery, SAME CI-fn arch (4-block transformer). The
only knobs scaled down are layer count (decompose the LAST `n_layers`, so the live
decomposed span the scan walks is exactly `n_layers` long), batch, seq, and the
warmup/step counts so we reach the real recon step fast for timing.

Usage: python gen_probe_config.py <n_layers> <dp> <out.yaml>
"""

import sys

import yaml

PER_KIND_C = {
    "q": 2048,
    "k": 2048,
    "v": 4096,
    "o": 4096,
    "gate": 8192,
    "up": 8192,
    "down": 10240,
}
N_TOTAL_LAYERS = 32


def main(n_layers: int, dp: int, out_path: str, seq: int = 256, batch: int | None = None) -> None:
    if batch is None:
        batch = dp  # per-rank 1
    first = N_TOTAL_LAYERS - n_layers
    n_sites = n_layers * len(PER_KIND_C)
    sites_per_chunk = n_sites  # one chunk: minimise compile, still the chunkwise path

    cfg = {
        "run_name": f"compileprobe-{n_layers}L-dp{dp}-seq{seq}",
        "cadence": {"keep_last_n_checkpoints": 1, "save_every": 100000, "train_log_every": 1},
        "data": {
            "buffer_size": 1000,
            "column_name": "input_ids",
            "data_files": "/mnt/data/artifacts/mechanisms/param-decomp/datasets/fineweb_llama_tok_512/*.parquet",
            "dataset_name": "parquet",
            "eval_split": "train",
            "is_tokenized": True,
            "max_seq_len": seq,
            "revision": None,
            "shuffle_each_epoch": True,
            "streaming": False,
            "tokenizer_name": "meta-llama/Llama-3.1-8B",
            "train_split": "train",
        },
        # eval disabled-ish: large `every` so no eval compile interferes with timing
        "eval": {
            "batch_size": batch,
            "every": 100000,
            "metrics": [
                {"ci_alive_threshold": 0.0, "groups": None, "type": "CI_L0"},
                {"rounding_threshold": 0.0, "type": "CEandKLLosses"},
            ],
            "n_steps": 1,
            "slow_every": 1000000,
            "slow_on_first_step": False,
        },
        "decomposition": {
            "sites": {
                "kind": "glu_transformer",
                "layers": {"kind": "range", "start": first, "end": N_TOTAL_LAYERS},
                "cs": dict(PER_KIND_C),
            },
            # one chunk spanning every decomposed block: minimise compile count while
            # still exercising the chunkwise path (matches the old sites_per_chunk=all)
            "ci": {
                "type": "chunkwise_transformer",
                "blocks_per_chunk": n_layers,
                "d_model": 4096,
                "n_blocks": 4,
                "n_heads": 64,
                "mlp_hidden": 16384,
            },
        },
        "pd": {
            "batch_size": batch,
            "ci_fn_optimizer": {
                "betas": [0.9, 0.999],
                "grad_clip_norm": None,
                "lr_schedule": {
                    "final_val_frac": 0.1,
                    "fn_type": "cosine",
                    "start_val": 2.0e-05,
                    "warmup_pct": 0.0,
                },
                "weight_decay": 0.0,
            },
            "components_optimizer": {
                "betas": [0.9, 0.999],
                "grad_clip_norm": 0.01,
                "lr_schedule": {
                    "final_val_frac": 0.1,
                    "fn_type": "cosine",
                    "start_val": 2.0e-05,
                    "warmup_pct": 0.0,
                },
                "weight_decay": 0.0,
            },
            "faithfulness_warmup_lr": 0.001,
            "faithfulness_warmup_steps": 2,
            "faithfulness_warmup_weight_decay": 0.0,
            "identity_decomposition_targets": None,
            "loss_metrics": [
                {
                    "coeff": 5.0e-06,
                    "eps": 1.0e-06,
                    "pnorm": {"start_val": 2.0, "fn_type": "linear", "final_val_frac": 0.2},
                    "type": "ImportanceMinimalityLoss",
                },
                {
                    "coeff": 2.0,
                    "n_samples": 1,
                    "routing": {"type": "uniform_k_subset"},
                    "sites_per_chunk": sites_per_chunk,
                    "type": "ChunkwiseSubsetReconLoss",
                },
                {
                    "coeff": 0.5,
                    "n_warmup_steps": 2,
                    "optimizer": {
                        "beta1": 0.01,
                        "beta2": 0.99,
                        "eps": 1.0e-08,
                        "lr_schedule": {
                            "final_val_frac": 1.0,
                            "fn_type": "constant",
                            "start_val": 0.01,
                            "warmup_pct": 0.025,
                        },
                        "type": "adam",
                    },
                    "scope": {"type": "bsc"},
                    "type": "PersistentPGDReconLoss",
                },
                {"coeff": 1000000.0, "type": "FaithfulnessLoss"},
            ],
            "seed": 0,
            "steps": 10,
        },
        "runtime": {
            "autocast_bf16": True,
            "device": "cuda:0",
            "dp": dp,
            "remat_recon_forwards": True,
        },
        "target": {
            "weights_dtype": "bfloat16",
            "spec": {
                "kind": "hf",
                "model_class": "transformers.LlamaForCausalLM",
                "model_name": "meta-llama/Llama-3.1-8B",
            },
        },
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(
        f"wrote {out_path}: {n_layers}L (layers {first}..31), {n_sites} sites, dp={dp}, batch={batch}, seq={seq}"
    )


if __name__ == "__main__":
    n_layers = int(sys.argv[1])
    dp = int(sys.argv[2])
    out_path = sys.argv[3]
    seq = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    main(n_layers, dp, out_path, seq=seq)
