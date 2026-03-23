from pathlib import Path

from spd.clustering.scripts.backfill_merge_wandb import parse_iteration_metrics


def test_parse_iteration_metrics_recovers_old_scalar_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "slurm-123.out"
    log_path.write_text(
        "\r".join(
            [
                "Loaded: 10007 components, 10000000 samples",
                "k=10006, mdl=0.1234, pair=0.0050:   0%|          | 1/20000 [00:05<27:46:40,  5.00s/iter]",
                "Compressed merge progress: iter=1/20000, elapsed=5.00s, sec_per_iter=5.0000, k_groups=10005",
                "k=10005, mdl=0.2345, pair=0.0070:   0%|          | 2/20000 [00:09<25:00:00,  4.50s/iter]",
                "Compressed merge progress: iter=2/20000, elapsed=9.00s, sec_per_iter=4.5000, k_groups=10004",
            ]
        )
    )

    metrics, meta = parse_iteration_metrics(log_path, n_samples=10_000_000)

    assert meta == {"n_components": 10007, "n_samples": 10000000}
    assert len(metrics) == 2

    first = metrics[0]
    assert first.step == 0
    assert first.total_iters == 20000
    assert first.k_groups == 10006
    assert first.merge_pair_cost == 0.005
    assert first.mdl_loss_norm == 0.1234
    assert first.mdl_loss == 1_234_000.0
    assert first.elapsed_s == 5.0
    assert first.sec_per_iter_avg == 5.0

    second = metrics[1]
    assert second.step == 1
    assert second.k_groups == 10005
    assert second.merge_pair_cost == 0.007
    assert second.mdl_loss_norm == 0.2345
    assert second.mdl_loss == 2_345_000.0
