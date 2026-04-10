# Sweep Summary Stats

Generates a markdown/LaTeX summary report from a set of WandB SPD runs (typically seed reruns of the same config). Reports mean and std across seeds for all key metrics.

## Usage

```bash
python spd/scripts/sweep_summary/sweep_summary_stats.py <run_id_1> <run_id_2> ...
python spd/scripts/sweep_summary/sweep_summary_stats.py <run_ids> --name my_sweep
python spd/scripts/sweep_summary/sweep_summary_stats.py <run_ids> --project goodfire/spd
python spd/scripts/sweep_summary/sweep_summary_stats.py <run_ids> --harvest-run <run_id>
python spd/scripts/sweep_summary/sweep_summary_stats.py <run_ids> --stdout
```

Results (`report.md`, `target_val_loss_curve.png`) are saved to `spd/scripts/sweep_summary/out/<name>/`. Defaults to the first run ID if `--name` is not provided. Use `--stdout` to print the report instead of saving.

## Metrics

- **Output Quality (CE/KL)**: KL divergence and CE difference between SPD and target model under each masking mode (unmasked, stochastic, CI, rounded, random, zero).
- **Eval Reconstruction Losses**: Aggregate eval losses (stochastic recon, PGD recon, hidden acts recon, faithfulness, importance minimality).
- **Hidden Acts Recon (per module)**: Reconstruction losses broken down by weight matrix, with cross-layer summaries.
- **Sparsity (CI-L0)**: Component mask sparsity, per-layer and per-module.
- **Training Losses**: Final-step training losses.
- **Training Compute Recovered**: Finds where the SPD model's CE loss falls on the target model's training loss curve and reports the percentage of training compute that point represents. E.g., 95% means the decomposition preserves performance equivalent to 95% through training.

## Training Compute Recovered

This is the headline metric. The script fetches the target model's validation loss curve from WandB, smooths it with a forward-only EMA, then interpolates to find the training step at which the target model first reached the SPD model's CE loss. The result is `step / total_steps * 100`. Reported per masking mode (unmasked, stochastic, CI, rounded).
