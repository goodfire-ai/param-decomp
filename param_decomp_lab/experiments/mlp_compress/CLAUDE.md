# `param_decomp_lab/experiments/mlp_compress/`

Post-hoc experiments that re-express the MLPs (and attention) of a *saved* LM
decomposition as smaller / disentangled replacements, keeping the decomposition's frozen
component readers/writers and CI masks fixed. These are **not** `Trainer`-based experiments
(no YAML / `ExperimentConfig`); each `run_*.py` is a self-contained `fire` script that loads
a `SavedLMRun`, hooks the target model's submodules, and trains only the replacement params.

All scripts target the saved run at `RUN_DIR` (`runs/p-55ea3f9b`, the current-schema port of
s-55ea3f9b), defined in `run.py`.

## Files

| File | What it does |
|---|---|
| `run.py` | Compress **block-0** MLP to a smaller neuron dim (KL distill on CI-masked forwards). Home of `CompressedMaskedMLP`, `compute_ci_and_teacher`, `RUN_DIR`. |
| `run_all.py` | Same bottleneck applied to **all** block MLPs jointly. |
| `run_attn.py` | Replace all blocks' attention (fewer heads) **and** MLPs (bottleneck) jointly. Home of `CompressedMaskedAttention`, `mlp_module_names`, `attn_module_names`. |
| `run_mse.py` | All-block attn+MLP replacement trained on residual-stream **relative MSE** (+ optional Lp sparsity), not logit KL. Home of `capture_residuals`, `relative_mse`, `resid_point_names`. |
| `run_component.py` | Disentangled per-block **`ComponentMLP`** (dense bypass + sparsely-wired GeLU neurons) trained on *stochastic*-masked output-KL. The actively-used experiment; see below. |

`run.py` is the dependency root — every other script imports `CompressedMaskedMLP` /
`RUN_DIR` / `compute_ci_and_teacher` from it; `run_mse` / `run_component` also import from
`run_attn`.

## `run_component.py` objective

Output-logit KL of the replacement (student) forward vs the original-MLP (teacher) forward
under the *same* masks, summed over up to three regimes with independent coefficients:

- `--stochastic_coeff` — one stochastic mask draw, subset-routed over the MLP blocks.
- `--adversarial_coeff` — persistent PPGD full-layer adversarial masks (0 disables).
- `--unmasked_coeff` — all component masks = 1, no weight-delta residual (PD's
  `UnmaskedReconLoss` regime; 0 disables).

The smoke (`ddp_smoke_dexp4.sbatch`) runs all three at `0.33`.

## Running

Run as installed modules (the package is editable via `make install-dev`), so no cwd or
`PYTHONPATH` juggling:

```bash
python -m param_decomp_lab.experiments.mlp_compress.run_component --d_expand 4 ...
torchrun --standalone --nproc_per_node=8 -m param_decomp_lab.experiments.mlp_compress.run_component ...
```

## SLURM scripts

The `.sbatch` files derive the repo root at runtime (walk up from `$SLURM_SUBMIT_DIR` to the
dir holding `.venv`) instead of hardcoding a checkout path — **submit from anywhere inside
the repo**. Logs use relative `%j` / `%A_%a` names (land in the submit dir). Multi-node
`ddp_component_adv_dexp4.sbatch` runs one `torchrun` per node via `srun`; the single-/multi-
node smoke is `ddp_smoke_dexp4.sbatch`. The other `smoke_*` / `sweep_*` / `train` scripts
cover the older `run` / `run_all` / `run_attn` / `run_mse` variants.
