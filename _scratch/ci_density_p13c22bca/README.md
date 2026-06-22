# CI density analysis — run `p-13c22bca`

Scratch analysis scripts behind the mean-CI / per-token-CI density report for
`goodfire/param-decomp-llama/p-13c22bca` (`jax-l18-b64-200k-eps1e6`), checkpoint step
200000 — Llama-3.1-8B decomposed at the **layer-18 MLP** (gate/up/down_proj, C=49152
each), **JAX** trainer.

Published report (figures + commentary):
http://mean-ci-p13c22bca.pages.goodfire.pub/  ·  source repo
`goodfire-ai/pages-mean-ci-p13c22bca`.

These are one-run throwaway scripts (hardcoded absolute workspace paths, single
checkpoint) kept here for provenance — hence `_scratch/` (excluded from ruff + pyright).
They run inside the repo `.venv` on a single B200.

## Scripts

| Script | GPU? | What it does |
|---|---|---|
| `render_mean_ci.py` | yes | Restores the run, accumulates the **sorted mean-CI-per-component** curve over the eval stream, snapshots at 8k→33.6M tokens, writes `mean_cis_*.npz` + per-snapshot scatter figures (the in-loop slow-eval figure, emulating the infinite-data limit). |
| `overlay_mean_ci.py` | no | Overlays the sorted mean-CI curves across all token counts (log + linear y), colored by token count. |
| `heatmap_mean_ci.py` | no | *(superseded)* mean-CI density heatmap — meaned over tokens first, so it just thickens the sorted curve. Kept for the record; the per-token version below is the real one. |
| `harvest_ci_hist.py` | yes | Restores the run, accumulates an **unreduced per-token CI histogram** per component (no mean): `jnp.digitize` into 90 log bins over `[1e-9, 1]` + a bin-0 exact-zero underflow, streamed over the eval batches. Saves per-component `(C, N_BINS)` counts + per-component CI sums at 131k / 2.1M / 33.6M tokens to `ci_hist_*.npz`. Never materialises the full `(tokens × C)` tensor. |
| `plot_ci_hist.py` | no | The per-token CI **density heatmap** from the cached `ci_hist_*.npz`. x = component (sorted desc by mean CI @ 33.6M, binned to 256 columns), y = per-token CI log `[1e-6, 1]`, cell = column-normalised density. Two normalisations (`full` over all obs / `active` over CI>0 obs), two color scales (`log`; `linear` rescaled per-column so col max = 1), with the sorted per-component mean CI overlaid on a twin axis `[1e-9, 1]`. |

## Cost / caching

The only expensive step is the GPU harvest (`render_mean_ci.py` ~6 min, `harvest_ci_hist.py`
~16 min, both to 33.6M tokens, single B200). The histogram's ~3× overhead over the mean is
mostly the 91-way one-hot bin stack — a `scatter-add`/`bincount` would shrink it. All
plotting reads the cached `*.npz` (CPU, seconds), so figure revisions are cheap.

Outputs land in `mean_ci_out/` in the Slack workspace (not committed); regenerate by
running the scripts (paths are hardcoded at the top of each).
