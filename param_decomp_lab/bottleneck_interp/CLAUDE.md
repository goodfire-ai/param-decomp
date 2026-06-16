# `param_decomp_lab/bottleneck_interp/`

Analysis tooling for interpreting the **sparse bottleneck code** of a CI fn trained with
a `CiBottleneckConfig` (double-sided JumpReLU bottleneck in
`GlobalSharedTransformerCiFn`; see `param_decomp/ci_fns.py`). Each token position
produces a signed sparse code `z ∈ R^D` (`ci.bottleneck_codes`); these scripts harvest,
characterise, and interpret the distribution of those codes.

All scripts run via `python -m param_decomp_lab.bottleneck_interp.<name>` (no `pd-*`
console scripts). Only `harvest_codes` needs a GPU; everything else is CPU and reads the
harvest output. Outputs go to `PARAM_DECOMP_OUT_DIR/.../bneck_codes/<name>/` by
convention (caller-specified `--out`).

## Pipeline

1. **`harvest_codes`** (GPU) — stream a trained LM decomposition's eval corpus, cache the
   post-gate codes `z` (fp16, chunked `codes_*.pt`), the token `sequences.pt` (for
   context windows), per-dim running `stats.pt`, learned `theta.pt`, and `meta.json`.
   `sequences.pt` makes the harvest *context-preserving*: global flat position `i` maps to
   `sequences[i // seq_len, i % seq_len]`.
2. **`geometry`** — support structure, intrinsic dimension (TwoNN + Levina–Bickel MLE),
   PCA variance curve. Owns `load_codes` (the chunk loader the rest import). Headline
   metric: intrinsic dim vs active-L0 vs linear-PCA dim (curvature signature).
3. **`cluster`** — correlation of code dims + hierarchical clustering (dendrogram + heatmap).
4. **`ising_topology`** — Bhalla/Fel Ising-topology pipeline (FISTA L1 IsingFit → EBIC →
   Louvain → capture/shatter), adapted from sp-viz; binarises on code *activity*. Heavier
   machinery; the simpler `cluster` covers most needs.
5. **`interpret`** — k-means the code vectors into manifold regions (top tokens + context
   windows per region) and pull top-`|z|` context windows per dim (split +z/−z).
6. **`viewer_codes` / `viewer_dims`** — 3D PCA + grand-tour viewers (HTML) over code
   vectors (positions) / code dims respectively; `viewer_3d` is the ported sp-viz
   renderer + template. `--umap` adds a 3D UMAP basis flippable against PCA in the UI.

## Shared helpers — `harvest_io.py`

`interpret` and both viewers load the same things, so that lives in one place:

- `load_harvest(code_dir, max_positions) -> Harvest` — `Harvest(meta, seq_len, tokenizer,
  sequences, codes, flat_tokens)` with `flat_tokens` already aligned to `codes`.
- `kmeans_regions(codes, n_regions)` — MiniBatchKMeans labels.
- `top_tokens(token_ids, tokenizer, k)` — most-common decoded tokens with counts.
- `token_thumbnails(token_ids, tokenizer, patch)` — `(N,3,P,P)` text-image thumbnails.
- `REGION_PALETTE` — RGB triples for region/group colours.

Don't re-inline these in a driver; extend `harvest_io` instead. `load_codes` itself
stays in `geometry` (imported widely); `harvest_io` builds on it.

## Notes / gotchas

- HF tokenizer is typed `Any` here — the transformers stubs don't expose `decode`.
- `viewer_3d.py` is a faithful sp-viz port (generic `build_latent_viewer_data` path only;
  the SAE-atom path was dropped). It keeps sp-viz's `dict[str, Any]` payload style rather
  than repo-native dataclasses — deliberate, to stay close to the original renderer.
- Viewer HTML is self-contained but loads three.js from CDN; size scales with point count
  (thumbnail atlas), so position viewers are tens-to-hundreds of MB. `scp` and open.
