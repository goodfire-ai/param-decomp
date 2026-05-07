# Paper figure & table catalog (`~/param-decomp/paper.md`)

For each figure / table / interactive widget in `paper.md`, this catalog
lists the production script, the WandB run / harvest / data file it consumes,
and the branch the script lives on.

This is a refresh of `paper_figures_catalog.md` against the latest `dev` branch.
Notable changes since the previous catalog:

- The Python package was renamed `spd` → `param_decomp`. All production scripts
  now live under `param_decomp/scripts/...` rather than `spd/scripts/...`.
- The blog-export scripts moved out of the package and now live at
  **top-level** `scripts/blog/{export_components,export_graphs,export_heatmap,export_manual_prompts}.py`.
  The intruder-comparison config moved to `scripts/intruder_comparison.json`.
- `PARAM_DECOMP_OUT_DIR` is now `/mnt/polished-lake/artifacts/mechanisms/param-decomp/`
  (was `.../spd/`). The blog-side `vpd-blog-replit/scripts/build_intruder_figure.py`
  still hardcodes the old `.../spd/intruder_comparison/data.json` path — see §5.
- A unified postprocessing CLI exists: `pd-postprocess <config.yaml>` runs harvest +
  attributions + autointerp + intruder + graph_interp in one shot
  (`param_decomp/postprocess/`).
- The paper now lives at `~/param-decomp/paper.md` (untracked, 3777 lines). It is
  *not* identical to `~/vpd-blog-replit/post.md` (4090 lines): paper.md drops
  `fig:vo_comp_weight_norm`, the 3D `attention-equation` widget, and the
  behavior-3 case study, and it adds `figures/Sum of components.png`. It uses the
  legacy ` ```attention `/` ```attention-multiprompt ` block tags rather than
  the `attention-cards-solo`/`attention-cards` tags that the current
  `vpd-blog-replit/scripts/build_html.py` recognises — see §5.

**Conventions.**

- *Script* paths are relative to `~/param-decomp` unless prefixed with
  `vpd-blog-replit/`.
- Main VPD run = `wandb:goodfire/spd/runs/s-55ea3f9b`. Target model =
  `wandb:goodfire/spd/runs/t-9d2b8f02`. The WandB project is still named `spd`
  — only the Python package and on-disk artefact directory were renamed.
- "Hand-drawn" = didactic figure, no producing script.
- "Interactive widget" = client-side JS in `vpd-blog-replit/js/` rendering a
  JSON file; the data is exported upstream from `param-decomp`.
- Branch defaults to `dev` unless noted otherwise.

---

## 1. Main paper

### 1.1 Method illustrations (didactic, hand-drawn)

| ID | File | Producer |
|---|---|---|
| `fig:sum_components` | `figures/Sum of components.png` | Hand-drawn (LS, contributions statement: "various didactic figures used in the paper"). New file in paper.md, not yet present in `vpd-blog-replit/figures/`; the previous post used `figures/transformer_decomposition.svg`. |
| `fig:simplicity` | `figures/simplicity.png` | Hand-drawn. |
| `fig:placeholder` | `figures/transformer_diag.png` | Hand-drawn. |
| `fig:attr-graph-expl` | `figures/Explaining attribution graphs.png` | Hand-drawn. |

### 1.2 Tables (target model & decomposition stats)

All values come from the main paper run `s-55ea3f9b` (and target `t-9d2b8f02`).
Table contents are copied by hand into the paper from logs / final configs;
there is no dedicated one-shot script.

| ID | Producer / Input |
|---|---|
| `tab:model-hyperparams` | Static; architecture of `t-9d2b8f02`. Final config: `wandb:goodfire/spd/runs/t-9d2b8f02/files/final_config.yaml`. |
| `tab:num-components-per-layer` | Per-layer alive subcomponent counts and L0 stats from `s-55ea3f9b`. Computed via the standard CI summary that `param_decomp/run_param_decomp.py` / `param_decomp/figures.py` log; numbers transcribed from eval output. |
| `tab:vpd-ce-compute-compar` | Validation CE under {target, unmasked, stochastic, rounded@{0,0.1,0.5}, CI-as-mask} on `s-55ea3f9b`. Standard eval pipeline of `param_decomp/run_param_decomp.py`; numbers assembled by hand from eval logs. |
| `tab:vpd-pgd-ce` | KL to target under PGD with $n^{\text{adv}} \in \{20,40,80,160,320\}$. Driven by `param_decomp/metrics/pgd_utils.py` / `param_decomp/persistent_pgd.py`. The CE-vs-`r` sweep helper is on `feature/attn_plots` (commit `ad181bfa Add fixed source (r) sweep script for CE loss comparison (#448)`); also produces `fig:adv-vs-no-adv`. |
| `tab:ci-hyperparams` | Static; from CI-function definition in `param_decomp/configs.py` and `s-55ea3f9b` final config. |
| `tab:vpd-loss-coefficients` / `tab:vpd-subcomponent-counts` | Static; from `s-55ea3f9b` final config YAML. |
| `tab:vpd-train-losses` / `tab:vpd-eval-losses` | Final-step losses + eval reconstruction losses from `s-55ea3f9b` (logged automatically). |

### 1.3 Generation comparisons (interactive)

| ID | Renderer | Data | Producer |
|---|---|---|---|
| `fig:generations-showcase` | `vpd-blog-replit/js/generation_comparisons.js` (block tag ` ```generations `) | `data/generation_comparisons.json` | Generations under {target, unmasked, stochastic, rounded@0/0.1/0.5, CI-as-mask, PGD@20-step} from `s-55ea3f9b`. **No checked-in exporter.** Data is pre-existing in the blog repo; if it needs regenerating, a one-shot would have to be written (likely belongs in `scripts/blog/`). |

### 1.4 Reconstruction-vs-sparsity Pareto plots

| ID | File | Producer |
|---|---|---|
| `fig:pareto-mse` | `figures/pareto_mse_v4.png` | **No producer in any open branch.** Compares VPD `s-55ea3f9b` capacity sweep vs PLT/CLT MSE sweeps (`pile_local_sweep_jose` 4k, `pile_local_sweep_jose_32k` 32k) in the `mats-sprint` WandB project. Authored by BB; one-off WandB-pulling matplotlib script that was not checked in. A newer `figures/pareto_mse_v5.png` exists in `vpd-blog-replit/figures/` but isn't referenced by `paper.md`. |
| `fig:pareto-e2e` | `figures/pareto_e2e_v4.png` | Same situation. Inputs: `pile_e2e_sweep_jose`, `pile_e2e_sweep_jose_32k` + VPD capacity sweep. |

### 1.5 Components showcase (interactive)

| ID | Renderer | Data | Producer |
|---|---|---|---|
| `fig:components-showcase` | `vpd-blog-replit/js/components.js` + `model_overview.js` (block tag ` ```components `) | `data/model-overview/{index.json, comp-index.json, labels.json, components/, weights/}` | `scripts/blog/export_components.py` (top-level `scripts/`, not under the package). Pulls harvest activation examples + autointerp labels for `s-55ea3f9b`; emits per-module weight tiles and per-component activation columns. Run as `uv run python -m scripts.blog.export_components --out-dir ../vpd-blog-replit/data/model-overview`. |

### 1.6 Intruder detection

| ID | File | Producer |
|---|---|---|
| `fig:intruder-score` | `figures/intruder_score_bar_chart_clean.png` | Two-stage. **(a)** `param_decomp/harvest/scripts/compare_intruder_scores.py` reads harvest DBs for the VPD + transcoder runs listed in `scripts/intruder_comparison.json` and writes `PARAM_DECOMP_OUT_DIR/intruder_comparison/data.json`. (Intruder scores per component come from `param_decomp/harvest/intruder.py`; per-run intruder evals are submittable via `pd-postprocess` or directly via the CLI in `param_decomp/harvest/scripts/run_intruder_slurm_cli.py`.) **(b)** `vpd-blog-replit/scripts/build_intruder_figure.py` reads that JSON and emits the PNG via Plotly. ⚠ The blog-side script still hardcodes `/mnt/polished-lake/artifacts/mechanisms/spd/intruder_comparison/data.json` — see §5. |

### 1.7 Feature splitting

| ID | File | Producer |
|---|---|---|
| `fig:feature_splitting` | `figures/feature_splitting.png` | "Alive subcomponents vs capacity": uses VPD `capacity_sweep` (0.5×, 1×, 2×, 4×) + PLT/CLT 4k+32k. **No producer in any open branch.** BB one-off (matches `fig:pareto-mse`). A `feature_splitting_v2.png` also sits in `vpd-blog-replit/figures/` but isn't referenced by paper.md. |
| `fig:splitting-heatmap` | `figures/split_heatmap.png` | Rendered by `vpd-blog-replit/scripts/build_split_heatmap.py`, but **values are hardcoded in that script** (parsed off the original chart). Re-running the cosine-sim splitting analysis end-to-end is not currently possible without re-deriving BB's notebook. |

### 1.8 Attention analysis (Layer 1) — main text

All scripts on `dev`. (`feature/attn_plots` was where the attention analysis was
originally developed and contains older, sometimes more extensive variants;
`plot_single_comp_frac.py` originated there and is now also on dev.)

| ID | File in paper | Script |
|---|---|---|
| `fig:qk_comp_weight_norm` | `figures/layer1_qk_combined.png` | `param_decomp/scripts/plot_component_head_norms/plot_component_head_norms.py` (`_plot_proj_pair_combined` with `pair_name="qk"`, line 152 → `out/<run_id>/layer{idx}_{pair_name}_combined.png`). |
| `fig:attn_contrib_grid` | `figures/layer1_qk_pair_lines_combined.png` | `param_decomp/scripts/plot_qk_c_attention_contributions/plot_qk_c_attention_contributions.py` (`_plot_pair_lines_combined`, line 785 → `out/<run_id>/lines_combined/layer{idx}_qk_pair_lines_combined.png`). The `_nobb`-suffixed copy on disk in `vpd-blog-replit/figures/layer1_qk_pair_lines_combined_nobb.png` is a historical manual rename; paper.md references the un-suffixed name. |
| `fig:prev_token_scores` | `figures/prev_token_scores_combined.png` | `param_decomp/scripts/detect_prev_token_heads/detect_prev_token_heads.py` (line 234 → `out/<run_id>/prev_token_scores_combined.png`). The random-token panel half is produced by sibling `detect_prev_token_heads_random_tokens.py`; the combined plot uses both `.npy` outputs. |
| `fig:attn_patterns_q_intv` | `figures/attn_q_L1_top10_n256_grid.png` | `param_decomp/scripts/attention_ablation_experiment/plot_single_comp_frac.py` (line 500, format `{plot_type}_{proj_label}_L{layer}_top{N}_n{n_samples}_grid.png`; with `attn`/`q`/`1`/`10`/`256` → exactly `attn_q_L1_top10_n256_grid.png`). Originated on `feature/attn_plots` (`9afe3f38 Add single-component ablation plots`); also lives on dev. |
| `fig:prev_tok_ov_overlap_k_329` | `figures/layer1_ov_paper_figure_k_329.png` | `param_decomp/scripts/plot_wv_subspace_overlap/plot_wv_subspace_overlap.py` (`_plot_ov_paper_figure`, lines 680+); parameterized by `--k_comp` for which K subcomponent to filter on. LaTeX writeup of the metric: `param_decomp/scripts/plot_wv_subspace_overlap/out/wv_overlap_writeup.tex` (if generated). |
| `fig:pkv` | `figures/pkv_layer1.png` (paper.md) | `param_decomp/scripts/plot_kv_coactivation/plot_kv_coactivation.py` (`_plot_pkv_combined`, line 190 → `out/<run_id>/p_kv_combined/layer{layer_idx}.png`). The on-disk file in `vpd-blog-replit/figures/` is `pkv_combo_layer1.png` (also a sibling `pkv_combo_layer1_sub.png`); paper.md refers to `pkv_layer1.png`, so the figure must be manually copied/renamed. |

Note: `fig:vo_comp_weight_norm` (the V/O analogue of `fig:qk_comp_weight_norm`)
appears in `post.md` but is **not** referenced in `paper.md`. The producing
script `plot_component_head_norms.py` already emits `layer{idx}_vo_combined.png`
in the same run, so the figure is one symlink/edit away if it gets restored.

### 1.8b Attention dynamic widgets (interactive)

All consume JSON exported from `param-decomp`; renderers live under
`vpd-blog-replit/js/`.

⚠ paper.md uses block tags ` ```attention ` and ` ```attention-multiprompt `,
neither of which is recognised by the current dispatcher in
`vpd-blog-replit/scripts/build_html.py` (which accepts
`attention-cards-solo` / `attention-cards` / `attention-equation` /
`attention-qk-grid`). For paper.md to render, either the tags need to be
updated in the markdown (most likely `attention` → `attention-cards-solo` and
`attention-multiprompt` → `attention-cards`) or the dispatcher needs an
alias. See §5.

| ID | Renderer (current) | Data | Producer |
|---|---|---|---|
| `fig:dynamic-1` | `js/attention_heatmap_cards.js` (intended `attention-cards-solo`) | `data/attention/intro-layer-1.json` | `param_decomp/scripts/interactive_qk_contributions/compute_data.py` (per-prompt per-pair Data-Dependent Interaction Strength tensor for a specified layer). The renamed package was previously `plot_qk_c_datapoint/`; rework on dev commit `312b9cf3 Rework compute_data.py + add compute_pair_data.py` and `c024d703` (rename). |
| `fig:dynamic-2` | `js/attention_heatmap_cards.js` (intended `attention-cards`) | `data/attention/30-dataset-layer-1.json` | Same producer with a 30-prompt dataset (`--dataset_samples 30 --seq_len_min 20 --seq_len_max 40`). |

The `attention-equation` widget (`js/attention_equation_3d.js`) and the
behavior-3 `attention-qk-grid` widgets that exist in `post.md` are **not**
present in `paper.md`.

### 1.9 Attribution graphs (interactive)

| ID | Renderer | Data | Producer |
|---|---|---|---|
| `graph:princess` | `js/graph.js` (block tag ` ```graph `) | `data/graphs/princess-full.json` (+ `princess-full-details.json`) | Full PGD-pruned attribution graph for `The princess lost her crown.` predicting `·her`. Pipeline: post-hoc CI optimization (`app:posthoc_ci`, code in `param_decomp/dataset_attributions/` + `param_decomp/graph_interp/`) → gradient-attribution computation → export via `scripts/blog/export_graphs.py`. The `graph` skill in `~/.claude/skills/graph` is the loader for analysis. |
| `graph:princess_ci_masked` | `js/graph.js` | `data/graphs/princess-minimal{,-details}.json` | Same prompt, CI-only pruning. Same exporter; toggled via `lambda_recon` / sampling settings in the post-hoc CI optimizer. |
| `graph:prince-full` | `js/graph.js` | `data/graphs/prince-full{,-details}.json` | `The prince lost his crown.` adversarially pruned. |
| `graph:prince-minimal` | `js/graph.js` | `data/graphs/prince-minimal{,-details}.json` | CI-only variant. |
| `graph:bracket` | `js/graph.js` | `data/graphs/bracket-full{,-details}.json` | `< u , v >` predicting `>` after `v`. |
| `graph:bracket_u` | `js/graph.js` | `data/graphs/bracket-u-full{,-details}.json` | Same prompt predicting `>` after `u`. |
| `graph:bracket_ci` | `js/graph.js` | `data/graphs/bracket-minimal{,-details}.json` | CI-only variant. |

Prompt-viewers (` ```prompt-viewer ` blocks, `js/components.js` activation widget)
read `data/manual-prompts/*.json`, exported via `scripts/blog/export_manual_prompts.py`.

### 1.10 Model editing

The producing scripts that build the **figures** are still **only on
`origin/paper/editing-section`**, not on `dev`. The branch contains the full
`spd/editing/` module:

```
spd/editing/component_trainer.py        # train the component edit
spd/editing/lora_baseline.py            # train LoRA baselines
spd/editing/run_pareto_export.py        # run α-sweep + λ-sweep
spd/editing/generate_pareto_plots.py    # produce figures/editing/pareto.{pdf,png}
spd/editing/export_blog_heatmap.py      # produce data/editing-kl-heatmap*.json
spd/editing/{compare,viz,utils}.py
```

Output artifacts are committed on that branch under
`figures/editing/{pareto,kl_histogram}.{pdf,png}` and
`figures/editing/{pareto_data,training-heatmap,training-heatmap-lora}.json`.

A different (lighter-weight) `param_decomp/editing/` module exists on `dev`
with `_editing.py`, `generate_token_divergence.py`, and a README — but
**not** `component_trainer.py`, `run_pareto_export.py`,
`generate_pareto_plots.py`, or `export_blog_heatmap.py`. The
`scripts/blog/export_heatmap.py` script on `dev` imports from
`param_decomp.editing.component_trainer` and
`param_decomp.editing.generate_pareto_plots`, which means **it cannot run on
dev as-is** — `paper/editing-section` must be merged in (or those modules
ported) for `fig:model-editing-heatmap` to be regenerated. ⚠

| ID | File / Data | Script |
|---|---|---|
| `fig:model-editing-heatmap` | `data/editing-kl-heatmap.json` (PD), `data/editing-kl-heatmap-lora.json` (LoRA) | `spd/editing/export_blog_heatmap.py` on `paper/editing-section`, or the dev-side `scripts/blog/export_heatmap.py` once the editing branch is merged. JS renderer: `vpd-blog-replit/js/editing_heatmap.js` (block tag ` ```heatmap `). |
| `fig:model-editing-pareto` | `figures/editing_pareto.png` | `spd/editing/run_pareto_export.py` produces `figures/editing/pareto_data.json`; `spd/editing/generate_pareto_plots.py` plots it as `pareto.png` (manually renamed to `editing_pareto.png` for the post). |

The untracked `figures/editing/` directory in the dev working tree
(`kl_histogram.{pdf,png}`, `pareto.{pdf,png}`, `pareto_data.json`,
`training-heatmap{,-lora}.json`) is exactly the artifacts checked into
`paper/editing-section`.

---

## 2. Appendix figures / tables

### 2.1 End-to-end transcoders & seed stability

| ID | File | Producer |
|---|---|---|
| `fig:pareto-e2e` | `figures/pareto_e2e_v4.png` | See §1.4 — no checked-in producer. |
| `tab:seed-mmcs` | inline | Cross-seed MMCS for VPD U/V vectors, PLT/CLT, etc. Inputs: VPD multiseed (`goodfire/spd?nw=n9l0amrrudc`), transcoder multiseed (`pile_multiseed_jose2`), hidden-act aux-loss VPD (`s-aa4fec0a`). **No checked-in producer**; LS authored this analysis. |

### 2.2 Stochastic vs adversarial training loss

| ID | File | Script |
|---|---|---|
| `fig:adv-vs-no-adv` | `figures/adv_vs_no_adv.png` | CE loss vs source $r$ for the main run vs no-adv control `s-05ef623e`. **Likely** `feature/attn_plots` commit `ad181bfa Add fixed source (r) sweep script for CE loss comparison (#448)` (matching semantics; same script also produces `tab:vpd-pgd-ce`). That commit appears to **not** be on `dev` — verify before reproducing. |

### 2.3 Layer-1 OV alignment — appendices `app:ov-alignment-k119` and the K.119 figure

| ID | File / Producer |
|---|---|
| `fig:prev_tok_ov_overlap_k_119` | `figures/layer1_ov_paper_figure_k_119.png` | Same script as `fig:prev_tok_ov_overlap_k_329` (`param_decomp/scripts/plot_wv_subspace_overlap/plot_wv_subspace_overlap.py`), run with `--k_comp 119`. |
| Heads 0–5 read-aligned V / write-aligned O top-5 tables (under `app:ov-alignment-k119`) | `param_decomp/scripts/plot_wv_subspace_overlap/analyze_ov_subspace_semantics.py` (sibling of `plot_wv_subspace_overlap.py`). |

### 2.4 Nonlinear interaction matrices (`app:non_linear_plots`, `app:interactions-gis-vs-coact`)

| ID | File | Script |
|---|---|---|
| `fig:I_h` (layers 0–3 in the appendix grid) | `figures/I_h_{0,1,2,3}_mlp_c_fc.png` | `param_decomp/scripts/geometric_interaction/interaction_analysis.py` (`plot_heatmaps`, line 162; output `<output_dir>/heatmaps/I.png` then per-module renamed/saved as `I_h_<safe_module_name>.png`). The package also contains `harvest_activations.py`, `harvest_interaction.py` (compute the H matrix from activation cross-products), and `statistical_analysis.py`. CLI: `python -m param_decomp.scripts.geometric_interaction.interaction_analysis --model_path=wandb:goodfire/spd/runs/s-55ea3f9b`. |
| `fig:I_dist` | `figures/I_dist_h_{0,1,2,3}_mlp_c_fc.png` | Same script (`plot_I_distribution`, line 199; `<output_dir>/histograms/I_dist_<safe_module_name>.png`). The `_h_<n>_mlp_c_fc` suffix encodes the Llama module name `h.<n>.mlp.c_fc` after `name.replace(".", "_")`. |

(The same two figures double as `fig:I_h` / `fig:I_dist` in the body and are
re-shown layer-by-layer in `app:non_linear_plots`.)

---

## 3. Production scripts inside `vpd-blog-replit`

Re-runnable directly from the blog repo once upstream data exists:

- `scripts/build_intruder_figure.py` → `figures/intruder_score_bar_chart_clean.png` (Plotly via Kaleido). ⚠ Hardcoded input `/mnt/polished-lake/artifacts/mechanisms/spd/intruder_comparison/data.json` (old `spd` path); the upstream writer `compare_intruder_scores.py` now writes to `/mnt/.../param-decomp/intruder_comparison/data.json`. Symlink, copy, or fix the path before running.
- `scripts/build_split_heatmap.py` → `figures/split_heatmap.png` (Plotly). **Input is hardcoded** in the script.
- All interactive widgets (`js/graph.js`, `attention_heatmap_cards.js`, `attention_qk_grid.js`, `attention_equation_3d.js`, `components.js`, `model_overview.js`, `editing_heatmap.js`, `generation_comparisons.js`) consume JSON in `data/` exported upstream from `param-decomp`.
- `scripts/build_html.py` is the static-site builder; relevant for §5.

---

## 4. Branches and where things live

| Branch | What it contains for paper figures |
|---|---|
| `dev` (current) | `scripts/blog/{export_components,export_graphs,export_heatmap,export_manual_prompts}.py`; the attention-analysis suite under `param_decomp/scripts/`: `plot_component_head_norms`, `plot_qk_c_attention_contributions`, `plot_attention_offset_profiles`, `plot_kv_coactivation`, `plot_wv_subspace_overlap` (+ `analyze_ov_subspace_semantics`), `plot_qk_c_datapoint`, `plot_prompt_attention`, `detect_prev_token_heads`, `attention_ablation_experiment` (incl. `plot_single_comp_frac`), `plot_component_activations`, `geometric_interaction`, `sweep_summary`, `interactive_qk_contributions` (`compute_data.py`, `compute_pair_data.py` for the dynamic attention widgets). Also `param_decomp/harvest/scripts/compare_intruder_scores.py`. The `param_decomp/postprocess/` module orchestrates harvest + attributions + autointerp + intruder + graph_interp. |
| `feature/attn_plots` | Where the attention analysis was originally developed; mostly the same scripts as dev (under the old `spd/` name). Extras vs dev: `26a66c50 Update single_comp_frac plots`, `9afe3f38 Add single-component ablation plots`, `2c64b3f5 Add multi-pair fractional attention change plot`, `4725d369 Add fractional attention change plots normalized by cross-head mean baseline`, `dc87a8dc Add per-sample attention pattern plots with token labels`, `ad181bfa Add fixed source (r) sweep script for CE loss comparison`. The `single_comp_frac` script originated here and is now on dev. The `fixed source (r) sweep` script (likely the producer of `fig:adv-vs-no-adv` and `tab:vpd-pgd-ce`) appears to **not** be on dev — verify before reproducing. |
| `paper/editing-section` (origin) | The entire `spd/editing/` module + committed artifacts under `figures/editing/`. Required for `fig:model-editing-pareto` and `fig:model-editing-heatmap` (the dev-side `scripts/blog/export_heatmap.py` imports from this module). |
| `feature/model-editing`, `merge/2c-editing` | Lighter-weight editing modules; do not contain `component_trainer.py` / `run_pareto_export.py` / `generate_pareto_plots.py` / `export_blog_heatmap.py`. |
| `spd-paper`, `feature/paper-vis` | No unique paper figure producers found beyond what's on dev. |

---

## 5. Things you cannot currently reproduce from a checked-in script

1. `fig:pareto-mse` and `fig:pareto-e2e` — BB's one-off WandB-pulling matplotlib script; only the rendered PNGs are in the blog repo.
2. `fig:feature_splitting` — same situation (BB's one-off).
3. `fig:splitting-heatmap` — analysis script lost; PNG is rendered by `vpd-blog-replit/scripts/build_split_heatmap.py` from a hardcoded array.
4. `tab:seed-mmcs` — LS's seed-MMCS analysis; not committed.
5. `fig:generations-showcase` data (`data/generation_comparisons.json`) — exporter not pinned to a checked-in path.
6. `fig:adv-vs-no-adv` and `tab:vpd-pgd-ce` — fixed-source-`r` sweep script lives only on `feature/attn_plots`.
7. `fig:model-editing-{pareto,heatmap}` — depends on the `spd/editing/` module that is only on `paper/editing-section`. The dev-side `scripts/blog/export_heatmap.py` references that module by import; it will `ImportError` until the editing branch is merged.

Other discrepancies that need fixing before a clean rebuild:

- `vpd-blog-replit/scripts/build_intruder_figure.py` reads from `/mnt/.../spd/intruder_comparison/data.json`; the writer (`param_decomp/harvest/scripts/compare_intruder_scores.py`) now writes to `/mnt/.../param-decomp/intruder_comparison/data.json`.
- `paper.md` uses block tags ` ```attention ` and ` ```attention-multiprompt `, but `vpd-blog-replit/scripts/build_html.py` only recognises `attention-cards-solo` / `attention-cards` / `attention-equation` / `attention-qk-grid`. paper.md needs either tag updates or a dispatcher alias to render.
- `paper.md` references `figures/Sum of components.png` and `figures/pkv_layer1.png`; the former is missing from `vpd-blog-replit/figures/`, and the latter has to be sourced/renamed from the actual output `pkv_combo_layer1.png` (or directly from `<run>/p_kv_combined/layer1.png`). The QK pair-lines figure is referenced as `layer1_qk_pair_lines_combined.png` while only the `_nobb` variant currently exists in the blog repo.

Everything else has a known producer (script + branch + input run id).

---

## 6. End-to-end reproduction recipe (sketch)

To rerun the full paper from `t-9d2b8f02` + `s-55ea3f9b`:

1. **Postprocess** the main VPD run:
   `pd-postprocess param_decomp/postprocess/pile.yaml`
   (harvest → autointerp interpret + detection/fuzzing evals → intruder eval → dataset_attributions → graph_interp; the run id is set inside the YAML, not on the CLI).
2. **Layer-1 attention figures**: run each `param_decomp/scripts/plot_*` and `detect_prev_token_heads`/`detect_prev_token_heads_random_tokens` against `wandb:goodfire/spd/runs/s-55ea3f9b`. Each writes to `param_decomp/scripts/<name>/out/<run_id>/`.
3. **Single-component ablation grid**: `python -m param_decomp.scripts.attention_ablation_experiment.plot_single_comp_frac --model_path=wandb:goodfire/spd/runs/s-55ea3f9b`.
4. **Interaction matrices**: `python -m param_decomp.scripts.geometric_interaction.interaction_analysis --model_path=wandb:goodfire/spd/runs/s-55ea3f9b`.
5. **Intruder bar chart**: the intruder eval is part of `pd-postprocess`; then run `python -m param_decomp.harvest.scripts.compare_intruder_scores` to write `data.json`, then `python vpd-blog-replit/scripts/build_intruder_figure.py` (fix the hardcoded `spd/` path first).
6. **Adversarial sweep**: run the `feature/attn_plots` "fixed source (r) sweep" script for `fig:adv-vs-no-adv` and `tab:vpd-pgd-ce`.
7. **Model editing**: switch to (or merge) `paper/editing-section`. Run `spd/editing/run_pareto_export.py` → `spd/editing/generate_pareto_plots.py` → `spd/editing/export_blog_heatmap.py` (or the dev-side `scripts/blog/export_heatmap.py` once the imports resolve).
8. **Interactive attention widgets** (paper §1.8b): `python -m param_decomp.scripts.interactive_qk_contributions.compute_data wandb:goodfire/spd/runs/s-55ea3f9b --layer 1 ...` for each prompt set; outputs `data/attention/*.json`.
9. **Blog data exports** (run from `~/param-decomp`):
   - `uv run python -m scripts.blog.export_components --out-dir ../vpd-blog-replit/data/model-overview`
   - `uv run python -m scripts.blog.export_graphs --out-dir ../vpd-blog-replit/data/graphs`
   - `uv run python -m scripts.blog.export_heatmap --out-dir ../vpd-blog-replit/data` (requires editing module on path, see §1.10)
   - `uv run python -m scripts.blog.export_manual_prompts --out-dir ../vpd-blog-replit/data/manual-prompts`
10. **Pareto / feature-splitting / seed-MMCS / generation-comparison JSON**: re-derive (no checked-in producer).
11. **Build the blog**: `vpd-blog-replit/scripts/build_static.py` (or `build_html.py`). For `paper.md` specifically, fix the ` ```attention ` block-tag mismatch first (see §5).
