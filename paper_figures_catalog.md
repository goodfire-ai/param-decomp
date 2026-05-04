# Paper figure & table catalog (`~/vpd-blog-replit/post.md`)

For each figure / table / interactive widget in the VPD paper, this catalog
lists the production script, the WandB run / harvest / data file it consumes,
and the branch the script lives on.

**Conventions.**

- *Script* = path in the spd repo unless prefixed with `vpd-blog-replit/`.
- Main VPD run = `s-55ea3f9b`. Target model = `t-9d2b8f02`. `PARAM_DECOMP_OUT_DIR =
  /mnt/polished-lake/artifacts/mechanisms/param-decomp/`.
- "Hand-drawn" = didactic figure, no producing script.
- "Interactive widget" = client-side JS in `vpd-blog-replit/js/` rendering a
  JSON file; the data is exported upstream from spd.
- "Inherited PNG" = sits in `vpd-blog-replit/figures/` but is generated upstream
  in spd and copied in.
- Branch defaults to `dev` unless noted otherwise.

---

## 1. Main paper

### 1.1 Method illustrations (didactic, hand-drawn)

| ID | File | Producer |
|---|---|---|
| `fig:sum_components` | `figures/transformer_decomposition.svg` | Hand-drawn (LS, contributions statement: "various didactic figures used in the paper"). |
| `fig:placeholder` | `figures/transformer_diag.png` | Hand-drawn. |
| `fig:simplicity` | `figures/simplicity.png` | Hand-drawn (currently commented out). |
| `fig:attr-graph-expl` | `figures/Explaining attribution graphs.png` | Hand-drawn. |

### 1.2 Tables (target model & decomposition stats)

| ID | Producer / Input |
|---|---|
| `tab:model-hyperparams` | Static; values are the architecture of `t-9d2b8f02`. Final config: `wandb:goodfire/spd/runs/t-9d2b8f02/files/final_config.yaml`. |
| `tab:num-components-per-layer` | Static; per-layer alive subcomponent counts and L0 stats from `s-55ea3f9b`. Computed via the standard CI summary that `spd/run_spd.py` / `spd/figures.py` log; no dedicated one-shot script. |
| `tab:vpd-ce-compute-compar` | Validation CE under {target, unmasked, stochastic, rounded@{0,0.1,0.5}, CI-as-mask} on `s-55ea3f9b` + `t-9d2b8f02`. Computed via the standard eval pipeline of `spd/run_spd.py`; numbers assembled by hand from eval logs. |
| `tab:vpd-pgd-ce` | KL to target under PGD with $n^{\text{adv}} \in \{20,40,80,160,320\}$. Driven by `spd/metrics/pgd_utils.py` / `spd/persistent_pgd.py`. The CE-vs-`r` sweep helper is `feature/attn_plots` commit `ad181bfa Add fixed source (r) sweep script for CE loss comparison (#448)` (also the producer of `fig:adv-vs-no-adv`). |
| `tab:ci-hyperparams` | Static; from CI-function definition in `spd/configs.py` and `s-55ea3f9b` final config. |
| `tab:vpd-loss-coefficients` / `tab:vpd-subcomponent-counts` | Static; from `s-55ea3f9b` final config YAML. |
| `tab:vpd-train-losses` / `tab:vpd-eval-losses` | Final-step losses + eval reconstruction losses from `s-55ea3f9b` (logged automatically). |

### 1.3 Generation comparisons (interactive)

| ID | Renderer | Data | Producer |
|---|---|---|---|
| `fig:generations-showcase` | `vpd-blog-replit/js/generation_comparisons.js` | `data/generation_comparisons.json` | Generations under {target, unmasked, stochastic, rounded@0/0.1/0.5, CI-as-mask, PGD@20-step} from `s-55ea3f9b`. **No checked-in exporter.** Likely a one-shot in the `spd/scripts/blog/` family. |

### 1.4 Reconstruction-vs-sparsity Pareto plots

| ID | File | Producer |
|---|---|---|
| `fig:pareto-mse` | `figures/pareto_mse_v4.png` | **No producer in any open branch.** Compares VPD `s-55ea3f9b` capacity sweep vs PLT/CLT MSE sweeps `pile_local_sweep_jose` (4k) and `pile_local_sweep_jose_32k` in `mats-sprint`. Authored by BB; one-off WandB-pulling matplotlib script. |
| `fig:pareto-e2e` | `figures/pareto_e2e_v4.png` | Same situation. Inputs: `pile_e2e_sweep_jose`, `pile_e2e_sweep_jose_32k` + VPD capacity sweep. |

### 1.5 Components showcase (interactive)

| ID | Renderer | Data | Producer |
|---|---|---|---|
| `fig:components-showcase` | `vpd-blog-replit/js/components.js` + `model_overview.js` | `data/model-overview/{index.json, comp-index.json, labels.json, components/, weights/}` | `spd/scripts/blog/export_components.py`. Pulls harvest activation examples + autointerp labels for `s-55ea3f9b`; emits per-module weight tiles (32×32). |

### 1.6 Intruder detection

| ID | File | Producer |
|---|---|---|
| `fig:intruder-score` | `figures/intruder_score_bar_chart_clean.png` | Two-stage: (a) `spd/harvest/scripts/compare_intruder_scores.py` reads `harvest.db` for VPD + transcoder runs listed in `spd/scripts/intruder_comparison.json` and writes `PARAM_DECOMP_OUT_DIR/intruder_comparison/data.json`; (b) `vpd-blog-replit/scripts/build_intruder_figure.py` reads that JSON and emits the PNG via Plotly. |

### 1.7 Feature splitting

| ID | File | Producer |
|---|---|---|
| `fig:feature_splitting` | `figures/feature_splitting.png` | "Alive subcomponents vs capacity": uses VPD `capacity_sweep` (0.5×, 1×, 2×, 4×) + PLT/CLT 4k+32k. **No producer in any open branch.** BB one-off. |
| `fig:splitting-heatmap` | `figures/split_heatmap.png` | The PNG is rendered by `vpd-blog-replit/scripts/build_split_heatmap.py`, but **values are hardcoded in that script** (parsed off the original chart). Re-running the cosine-sim splitting analysis end-to-end is not currently possible without re-deriving BB's notebook. |

### 1.8 Attention analysis (Layer 1) — main text

All scripts on `dev`. (`feature/attn_plots` contains older versions; `single_comp_frac.py` originated there and is now also on dev.)

| ID | File | Script |
|---|---|---|
| `fig:qk_comp_weight_norm` | `figures/layer1_qk_combined.png` | `spd/scripts/plot_component_head_norms/plot_component_head_norms.py` (output `layer{idx}_{pair_name}_combined.png` with `pair_name="qk"`, line ~151). |
| `fig:vo_comp_weight_norm` | `figures/layer1_vo_combined.png` | Same script; `_plot_vo_combined`, line 196. Added in dev commits `7d519298` and `430b33e9`. |
| `fig:attn_contrib_grid` | `figures/layer1_qk_pair_lines_combined_nobb.png` | `spd/scripts/plot_qk_c_attention_contributions/plot_qk_c_attention_contributions.py` (output `layer{idx}_qk_pair_lines_combined.png`, line 785). The `_nobb` suffix in the paper filename is a manual rename / variant. |
| `fig:prev_token_scores` | `figures/prev_token_scores_combined.png` | `spd/scripts/detect_prev_token_heads/detect_prev_token_heads.py` (line 234); sibling `detect_prev_token_heads_random_tokens.py` for the random-token panel. |
| `fig:attn_patterns_q_intv` | `figures/attn_q_L1_top10_n256_grid.png` | `spd/scripts/attention_ablation_experiment/plot_single_comp_frac.py` (line 445, format `{plot_type}_{proj_label}_L{layer}_top{N}_n{n_samples}_grid.png`; with `attn`/`q`/`1`/`10`/`256` → exactly `attn_q_L1_top10_n256_grid.png`). Originated on `feature/attn_plots` (`9afe3f38 Add single-component ablation plots`). |
| `fig:prev_tok_ov_overlap_k_329` | `figures/layer1_ov_paper_figure_k_329.png` | `spd/scripts/plot_wv_subspace_overlap/plot_wv_subspace_overlap.py`, parameterized by which K subcomponent filters the dataset. LaTeX writeup of the metric: `spd/scripts/plot_wv_subspace_overlap/out/wv_overlap_writeup.tex`. |
| `fig:prev_tok_ov_overlap_k_119` | `figures/layer1_ov_paper_figure_k_119.png` | Same script, run with `--k_comp 119`. |
| `fig:pkv` | `figures/pkv_combo_layer1.png` | `spd/scripts/plot_kv_coactivation/plot_kv_coactivation.py` (`_plot_pkv_combined`, lines 160-194). |

### 1.8b Attention dynamic widgets (interactive)

All consume JSON exported from spd; renderer is `vpd-blog-replit/js/`.

| ID | Renderer | Data | Producer |
|---|---|---|---|
| `attention-equation` (intro) | `js/attention_equation_3d.js` | `data/attention/intro-layer-1.json` | Per-prompt per-pair Data-Dependent Interaction Strength tensor. Almost certainly produced by `compute_data.py` / `compute_pair_data.py` (dev commit `312b9cf3 Rework compute_data.py + add compute_pair_data.py`). Likely under `spd/scripts/plot_qk_c_datapoint/` or the renamed `interactive_qk_contributions/` (commit `c024d703`). |
| `fig:dynamic-1` | `js/attention_heatmap_cards.js` | `data/attention/intro-layer-1.json` | Same data file. |
| `fig:dynamic-2` | `js/attention_heatmap_cards.js` | `data/attention/30-dataset-layer-1.json` | Same producer with a 30-prompt dataset. |
| Behavior-3 attention-qk-grids | `js/attention_qk_grid.js` | `data/attention/behavior-3/outputs/{k-218-copula,k-218-overzealous,k-485-copula,compare-single}.json` | Same producer, different prompt sets selected for q.308×k.218 and q.308×k.485 analyses. |

### 1.9 Attribution graphs (interactive)

| ID | Renderer | Data | Producer |
|---|---|---|---|
| `graph:princess` | `js/graph.js` | `data/graphs/princess-full{,-details}.json` | Full PGD-pruned attribution graph for `The princess lost her crown.` predicting `·her`. Pipeline: post-hoc CI optimization (`app:posthoc_ci`, `spd/dataset_attributions/` + `spd/graph_interp/`) → gradient-attribution computation → export via `spd/scripts/blog/export_graphs.py`. The `graph` skill in `~/.claude/skills/graph` is the loader. |
| `graph:princess_ci_masked` | `js/graph.js` | `data/graphs/princess-minimal*.json` | Same prompt, CI-only pruning. Same exporter; toggled via `lambda_recon` / sampling settings in the post-hoc CI optimizer. |
| `graph:prince-full` | `js/graph.js` | `data/graphs/prince-full*.json` | `The prince lost his crown.` adversarially pruned. |
| `graph:prince-minimal` | `js/graph.js` | `data/graphs/prince-minimal*.json` | CI-only variant. |
| `graph:bracket` | `js/graph.js` | `data/graphs/bracket-full*.json` | `< u , v >` predicting `>` after `v`. |
| `graph:bracket_u` | `js/graph.js` | `data/graphs/bracket-u-full*.json` | Same prompt predicting `>` after `u`. |
| `graph:bracket_ci` | `js/graph.js` | `data/graphs/bracket-minimal*.json` | CI-only variant. |

Prompt-viewers (`prompt-viewer` widgets) read `data/manual-prompts/*.json`,
exported via `spd/scripts/blog/export_manual_prompts.py`.

### 1.10 Model editing

The producing scripts are **only on `origin/paper/editing-section`**, not on `dev`. The branch contains the full `spd/editing/` module:

```
spd/editing/component_trainer.py        # train the component edit
spd/editing/lora_baseline.py            # train LoRA baselines
spd/editing/run_pareto_export.py        # run α-sweep + λ-sweep
spd/editing/generate_pareto_plots.py    # produce figures/editing/pareto.{pdf,png}
spd/editing/export_blog_heatmap.py      # produce data/editing-kl-heatmap*.json
spd/editing/{compare,viz,utils}.py
```

Output artifacts are committed to that branch under `figures/editing/{pareto,kl_histogram}.{pdf,png}` and `figures/editing/{pareto_data,training-heatmap,training-heatmap-lora}.json`.

| ID | File / Data | Script |
|---|---|---|
| `fig:model-editing-heatmap` | `data/editing-kl-heatmap.json`, `data/editing-kl-heatmap-lora.json` | `spd/editing/export_blog_heatmap.py` (on `paper/editing-section`); JS renderer is `vpd-blog-replit/js/editing_heatmap.js`. |
| `fig:model-editing-pareto` | `figures/editing_pareto.png` | `spd/editing/run_pareto_export.py` produces `figures/editing/pareto_data.json`; `spd/editing/generate_pareto_plots.py` plots it as `pareto.png` (manually renamed to `editing_pareto.png` for the post). |

The untracked `figures/editing/` directory in the working tree (`kl_histogram.{pdf,png}`, `pareto.{pdf,png}`, `pareto_data.json`, `training-heatmap{,-lora}.json`) is exactly the artifacts checked into `paper/editing-section`.

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
| `fig:adv-vs-no-adv` | `figures/adv_vs_no_adv.png` | CE loss vs source $r$ for main run vs no-adv control `s-05ef623e`. **Likely** `feature/attn_plots` commit `ad181bfa Add fixed source (r) sweep script for CE loss comparison (#448)` (matching semantics). Branch only. |

### 2.3 Layer-1 OV alignment top-5 tables (`app:ov-alignment-k119`)

| ID | Producer |
|---|---|
| Heads 0–5 read-aligned V / write-aligned O top-5 | `spd/scripts/plot_wv_subspace_overlap/analyze_ov_subspace_semantics.py` (sibling of `plot_wv_subspace_overlap.py`). |

### 2.4 Nonlinear interaction matrices

| ID | File | Script |
|---|---|---|
| `fig:I_h` (layers 0–3) | `figures/I_h_{0,1,2,3}_mlp_c_fc.png` | `spd/scripts/geometric_interaction/interaction_analysis.py`. The package also contains `harvest_activations.py`, `harvest_interaction.py` (compute the H matrix from activation cross-products), `statistical_analysis.py`. CLI: `python spd/scripts/geometric_interaction/interaction_analysis.py --model_path="wandb:goodfire/spd/runs/s-55ea3f9b"`. |
| `fig:I_dist` | `figures/I_dist_h_{0,1,2,3}_mlp_c_fc.png` | Same script, histogram outputs. |

---

## 3. Production scripts inside `vpd-blog-replit`

Re-runnable directly from the blog repo once upstream data exists:

- `scripts/build_intruder_figure.py` → `figures/intruder_score_bar_chart_clean.png` (Plotly via Kaleido). Input: `PARAM_DECOMP_OUT_DIR/intruder_comparison/data.json`.
- `scripts/build_split_heatmap.py` → `figures/split_heatmap.png` (Plotly). **Input is hardcoded** in the script.
- All interactive widgets (`js/graph.js`, `attention_heatmap_cards.js`, `attention_qk_grid.js`, `attention_equation_3d.js`, `components.js`, `model_overview.js`, `editing_heatmap.js`, `generation_comparisons.js`) consume JSON in `data/` exported upstream from spd.

---

## 4. Branches and where things live

| Branch | What it contains for paper figures |
|---|---|
| `dev` (current) | `spd/scripts/blog/{export_components,export_graphs,export_heatmap,export_manual_prompts}.py`; `plot_component_head_norms`, `plot_qk_c_attention_contributions`, `plot_attention_offset_profiles`, `plot_kv_coactivation`, `plot_wv_subspace_overlap` (+ `analyze_ov_subspace_semantics`), `plot_qk_c_datapoint`, `plot_prompt_attention`, `detect_prev_token_heads`, `attention_ablation_experiment` (incl. `plot_single_comp_frac`), `plot_component_activations`, `geometric_interaction`, `sweep_summary`. Also `harvest/scripts/compare_intruder_scores.py`. Recent commits added the V/O combined plots and reworked the interactive QK datapoint scripts. |
| `feature/attn_plots` | Where the attention analysis was originally developed; mostly the same scripts as dev. Extras vs dev: `26a66c50 Update single_comp_frac plots`, `9afe3f38 Add single-component ablation plots`, `2c64b3f5 Add multi-pair fractional attention change plot`, `4725d369 Add fractional attention change plots normalized by cross-head mean baseline`, `dc87a8dc Add per-sample attention pattern plots with token labels`, `ad181bfa Add fixed source (r) sweep script for CE loss comparison`. The `single_comp_frac` script originated here and is now on dev. The `fixed source (r) sweep` script (likely the producer of `fig:adv-vs-no-adv` and `tab:vpd-pgd-ce`) appears to **not** be on dev — verify before reproducing. |
| `paper/editing-section` (origin) | The entire `spd/editing/` module + committed artifacts under `figures/editing/`. Required for `fig:model-editing-pareto` and `fig:model-editing-heatmap`. |
| `spd-paper`, `feature/paper-vis`, `feature/global-reverse-attn-transition` | No unique paper figure producers found beyond what's on dev. |

---

## 5. Things you cannot currently reproduce from a checked-in script

1. `fig:pareto-mse` and `fig:pareto-e2e` — BB's one-off WandB-pulling matplotlib script; only the rendered PNGs are in the blog repo.
2. `fig:feature_splitting` — same situation (BB's one-off).
3. `fig:splitting-heatmap` — analysis script lost; PNG is rendered by `build_split_heatmap.py` from a hardcoded array.
4. `tab:seed-mmcs` — LS's seed-MMCS analysis; not committed.
5. `fig:generations-showcase` data (`data/generation_comparisons.json`) — exporter not pinned to a checked-in path.
6. `data/attention/*.json` — `compute_data.py` / `compute_pair_data.py` on dev (commit `312b9cf3`) likely produces these; precise location is the QK-datapoint / `interactive_qk_contributions` package, not yet opened to confirm the JSON path mapping.

Everything else has a known producer (script + branch + input run id).

---

## 6. End-to-end reproduction recipe (sketch)

To rerun the full paper from `t-9d2b8f02` + `s-55ea3f9b`:

1. **Postprocess** the main VPD run: `param-decomp-postprocess <config.yaml>` (harvest → autointerp interpret + evals incl. intruder → dataset_attributions → graph_interp).
2. **Layer-1 attention figures**: run each `spd/scripts/plot_*` and `detect_prev_token_heads` against `wandb:goodfire/spd/runs/s-55ea3f9b`. Each writes to `spd/scripts/<name>/out/<run_id>/`.
3. **Interaction matrices**: `python spd/scripts/geometric_interaction/interaction_analysis.py --model_path=wandb:goodfire/spd/runs/s-55ea3f9b`.
4. **Intruder bar chart**: `python -m spd.harvest.scripts.compare_intruder_scores`, then `python vpd-blog-replit/scripts/build_intruder_figure.py`.
5. **Adversarial sweep**: run the `feature/attn_plots` "fixed source (r) sweep" script for `fig:adv-vs-no-adv` and `tab:vpd-pgd-ce`.
6. **Model editing**: switch to `paper/editing-section` and run `spd/editing/run_pareto_export.py` → `spd/editing/generate_pareto_plots.py` → `spd/editing/export_blog_heatmap.py`.
7. **Blog data exports**: on dev, `python -m spd.scripts.blog.{export_components,export_graphs,export_heatmap,export_manual_prompts}` (each with `--out-dir ../vpd-blog-replit/data/...`).
8. **Pareto / feature-splitting / seed-MMCS / generation-comparison JSON**: re-derive (no checked-in producer).
9. **Build the blog**: `vpd-blog-replit/scripts/build_static.py` (or `build_html.py`).
