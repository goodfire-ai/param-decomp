# paper_vis — VPD Paper Visualization Pipeline

Generates static HTML dashboards comparing VPD parameter components with transcoder latents, intended for the VPD paper and blog post.

## Status (2026-03-13)

### Done
- **Transcoder harvest integration**: `TranscoderAdapter` + `TranscoderHarvestFn` working end-to-end via the existing `spd-harvest` CLI
- **Deterministic transcoder IDs**: `TranscoderHarvestConfig.id` uses sha256 (was Python `hash()` which varies per process)
- **EncoderConfig cleaned up**: all defaults removed, values come from checkpoint `config.json`
- **Small harvest completed**: `tc-3f297233` with 200 batches — enough for prototyping
- **Jose (s-55ea3f9b) harvest**: exists from prior work (`h-20260227_010249`, 20K batches)
- **Jose autointerp**: 38,474 interpretations via `dual_view` + `gemini-3.1-pro-preview` (no scoring yet)
- **Dashboard**: side-by-side carousel (VPD left, TC right), incremental per-component JSON loading, Goodfire brand fonts (Suisse Works/Intl, IBM Plex Mono)
- **Research post**: goodfire.ai-styled article with paper content, KaTeX equations, inlined dashboard (no iframe)
- **Human-readable layer names**: `human_layer_desc()` used in dashboard via `ModelMetadata.layer_descriptions`
- **`"transcoder"` added to `DecompositionMethod`** with description for autointerp prompts

### Not yet done
- **Transcoder autointerp**: config ready at `/tmp/tc_autointerp.yaml` (50 components, dual_view, gemini-3.1-pro). Needs harvest subrun `h-20260313_135359`. Command:
  ```
  spd-autointerp tc-3f297233 --config /tmp/tc_autointerp.yaml --harvest_subrun_id h-20260313_135359
  ```
- **Jose scoring**: detection + fuzzing evals not yet run
- **Larger TC harvest**: 200 batches is fine for prototyping but production needs ~20K batches (~2 hours on 4 GPUs)
- **Dashboard polish**: center-on-highlighted-token in examples (TODO in code), more components

## Architecture

```
spd/paper_vis/
├── data.py              # Pydantic types: ComponentDashboardData, DecompositionData, etc.
├── generate.py          # Reads harvest DB + autointerp DB → per-component JSON files
├── build_dashboard.py   # Assembles dashboard HTML + research post with inlined dashboard
├── dashboard.html       # Template: carousel with per-component fetch
└── research_post.html   # Template: goodfire.ai-styled article, dashboard inlined at build time
```

### Data flow
1. `build_dashboard.py` calls `generate.py` for each decomposition (VPD, TC)
2. `generate.py` reads from `HarvestRepo` + `InterpRepo` (no model loading needed)
3. Outputs per-component JSON files + a manifest with lightweight component index
4. `build_dashboard.py` injects manifest JSON into `dashboard.html` template → `index.html`
5. Also extracts dashboard CSS/JS/markup and inlines into `research_post.html` → `research_post.html`

### Output structure
```
<out_dir>/
├── index.html              # Standalone dashboard
├── research_post.html      # Full article with inlined dashboard
├── vpd/
│   └── components/         # Per-component JSON (fetched on navigate)
└── tc/
    └── components/
```

### Rebuild command
```bash
python -m spd.paper_vis.build_dashboard \
    --vpd_id s-55ea3f9b --tc_id tc-3f297233 --limit 20 \
    --out_dir /mnt/polished-lake/artifacts/mechanisms/spd/www/paper_dashboard
```

### Live URLs
- Dashboard: `http://goodfre-login:8081/spd/paper_dashboard/`
- Research post: `http://goodfre-login:8081/spd/paper_dashboard/research_post.html`

## Key IDs
| What | ID | Notes |
|------|----|-------|
| Jose (VPD) | `s-55ea3f9b` | pile_llama_simple_mlp-4L, base model `t-9d2b8f02` |
| Transcoder | `tc-3f297233` | BatchTopK k=32, 4096 dict, base model `t-32d1bb3b` |
| TC artifacts | `mats-sprint/pile_transcoder_sweep3/...` | 4 layers, from Bart Bussmann |
| Jose harvest | `h-20260227_010249` | 20K batches |
| TC harvest | `h-20260313_135359` | 200 batches (small, for prototyping) |
| Jose autointerp | `a-20260303_134527` | 38K labels, dual_view, no scoring |

## Base model mismatch note
Jose decomposes `t-9d2b8f02`, transcoders are trained on `t-32d1bb3b`. Both are `pile_llama_simple_mlp-4L` (same architecture, same tokenizer, same data distribution) but different shuffle seeds. Fine for the comparison.
