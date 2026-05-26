# Autointerp Module

LLM-based automated interpretation of PD components. Consumes pre-harvested data from `param_decomp_lab/harvest/` (see `param_decomp_lab/harvest/CLAUDE.md`).

## Usage

Autointerp requires harvest data to already exist (see [`../harvest/CLAUDE.md`](../harvest/CLAUDE.md));
every CLI takes a `--harvest_subrun_id`.

```bash
# Via SLURM
pd-autointerp <decomposition_id> --config path/to/autointerp_slurm_config.yaml --harvest_subrun_id h-YYYYMMDD_HHMMSS

# Direct execution (one process, inline JSON config)
python -m param_decomp_lab.autointerp.scripts.run_interpret <decomposition_id> \
    --config_json '{...AutointerpConfig...}' \
    --harvest_subrun_id h-YYYYMMDD_HHMMSS
```

`<decomposition_id>` is the decomposition's identifier — for PD runs, the wandb path
like `entity/project/runs/<run_id>`.

Interpretation settings (model, reasoning effort, forbidden words, prompt variant, …)
live inside the YAML / JSON `AutointerpConfig` — they are *not* CLI flags. The config
is a discriminated union over strategies (`compact_skeptical`, `dual_view`,
`rich_examples`, `canon`); see the Config section below.

Requires the API key for your chosen provider (e.g. `OPENROUTER_API_KEY`, or `GEMINI_API_KEY` when `llm.type` is `google_ai` — key from [Google AI Studio](https://aistudio.google.com/app/apikey)).

## Data Storage

Each autointerp subrun has its own SQLite database:

```
PARAM_DECOMP_OUT_DIR/autointerp/<spd_run_id>/
└── <autointerp_run_id>/           # e.g. a-20260206_153040
    ├── interp.db                  # SQLite DB: interpretations + scores (WAL mode)
    └── config.yaml                # AutointerpConfig (for reproducibility)
```

`InterpRepo` reads from the latest subrun (by lexicographic sort of `a-*` dir names).

The `interp.db` schema has three tables:
- `interpretations`: component_key -> label, reasoning, raw_response, prompt
- `scores`: (component_key, score_type) -> score, details (JSON blob with trial data)
- `config`: key-value store

Score types: `detection`, `fuzzing`.

**Note on intruder scores**: Intruder evaluation lives in `param_decomp_lab/harvest/` (not here) because it tests decomposition quality, not label quality. Intruder scores are stored in `harvest.db`. Detection and fuzzing evaluate interpretation labels and belong here.

## Architecture

### Config (`config.py`)

`AutointerpConfig` is a discriminated union over interpretation strategy configs. Each variant specifies everything that affects interpretation output (model, prompt params, reasoning effort). Admin/execution params (cost limits, parallelism) are NOT part of the config.

Current strategies:
- `CompactSkepticalConfig` — compact prompt, skeptical tone, structured JSON output
- `DualViewConfig` — dual-view prompt (separate input/output token framings)
- `RichExamplesConfig` — richer per-example formatting
- `CanonConfig` — canonical baseline prompt

Also contains `DetectionEvalConfig` / `FuzzingEvalConfig` for eval jobs.

### Strategies (`strategies/`)

Each strategy config type has a corresponding prompt implementation:
- `strategies/compact_skeptical.py`, `dual_view.py`, `rich_examples.py`, `canon.py`
- `strategies/dispatch.py` — routes `AutointerpConfig` → strategy implementation via `match`

### Database (`db.py`)

`InterpDB` class wrapping SQLite for interpretations and scores. Uses WAL mode for concurrent reads. Serialization via `orjson`.

### Repository (`repo.py`)

`InterpRepo` provides read/write access to autointerp data for a run. Lazily opens the SQLite database on first access. Used by the app backend.

### Interpret (`interpret.py`)

- Uses OpenRouter, Anthropic, OpenAI, or Google AI (Gemini) with structured JSON outputs (`LLMConfig` in `providers.py`)
- Maximum parallelism with exponential backoff on rate limits
- Resume support: Skips already-completed components via `db.get_completed_keys()`
- Progress logging via `param_decomp.log.logger`
- `interpret_component()` interprets a single component
- `run_interpret()` orchestrates batch interpretation with resume support

## Key Types (`schemas.py`)

```python
InterpretationResult  # component_key + label + reasoning + raw_response + prompt
ArchitectureInfo      # Model architecture context for prompts
```
