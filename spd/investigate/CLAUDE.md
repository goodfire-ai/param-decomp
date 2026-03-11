# Investigation Module

Launch a Claude Code agent to investigate a specific research question about an SPD model decomposition.

## Usage

```bash
spd-investigate <wandb_path> "How does the model handle gendered pronouns?"
spd-investigate <wandb_path> "What circuit handles verb agreement?" --max_turns 30 --time 4:00:00
```

For parallel investigations, run the command multiple times with different prompts.

## Architecture

```
spd/investigate/
├── __init__.py           # Public exports
├── CLAUDE.md             # This file
├── schemas.py            # Pydantic models for outputs (BehaviorExplanation, InvestigationEvent)
├── agent_prompt.py       # System prompt template with model info injection
└── scripts/
    ├── __init__.py
    ├── run_slurm_cli.py  # CLI entry point (spd-investigate)
    ├── run_slurm.py      # SLURM submission logic
    └── run_agent.py      # Worker script (runs in SLURM job)
```

## How It Works

1. `spd-investigate` creates output dir, metadata, git snapshot, and submits a single SLURM job
2. The SLURM job runs `run_agent.py` which:
   - Starts an isolated FastAPI backend with MCP support
   - Loads the SPD run onto GPU
   - Fetches model architecture info
   - Generates the agent prompt (research question + model context + methodology)
   - Launches Claude Code with MCP tools
3. The agent investigates using MCP tools and writes findings to the output directory

## MCP Tools

The agent accesses all SPD functionality via MCP at `/mcp`:

**Circuit Discovery:**
- `optimize_graph` — Find minimal circuit for a behavior (streams progress)
- `create_prompt` — Tokenize text and get next-token probabilities

**Component Analysis:**
- `get_component_info` — Interpretation, token stats, correlations
- `probe_component` — Fast CI probing on custom text
- `get_component_activation_examples` — Training examples where a component fires
- `get_component_attributions` — Dataset-level component dependencies
- `get_attribution_strength` — Attribution between specific component pairs

**Testing:**
- `run_ablation` — Test circuit with only selected components
- `search_dataset` — Search training data

**Metadata:**
- `get_model_info` — Architecture details

**Output:**
- `update_research_log` — Append to research log (PRIMARY OUTPUT)
- `save_graph_artifact` — Save graph for inline visualization
- `save_explanation` — Save complete behavior explanation
- `set_investigation_summary` — Set title/summary for UI

## Output Structure

```
SPD_OUT_DIR/investigations/<inv_id>/
├── metadata.json          # Investigation config (wandb_path, prompt, etc.)
├── research_log.md        # Human-readable progress log (PRIMARY OUTPUT)
├── events.jsonl           # Structured progress events
├── explanations.jsonl     # Complete behavior explanations
├── summary.json           # Agent-provided title/summary for UI
├── artifacts/             # Graph artifacts for visualization
│   └── graph_001.json
├── app.db                 # Isolated SQLite database
├── backend.log            # Backend subprocess output
├── claude_output.jsonl    # Raw Claude Code output
├── agent_prompt.md        # The prompt given to the agent
└── mcp_config.json        # MCP server configuration
```

## Environment

The backend runs with `SPD_INVESTIGATION_DIR` set to the investigation directory. This controls:
- Database location: `<dir>/app.db`
- Events log: `<dir>/events.jsonl`
- Research log: `<dir>/research_log.md`

## Configuration

CLI arguments:
- `wandb_path` — Required. WandB run path for the SPD decomposition.
- `prompt` — Required. Research question or investigation directive.
- `--context_length` — Token context length (default: 128)
- `--max_turns` — Max Claude turns (default: 50, prevents runaway)
- `--partition` — SLURM partition (default: h200-reserved)
- `--time` — Job time limit (default: 8:00:00)
- `--job_suffix` — Optional suffix for job names

## Monitoring

```bash
# Watch research log
tail -f SPD_OUT_DIR/investigations/<inv_id>/research_log.md

# Watch events
tail -f SPD_OUT_DIR/investigations/<inv_id>/events.jsonl

# View explanations
cat SPD_OUT_DIR/investigations/<inv_id>/explanations.jsonl | jq .

# Check SLURM job status
squeue --me
```
