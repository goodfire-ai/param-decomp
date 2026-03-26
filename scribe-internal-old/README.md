# Scribe

<p align="center">
  <img src="assets/scribe_logo.png" alt="Scribe Logo" width="200">
</p>

<p align="center">
  <strong>Goodfire's research agent for collaborative and autonomous experimentation</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.11+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/status-beta-yellow.svg" alt="Status">
</p>

---

## Quick Start

```bash
uv tool install git+https://github.com/goodfire-ai/scribe-internal
```

### Usage
`scribe [--provider] [CLI_ARGS]`
- Starts a CLI chat session with an AI agent. Behind the scenes, a server has been started and the agent has tools to run persistent Python sessions in an ipython kernel while automatically documenting all code execution in Jupyter notebooks.
- Creates a `notebooks` directory with the notebooks that log the agent's code.  
- Pass `claude` or `codex` or `gemini` as CLI providers
- Any arguments to the CLI are passed through (e.g. `scribe copilot -c` will pass the `-c` flag to Claude Code to continue the most recent chat)

## Details

### Start with command:
```bash
scribe [claude|gemini|codex]
```
- Launches you into a chat session with an agent
- Uses the specified CLI under the hood. The scribe command handles running Claude Code / Gemini CLI / Codex CLI with the proper notebook MCP tool already configured.

### Start a new session
```
You: Start a new session, let's do some image generation experiments

Claude: I'll start a new Scribe session for image generation.

Session started successfully! I've created a new notebook at notebooks/2025-01-09-10-30_ImageGeneration.ipynb. 
The kernel is ready for your image generation tasks.
```
- New scribe session is started
- Claude creates a notebook and starts a kernel

### Actions Scribe can take:
- Start session
- End session
- Execute code cells
- Add markdown cells
- Edit cells (automatically re-runs code)
- Delete cells
- Resume notebook (re-runs all cells)
- Start new session from an existing notebook (creates new notebook as a copy of the existing one, and re-runs all cells)

## Additional resources
- [Full blog post](https://www.goodfire.ai/blog/you-and-your-research-agent)
