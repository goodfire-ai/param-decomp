# Repository guidance

Before planning, running, or interpreting a decomposition experiment, read both guides:

- [`docs/handbook.md`](docs/handbook.md) — the science, evidence standards, failure modes, and interpretation limits.
- [`docs/skill.md`](docs/skill.md) — the repository-specific recipe for targets, objectives, sweeps, validation, and analysis.

For other work, start with the smallest relevant source of truth:

- [`README.md`](README.md) — installation, runnable entry points, datasets, packaging, and development commands.
- [`CONFIGS.md`](CONFIGS.md) — which configurations belong in the repository and how they stay portable.
- The nearest module-level `CLAUDE.md` — local architecture and interfaces. These exist under `core`, `routed`, `experiments`, and `experiments/lm/pretrain`; `targets` carries a `README.md`.

## Repository-wide constraints

- `param_decomp/` must not know where it runs: no scheduler, submission, code-shipping, cluster path, mount, partition, or team namespace. Paths are explicit required inputs, and configs identify external resources by portable names. Deployment adapters may depend on the library; the library may never depend on them.
- Current training is JAX. The retired Torch implementation is only a semantic oracle at git tag `torch-oracle`; `nano_param_decomp/` is a standalone Torch reference and is not imported by either package.
- Keep the functional core pure and put I/O at entry points. Encode invariants in types when possible and assert the rest. Fail closed rather than adding fallbacks, compatibility shims, or degraded modes.
- Dispatch on a tag or kind is an exhaustive `match` with every arm written out — no `case _` catch-alls. Where basedpyright proves the match exhaustive, that proof is the fail-closed mechanism (a trailing `case _: raise` is dead code and is rejected); where it cannot, the last arm raises.
- Operations are locally truthful: each is correct given only what its input types promise, and the types carry those guarantees across seams. An operation whose correctness rests on an informal bound elsewhere in the program (a value "known" to be in range because something upstream clips it) is a defect — encode the bound in the type at the seam (a bounded type, a discriminated union) or make the operation correct for every value its input type admits.
- No code is written for a test's convenience. A test that needs a known state computes its expectation from real, in-domain inputs; production code never gains an out-of-range knob, debug flag, or special arm so a test can reach a state.
- Import public names from the modules that define them; package-level re-exports are exceptional. Update the nearest guide when changing a documented structure or interface.

## Development

Use the root uv project; `make install-dev` installs the library and development tools into one environment. Run commands with `uv run` or activate `.venv`.

Use the narrowest useful check while iterating:

```bash
make test       # testmon-selected tests, excluding slow
make check      # ruff format/lint and basedpyright
make test-all   # exhaustive suite
```

Never bypass pre-commit hooks. Add files explicitly rather than with `git add .`.
