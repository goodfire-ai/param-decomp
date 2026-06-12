# param-decomp-jax

The JAX single-pool VPD trainer distribution. Everything of substance lives in
[`jax_single_pool/`](jax_single_pool/README.md) (trainer, targets, configs, tests,
SPEC); `vendored_jax/` is the bit-parity JAX Llama it builds on.

Not a member of the root uv workspace — it keeps its own venvs so the JAX and torch
CUDA stacks never share an environment (their pinned `nvidia-*` wheels conflict):

- `.venv` — CPU dev env (tests, typecheck): `make install-jax` from the repo root.
- `.venv-cuda` — GPU runtime env: `make install-jax-cuda`. This is also what
  `pd-jax-lm` builds inside each launch workspace.

The shared config schema (`param-decomp-config`) installs editably from the sibling
in-tree package in both.

Launches go through `pd-jax-lm` (torch venv, repo root) — see
`jax_single_pool/CLAUDE.md`.
