"""Canonical key set for the `pd-jax-lm` wrapper yaml — the single source of truth
shared by the lab-side launcher (`jax_launch._validate_wrapper`, torch venv) and the
runtime loader (`jax_single_pool.config.load_wrapper`, jax venv).

Lives here, in the torch-free config package, because it's the only distribution both
venvs install; neither side can import the other's package. See the wrapper schema in
`jax_single_pool.config`'s module docstring.

Three submit-time key groups:
- `run_id` is always minted by `pd-jax-lm` and must be absent in a hand-authored wrapper.
- `wandb_group` / `wandb_tags` come from `--group` / `--tags` and are appended only when
  supplied; a hand-authored wrapper never carries them.
- `out_dir` is author-overridable: a wrapper MAY set it (e.g. the llama8b wrappers'
  `jax_runs`), and the launcher mints `PARAM_DECOMP_OUT_DIR/runs` when it is absent.

So a hand-authored wrapper carries `WRAPPER_KEYS_BEFORE_SUBMIT` (the required keys plus
an optional `out_dir`), and the stamped copy the loader reads carries the full required
`WRAPPER_KEYS` plus whichever `WRAPPER_OPTIONAL_KEYS` the launch supplied.
"""

RUN_ID_KEY = "run_id"
OUT_DIR_KEY = "out_dir"

WRAPPER_KEYS = frozenset(
    {"torch_config", RUN_ID_KEY, "run_name", OUT_DIR_KEY, "remat_recon_forwards"}
)

WRAPPER_OPTIONAL_KEYS = frozenset({"wandb_group", "wandb_tags"})

# Minted at submit and never present in a hand-authored wrapper.
SUBMIT_MINTED_KEYS = frozenset({RUN_ID_KEY}) | WRAPPER_OPTIONAL_KEYS

# Minted at submit when absent, but a hand-authored wrapper MAY supply it to override.
AUTHOR_OVERRIDABLE_KEYS = frozenset({OUT_DIR_KEY})

WRAPPER_REQUIRED_BEFORE_SUBMIT = WRAPPER_KEYS - SUBMIT_MINTED_KEYS - AUTHOR_OVERRIDABLE_KEYS

WRAPPER_KEYS_BEFORE_SUBMIT = WRAPPER_REQUIRED_BEFORE_SUBMIT | AUTHOR_OVERRIDABLE_KEYS
