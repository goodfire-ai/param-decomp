"""Canonical key set for the `pd-jax-lm` wrapper yaml — the single source of truth
shared by the lab-side launcher (`jax_launch._validate_wrapper`, torch venv) and the
runtime loader (`jax_single_pool.config.load_wrapper`, jax venv).

Lives here, in the torch-free config package, because it's the only distribution both
venvs install; neither side can import the other's package. See the wrapper schema in
`jax_single_pool.config`'s module docstring.

`run_id`, `wandb_group`, and `wandb_tags` are minted from the launch (`pd-jax-lm`
mints the id; `--group`/`--tags` supply the rest) and appended to the workspace copy
at submit time — so a hand-authored wrapper carries `WRAPPER_KEYS_BEFORE_SUBMIT` and
the stamped copy the loader reads carries the required `WRAPPER_KEYS` plus whichever
`WRAPPER_OPTIONAL_KEYS` the launch supplied.
"""

RUN_ID_KEY = "run_id"

WRAPPER_KEYS = frozenset(
    {"torch_config", RUN_ID_KEY, "run_name", "out_dir", "remat_recon_forwards"}
)

WRAPPER_OPTIONAL_KEYS = frozenset({"wandb_group", "wandb_tags"})

SUBMIT_MINTED_KEYS = frozenset({RUN_ID_KEY}) | WRAPPER_OPTIONAL_KEYS

WRAPPER_KEYS_BEFORE_SUBMIT = WRAPPER_KEYS - SUBMIT_MINTED_KEYS
