"""Names of the files a run directory holds, importable without JAX."""

LAUNCH_CONFIG_FILENAME = "launch_config.yaml"
"""The self-contained run config pinned into each run dir. Not `config.yaml` — that basename
clashes with wandb's own run-config file, which `wandb.save` would clobber via symlink."""
