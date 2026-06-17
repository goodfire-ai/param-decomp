"""Frozen-target / prefix unions shared by the two call sites that dispatch on
`cfg.target` (`run.py::main` and `load_run.py::build_target`)."""

from jax_single_pool.llama8b import Prefix, Target
from jax_single_pool.llama_simple_mlp import SimpleMLPPrefix, SimpleMLPTarget
from jax_single_pool.resid_mlp import ResidMLPTarget
from jax_single_pool.tms import TMSTarget

AnyFrozenTarget = Target | SimpleMLPTarget | TMSTarget | ResidMLPTarget
AnyPrefix = Prefix | SimpleMLPPrefix | None
"""TMS has no prefix (the whole model is decomposed); its `prefix` slot is `None`. ResidMLP
carries its `W_E` prefix inside the frozen target itself, so it also uses no `prefix` slot
(`train_resid_mlp` embeds via `resid_mlp.resid_mlp_input_residual`)."""
