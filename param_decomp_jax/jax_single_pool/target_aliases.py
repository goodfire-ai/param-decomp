"""Frozen-target / prefix unions shared by the two call sites that dispatch on
`cfg.target` (`run.py::main` and `load_run.py::build_target`)."""

from jax_single_pool.llama8b import Prefix, Target
from jax_single_pool.llama_simple_mlp import SimpleMLPPrefix, SimpleMLPTarget

AnyFrozenTarget = Target | SimpleMLPTarget
AnyPrefix = Prefix | SimpleMLPPrefix
