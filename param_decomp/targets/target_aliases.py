"""Decomposed-model union for the LM targets `run.py::main` and
`load_run.py::build_target` dispatch over.

The toy targets (TMS, ResidMLP) live in the lab and are NOT members of this union — the
generic engine (`run_decomposition_training`) takes the model as `DecomposedModel`, so the
core never names a toy."""

from param_decomp.targets.llama8b import LlamaDecomposedModel
from param_decomp.targets.llama_simple_mlp import SimpleMLPDecomposedModel

AnyDecomposedModel = LlamaDecomposedModel | SimpleMLPDecomposedModel
