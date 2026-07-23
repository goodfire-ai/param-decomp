# param_decomp.targets

The vendored decomposition targets: **every `DecomposedModel` implementation lives
here**, one slice per architecture, between the generic engine (`param_decomp.core`)
and the composition layers above.

The dependency direction is `lab → targets → engine`, pinned by
`param_decomp/tests/test_runtime_standalone.py`. The engine never imports a target — it
sees only the `DecomposedModel` protocol (`param_decomp.core.model`) and the `ArchFamily`
grammar contract (`param_decomp.core.family`). The lab composes: the model-name → family
registry and all authoring vocabulary stay composition-side
(`param_decomp/experiments/lm/config.py`).

This layer exists so that *what a target is* and *what we distribute* are independent
decisions: a public release selects packages (engine + whichever slices are shareable);
internal-only slices simply aren't in the set. No slice is ever special to the engine.

## The slices

| Slice | Target |
|---|---|
| `glu_transformer` | Shared HF GLU-transformer machinery (site grammar, `FrozenAttn`/`GLULayer`, `GLUDecomposedModel`, the scan/masked-forward engine, HF safetensors loading) |
| `llama8b` | Llama-3.1-8B (vendored `LlamaConfig`, llama3 rope) — a `glu_transformer` family |
| `qwen3_8b` | Qwen3-8B-Base (`Qwen3FrozenAttn`: required QK-norm via the `_prep_qk` hook) — a `glu_transformer` family |
| `llama_simple_mlp` | The pile-pretrained `LlamaSimpleMLP` (loads from the `pretrain/` cache) |
| `transformer_taps` | The transformer families' activation-tap vocabulary (opaque strings to the engine) |
| `tms` | Toy: TMS (positionless, in-process pretrain) |
| `resid_mlp` | Toy: residual MLP (positionless, in-process pretrain) |

A slice owns everything about its architecture: the frozen modules, the decomposed
forward (including its sharding/remat strategy behind the protocol), its `ArchFamily`,
and its weight loading. `tests/` holds the per-target parity/golden suites; engine
behavior tests that merely use a target as a fixture stay with the engine.
`invariance_check.py` is the SPEC D4 device-count invariance harness (a tiny GLU target
driven through the engine at simulated device counts).
