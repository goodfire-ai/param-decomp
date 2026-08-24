"""AOT GPU-fit check: compile the run's REAL `jit_step` against a described GPU topology
from a CPU-only process and print the per-device memory verdict — the receipt a launch
reads BEFORE burning GPU nodes on an OOM.

The step is assembled exactly as the engine assembles it (`run.py`): the same
`build_optimizers` / `init_train_state` / `ForwardSubstrate.of` / `make_train_step`
composition, the same donation (state/batch/key donated, model not), the same
`compiler_options`. Inputs are `ShapeDtypeStruct`s carrying the run's declared shardings:
the model from its own `.shardings(rules)` tree, the train state from the AOT-compiled
init's OWN output shardings (the layout a checkpoint restore reproduces). Nothing
executes — `.lower(...).compile()` on compile-only topology devices is a deviceless XLA
compile, so this runs on any CPU box with the CUDA jaxlib installed.

Caveat carried in the output: a deviceless compile has no device to autotune against, so
fusion/algorithm choices (and therefore the arena) can differ from an attached compile by
the autotuner's picks. Buffer CLASSES and the big collective materializations — the
things fit verdicts hinge on — are layout facts, not autotune facts.

The per-domain entry that resolves a run YAML into these arguments is
`python -m param_decomp.experiments.lm.fit_check` (composition is lab-side; this module
is engine-side and takes built objects).
"""

import dataclasses
import math
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.adversary import PersistentAdversary, init_sources_adam_state
from param_decomp.core.built_run import BuiltRun
from param_decomp.core.ci_fn import resolve_ci_placement
from param_decomp.core.configs import PDConfig
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.init_placed import ci_fn_shardings, persistent_sources_shardings
from param_decomp.core.model import PlacedModel, PositionAxis
from param_decomp.core.objective import (
    build_objective,
    build_recon_terms,
)
from param_decomp.core.placement import component_stacks_shardings
from param_decomp.core.recon import persistent_configs
from param_decomp.core.run_state import build_optimizers, init_train_state
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
    TrainState,
    make_train_step,
)

GIB = 2**30

RUNTIME_WORKSPACE_MARGIN_GIB = 5.0
"""Non-XLA per-device HBM the arena number does not see (cuBLAS/cuDNN/NCCL workspaces,
CUDA context) — the perf docket's #f convention: subtract it from the pool before the
verdict, never hand-wave it after."""


@dataclasses.dataclass(frozen=True)
class FitReport:
    """`compiled.memory_analysis()` of the real jit_step, per device, plus the verdict."""

    argument_bytes: int
    output_bytes: int
    temp_bytes: int
    alias_bytes: int
    generated_code_bytes: int
    pool_gib: float

    @property
    def demanded_bytes(self) -> int:
        """Peak HBM the step demands: resident inputs + arena; donated inputs alias
        outputs, so outputs beyond `alias` are new allocations."""
        return self.argument_bytes + self.temp_bytes + max(0, self.output_bytes - self.alias_bytes)

    @property
    def effective_pool_gib(self) -> float:
        return self.pool_gib - RUNTIME_WORKSPACE_MARGIN_GIB

    @property
    def fits(self) -> bool:
        return self.demanded_bytes / GIB <= self.effective_pool_gib

    def render(self) -> str:
        lines = [
            f"arguments (params + state, resident): {self.argument_bytes / GIB:8.2f} GiB",
            f"outputs:                              {self.output_bytes / GIB:8.2f} GiB"
            f" (aliased via donation: {self.alias_bytes / GIB:.2f} GiB)",
            f"temp (XLA arena):                     {self.temp_bytes / GIB:8.2f} GiB",
            f"generated code:                       {self.generated_code_bytes / GIB:8.2f} GiB",
            f"DEMANDED per device:                  {self.demanded_bytes / GIB:8.2f} GiB",
            f"pool {self.pool_gib:.2f} GiB - {RUNTIME_WORKSPACE_MARGIN_GIB:.1f} GiB runtime"
            f" workspace margin (docket #f) = {self.effective_pool_gib:.2f} GiB",
            f"VERDICT: {'FITS' if self.fits else 'DOES NOT FIT'}"
            f" ({self.demanded_bytes / GIB:.2f} vs {self.effective_pool_gib:.2f} GiB)",
            "(deviceless compile: arena may shift with on-device autotuning)",
        ]
        return "\n".join(lines)


def _abstract_like(tree: Any, shardings: Any) -> Any:
    """`ShapeDtypeStruct` leaves carrying the given shardings; statics pass through
    (the `place_via_shardings` idiom, minus the placement)."""
    is_array = lambda x: hasattr(x, "shape") and hasattr(x, "dtype")  # noqa: E731
    return jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s) if is_array(a) else a,
        tree,
        shardings,
        is_leaf=lambda x: x is None,
    )


def abstract_placed_model(model: Any, rules: Any) -> PlacedModel:
    """The placed-model bundle with abstract leaves on the rules' declared shardings —
    `place_target` without the placement (no data ever moves onto the described mesh)."""
    return PlacedModel(model=_abstract_like(model, model.shardings(rules)), placement=rules)


def _sharding_of(leaf: Any, mesh: Mesh) -> NamedSharding:
    """A leaf's type-level sharding re-anchored on the concrete mesh; an untyped leaf
    (eagerly-created scalars: step counters, Adam counts) is replicated, exactly
    `run._ensure_global`'s treatment."""
    sharding = getattr(leaf, "sharding", None)
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
    return NamedSharding(mesh, spec)


def _mirrored(fn: Any, typed_arg: Any, mesh: Mesh) -> Any:
    """eval_shape `fn` over a declared-sharding-typed argument: derived state (optimizer
    moments are `zeros_like` mirrors) inherits the argument's type-level shardings, which
    is exactly how the eager runtime places it."""
    out = jax.eval_shape(fn, typed_arg)
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=_sharding_of(leaf, mesh)
        ),
        out,
    )


def _declared_state(
    state_struct: TrainState,
    pd: Any,
    model: PlacedModel,
    positions: PositionAxis,
    mesh: Mesh,
    rules: Any,
    opt_vu: Any,
    opt_ci: Any,
) -> TrainState:
    """The abstract `TrainState` typed with the shardings the RUNTIME state actually
    carries — assembled from the same declared sources the eager init places onto
    (`component_stacks_shardings`, `ci_fn_shardings`, `persistent_sources_shardings`),
    with optimizer moments mirroring their parameters and scalars replicated.

    An AOT compile of the init function is NOT a valid source: `out_shardings` on the
    seeded-init jits is a placement directive, not a type constraint, so both
    `eval_shape` avals and a compiled init's `output_shardings` come back
    compiler-chosen — the first fit-check run compiled the whole 1.8 TiB state
    replicated per device that way."""
    components_typed = _abstract_like(
        state_struct.decomposition.components,
        component_stacks_shardings(state_struct.decomposition.components, rules),
    )
    ci_fn_typed = _abstract_like(
        state_struct.decomposition.ci_fn,
        ci_fn_shardings(state_struct.decomposition.ci_fn, mesh, rules),
    )
    persistent = persistent_configs(build_recon_terms(pd.loss_metrics, model.site_names))
    adversaries: dict[str, PersistentAdversary] = {}
    for state_key, adv in state_struct.training.adversaries.items():
        sources_typed = _abstract_like(
            adv.sources,
            persistent_sources_shardings(
                model.site_names,
                positions,
                persistent[state_key].source_shape,
                pd.batch_size,
                mesh,
            ),
        )
        adversaries[state_key] = PersistentAdversary(
            sources=sources_typed,
            opt_state=_mirrored(init_sources_adam_state, sources_typed, mesh),
            state_key=adv.state_key,
            adam=adv.adam,
            n_warmup=adv.n_warmup,
        )
    return TrainState(
        decomposition=Decomposition(components=components_typed, ci_fn=ci_fn_typed),
        training=TrainingItem(
            components_opt_state=_mirrored(
                lambda c: opt_vu.init(eqx.filter(c, eqx.is_array)), components_typed, mesh
            ),
            ci_fn_opt_state=_mirrored(
                lambda c: opt_ci.init(eqx.filter(c, eqx.is_array)), ci_fn_typed, mesh
            ),
            adversaries=adversaries,
            # The tree deliberately carries abstract leaves in Array positions — it only
            # ever feeds `.lower(...)`.
            freq_ema=None,
            step=cast(Any, jax.ShapeDtypeStruct((), jnp.int32, sharding=NamedSharding(mesh, P()))),
        ),
    )


def argument_audit(args: Any, pool_gib: float) -> None:
    """Per-device resident bytes of every step argument, largest first, printed BEFORE the
    compile spends minutes — then a fail-closed gate: a resident-argument total at
    multiples of the pool means the entry shardings are broken (an accidentally
    replicated state), and no verdict may be emitted from such a compile."""
    world: int | None = None
    rows: list[tuple[int, str, str]] = []
    total = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(args)[0]:
        sharding = leaf.sharding
        assert isinstance(sharding, NamedSharding), (path, sharding)
        leaf_mesh = sharding.mesh
        assert isinstance(leaf_mesh, Mesh), (path, type(leaf_mesh))
        if world is None:
            world = leaf_mesh.devices.size
        assert leaf_mesh.devices.size == world, (path, leaf_mesh.devices.size, world)
        shard_bytes = math.prod(sharding.shard_shape(leaf.shape)) * leaf.dtype.itemsize
        total += shard_bytes
        rows.append((shard_bytes, jax.tree_util.keystr(path), str(sharding.spec)))
    rows.sort(reverse=True)
    print(f"resident step arguments: {total / GIB:.2f} GiB/device on {world} devices; largest:")
    for shard_bytes, path, spec in rows[:8]:
        print(f"  {shard_bytes / GIB:7.2f} GiB  {path}  {spec}")
    assert total / GIB < 4 * pool_gib, (
        f"resident arguments {total / GIB:.1f} GiB/device is unsharded-state scale "
        f"(pool {pool_gib}): the entry shardings are broken — refusing to compile a verdict"
    )


@dataclasses.dataclass(frozen=True)
class DeclaredRun:
    """The abstract `TrainState` typed with the run's DECLARED shardings, plus the
    optimizers and resolved CI placement that shaped it — the shared entry for every AOT
    fit compile (the train step here, the scalar eval steps in the LM composition)."""

    state: TrainState
    opt_vu: Any
    opt_ci: Any
    ci_placement: Any


def declared_run(
    built: BuiltRun[Any, Any, Any], model: PlacedModel, positions: PositionAxis
) -> DeclaredRun:
    pd = built.pd
    rules = model.placement
    assert rules is not None, "fit check is a placed-run question"
    mesh = rules.mesh
    assert isinstance(mesh, Mesh), type(mesh)

    ci_placement = resolve_ci_placement(built.ci_fn, rules)
    opt_vu, opt_ci, _ = build_optimizers(pd, built.ci_fn, mesh, rules, ci_placement)

    def init(m: PlacedModel, init_key: Any, src_key: Any):
        return init_train_state(
            pd,
            m,
            built.ci_fn,
            positions,
            opt_vu,
            opt_ci,
            init_key,
            src_key,
        )

    key_struct = jax.eval_shape(lambda: random.PRNGKey(0))
    print("assembling the state's declared shardings ...", flush=True)
    state_struct = jax.eval_shape(init, model, key_struct, key_struct)
    state = _declared_state(state_struct, pd, model, positions, mesh, rules, opt_vu, opt_ci)
    return DeclaredRun(state=state, opt_vu=opt_vu, opt_ci=opt_ci, ci_placement=ci_placement)


def fit_report_of_compiled(compiled: Any, pool_gib: float) -> FitReport:
    mem = compiled.memory_analysis()
    assert mem is not None, "compiled executable reported no memory analysis"
    return FitReport(
        argument_bytes=mem.argument_size_in_bytes,
        output_bytes=mem.output_size_in_bytes,
        temp_bytes=mem.temp_size_in_bytes,
        alias_bytes=mem.alias_size_in_bytes,
        generated_code_bytes=mem.generated_code_size_in_bytes,
        pool_gib=pool_gib,
    )


def aot_fit_check(
    built: BuiltRun[Any, Any, Any],
    model: PlacedModel,
    positions: PositionAxis,
    batch: jax.ShapeDtypeStruct,
    *,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    compiler_options: dict[str, bool | int | str],
    pool_gib: float,
    dump_dir: Path | None,
) -> FitReport:
    """Compile the run's jit_step AOT (model already abstract, on a compile-only or real
    mesh) and report per-device memory vs the stated pool."""
    pd = built.pd
    rules = model.placement
    assert rules is not None, "fit check is a placed-run question"
    mesh = rules.mesh
    assert isinstance(mesh, Mesh), type(mesh)
    jax.set_mesh(mesh)

    assert isinstance(pd, PDConfig), f"fit check is a plain-VPD question, got {type(pd)}"
    objective = build_objective(pd.loss_metrics, model.site_names)
    faithfulness = faithfulness_loss_for(model.model)

    declared = declared_run(built, model, positions)
    state, opt_vu, opt_ci = declared.state, declared.opt_vu, declared.opt_ci
    argument_audit((model, state, batch), pool_gib)

    substrate = ForwardSubstrate.of(
        model,
        remat_recon_forwards=remat_recon_forwards,
        remat_ci_fn=remat_ci_fn,
        ci_capture_keys=state.decomposition.ci_fn.capture_keys,
        ci_placement=declared.ci_placement,
    )
    # `compiler_options=None` here, NOT the run's: jax refuses options on a nested jit,
    # and the engine's own filter_jit becomes nested under the outer AOT jit below, which
    # restates the run's options at the top level where the compile actually reads them.
    step_fn = make_train_step(
        model_static=model,
        substrate=substrate,
        objective=objective,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=pd.steps,
        faithfulness=faithfulness,
        compiler_options=None,
    )

    options: dict[str, bool | int | str] = dict(compiler_options)
    if dump_dir is not None:
        options["xla_dump_to"] = str(dump_dir)

    # The engine's donation exactly (`filter_jit(step, donate="all-except-first")`): the
    # inner eqx jit inlines under this outer trace, so donation and compiler options must
    # be restated here to reach the compile.
    def step(m: PlacedModel, state_arg: TrainState, batch_arg: Any, key_arg: Any):
        return step_fn(m, state_arg, batch_arg, key_arg)

    outer = jax.jit(step, donate_argnums=(1, 2, 3), compiler_options=options)
    step_key = jax.eval_shape(lambda: random.fold_in(random.PRNGKey(0), 0))
    print("compiling jit_step AOT ...", flush=True)
    compiled = outer.lower(model, state, batch, step_key).compile()
    return fit_report_of_compiled(compiled, pool_gib)
