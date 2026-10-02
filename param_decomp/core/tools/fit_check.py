"""AOT GPU-fit check: compile the run's REAL `jit_step` against a described GPU topology
from a CPU-only process and print the per-device memory verdict — the receipt a launch
reads BEFORE burning GPU nodes on an OOM.

The step is assembled exactly as the engine assembles it (`run.py`): the same
`build_optimizers` / algorithm-specific state initialization / `ForwardSubstrate.of`
composition, the same donation (state/batch/key donated, model not), the same
`compiler_options`. Inputs are `ShapeDtypeStruct`s carrying the run's declared shardings:
the model from its own `.shardings(rules)` tree, the train state from shape evaluation
of its ordinary placed initializer. Nothing executes — `.lower(...).compile()` on
compile-only topology devices is a deviceless XLA
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

import equinox as eqx
import jax
from beartype import beartype
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PRNGKeyArray, jaxtyped

from param_decomp.core.ci_fn.architecture import CIFnArchitecture
from param_decomp.core.components import site_stack_indices_for
from param_decomp.core.configs import NontargetConfig, PDConfig, TargetedPDConfig
from param_decomp.core.faithfulness import FaithfulnessLossFn, make_faithfulness_loss
from param_decomp.core.init_placed import seeded_ci_fn_initializer
from param_decomp.core.model import ComponentActivations, DecomposedModel, PlacedModel, PositionAxis
from param_decomp.core.optimizer import ScheduledOptimizer
from param_decomp.core.placement import PlacementRules
from param_decomp.core.pytree import ArrayTree, ShapeTree, ShardingTree
from param_decomp.core.run_state import (
    build_optimizers,
    init_decomposition,
    init_pd_state,
    init_targeted_pd_state,
)
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    PDState,
    PDTrainingState,
    TargetedPDState,
    TargetedPDTrainingState,
    TrainingProgress,
    TrainState,
    make_targeted_train_step,
    make_train_step,
)

GIB = 2**30


@dataclasses.dataclass(frozen=True)
class DumpConfig:
    """What a fit compile dumps under `xla_dump_to`. `hlo_protos` adds the
    after-optimizations `HloProto` (module + buffer assignment) — the input of both the
    memory ledger (`core.tools.memory_ledger`) and the profiler's memory viewer;
    `graph_html` adds XLA's HTML renders (see the LM fit_check CLI docstring for the
    size caveats)."""

    dir: Path
    hlo_protos: bool
    graph_html: bool

    def compiler_options(self) -> dict[str, bool | int | str]:
        options: dict[str, bool | int | str] = {"xla_dump_to": str(self.dir)}
        if self.hlo_protos:
            options["xla_dump_hlo_as_proto"] = True
        if self.graph_html:
            options["xla_dump_hlo_as_html"] = True
            options["xla_dump_fusion_visualization"] = True
        return options


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


@jaxtyped(typechecker=beartype)
def _abstract_like[TreeT](
    tree: ArrayTree[TreeT] | ShapeTree[TreeT], shardings: ShardingTree
) -> ShapeTree[TreeT]:
    """Describe the same array tree under its declared shardings without placing data."""
    return jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
        tree,
        shardings,
    )


@jaxtyped(typechecker=beartype)
def abstract_placed_model[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    model: DecomposedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    rules: PlacementRules,
) -> ShapeTree[PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT]]:
    """The placed-model bundle with abstract leaves on the rules' declared shardings —
    `place_target` without the placement (no data ever moves onto the described mesh)."""
    return PlacedModel(model=_abstract_like(model, model.shardings(rules)), placement=rules)


def standin_faithfulness_loss[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    abstract_placed: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
) -> FaithfulnessLossFn:
    """`faithfulness_loss_for` for a shapes-only model: the frozen norms are host floats
    the loss merely SCALES by, and an abstract model has no values to read, so
    shape-matched 1.0 stand-ins bind instead — the lowered/compiled step is the run's
    (the norms enter the jaxpr as scalar constants either way; no buffer moves). The
    bundle's placement census supplies the persist-stack pads, exactly as the runtime
    binding does."""
    abstract_model = abstract_placed.model
    norm_shapes = eqx.filter_eval_shape(lambda m: m.target_weight_sq_norms(), abstract_model)
    stack_pads = (
        {}
        if abstract_placed.placement is None
        else {
            group: entry.stack_pad
            for group, entry in abstract_placed.placement.components.group_census.items()
        }
    )
    return make_faithfulness_loss(
        site_stack_indices_for(abstract_model.sites),
        {group: (1.0,) * struct.shape[0] for group, struct in norm_shapes.items()},
        stack_pads,
    )


def _sharding_of(leaf: jax.ShapeDtypeStruct, mesh: Mesh) -> NamedSharding:
    """Re-anchor abstract sharding on the concrete mesh; unsharded leaves replicate."""
    sharding = leaf.sharding
    spec = sharding.spec if isinstance(sharding, NamedSharding) else P()
    return NamedSharding(mesh, spec)


@jaxtyped(typechecker=beartype)
def _on_mesh[TreeT](tree: ShapeTree[TreeT], mesh: Mesh) -> ShapeTree[TreeT]:
    """Bind shape evaluation's abstract mesh axes to the consumer's devices."""
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(
            leaf.shape, leaf.dtype, sharding=_sharding_of(leaf, mesh)
        ),
        tree,
    )


@jaxtyped(typechecker=beartype)
def declared_decomposition[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    ci_fn: CIFnArchitecture[Conditioning],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
) -> ShapeTree[Decomposition[Conditioning]]:
    """Describe the trained product on its declared shardings without allocating arrays."""
    rules = model.placement
    assert rules is not None, "fit check is a placed-run question"
    mesh = rules.mesh
    assert isinstance(mesh, Mesh), type(mesh)
    key = jax.eval_shape(lambda: random.PRNGKey(0))
    initializer = seeded_ci_fn_initializer(ci_fn, model.model.sites, rules)
    decomposition = jax.eval_shape(lambda m, k: init_decomposition(m, initializer, k), model, key)
    return _on_mesh(decomposition, mesh)


def argument_audit(args: object, pool_gib: float) -> None:
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


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class DeclaredRun[Conditioning, Training: TrainingProgress]:
    """A concrete algorithm's abstract state and the optimizers that shaped it."""

    state: ShapeTree[TrainState[Conditioning, Training]]
    opt_vu: ScheduledOptimizer
    opt_ci: ScheduledOptimizer


def declared_pd_run[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT](
    pd: PDConfig,
    ci_fn: CIFnArchitecture[Conditioning],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    positions: PositionAxis,
) -> DeclaredRun[Conditioning, PDTrainingState]:
    rules = model.placement
    assert rules is not None, "fit check is a placed-run question"
    mesh = rules.mesh
    assert isinstance(mesh, Mesh), type(mesh)
    opt_vu, opt_ci = build_optimizers(pd, rules, model.model.sites)
    initializer = seeded_ci_fn_initializer(ci_fn, model.model.sites, rules)

    def init(
        m: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        key: PRNGKeyArray,
    ) -> PDState[Conditioning]:
        return init_pd_state(pd, m, initializer, positions, opt_vu, opt_ci, key, key)

    key = jax.eval_shape(lambda: random.PRNGKey(0))
    state = _on_mesh(jax.eval_shape(init, model, key), mesh)
    return DeclaredRun(state=state, opt_vu=opt_vu, opt_ci=opt_ci)


def declared_targeted_run[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    pd: TargetedPDConfig,
    ci_fn: CIFnArchitecture[Conditioning],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    positions: PositionAxis,
    nontarget: NontargetConfig,
) -> DeclaredRun[Conditioning, TargetedPDTrainingState]:
    rules = model.placement
    assert rules is not None, "fit check is a placed-run question"
    mesh = rules.mesh
    assert isinstance(mesh, Mesh), type(mesh)
    opt_vu, opt_ci = build_optimizers(pd, rules, model.model.sites)
    initializer = seeded_ci_fn_initializer(ci_fn, model.model.sites, rules)

    def init(
        m: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
        key: PRNGKeyArray,
    ) -> TargetedPDState[Conditioning]:
        return init_targeted_pd_state(
            pd, m, initializer, positions, opt_vu, opt_ci, key, key, nontarget
        )

    key = jax.eval_shape(lambda: random.PRNGKey(0))
    state = _on_mesh(jax.eval_shape(init, model, key), mesh)
    return DeclaredRun(state=state, opt_vu=opt_vu, opt_ci=opt_ci)


def fit_report_of_compiled(compiled: jax.stages.Compiled, pool_gib: float) -> FitReport:
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


def lowered_train_step[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    pd: PDConfig,
    ci_fn: CIFnArchitecture[Conditioning],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    positions: PositionAxis,
    batch: TargetIn,
    faithfulness: FaithfulnessLossFn,
    *,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    compiler_options: dict[str, bool | int | str] | None,
) -> tuple[jax.stages.Lowered, DeclaredRun[Conditioning, PDTrainingState]]:
    """Assemble and LOWER the run's real jit_step at the declared placement — the trace
    gate's whole job (explicit-sharding refusals fire here, before any compile), and the
    fit check's front half. `faithfulness` arrives bound (`faithfulness_loss_for` reads
    frozen norms as host floats, which a shape-only model cannot serve — the trace gate
    binds shape-matched stand-ins instead). Returns the `jax.stages.Lowered` and the
    declared state."""
    rules = model.placement
    assert rules is not None, "fit check is a placed-run question"
    mesh = rules.mesh
    assert isinstance(mesh, Mesh), type(mesh)
    with jax.set_mesh(mesh):
        declared = declared_pd_run(pd, ci_fn, model, positions)
        state, opt_vu, opt_ci = declared.state, declared.opt_vu, declared.opt_ci

        substrate = ForwardSubstrate.of(
            model,
            remat_recon_forwards=remat_recon_forwards,
            remat_ci_fn=remat_ci_fn,
            ci_capture_keys=state.decomposition.ci_fn.capture_keys,
        )
        step_fn = make_train_step(
            model_static=model,
            substrate=substrate,
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=pd.steps,
            faithfulness=faithfulness,
        )

        step_key = jax.eval_shape(lambda: random.fold_in(random.PRNGKey(0), 0))
        print("lowering jit_step AOT ...", flush=True)
        outer = jax.jit(step_fn, donate_argnums=(1, 2, 3), compiler_options=compiler_options)
        return outer.lower(model, state, batch, step_key), declared


def lowered_targeted_train_step[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    pd: TargetedPDConfig,
    nontarget: NontargetConfig,
    ci_fn: CIFnArchitecture[Conditioning],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    positions: PositionAxis,
    target_batch: TargetIn,
    nontarget_batch: TargetIn,
    *,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    compiler_options: dict[str, bool | int | str] | None,
) -> tuple[jax.stages.Lowered, DeclaredRun[Conditioning, TargetedPDTrainingState]]:
    """`lowered_train_step`'s tPD twin: assemble and LOWER the two-stream
    targeted step at the declared placement, exactly as `run_targeted_decomposition_
    training` assembles it. `positions` is the TARGET stream's waist geometry —
    persistent sources live in the target pass."""
    rules = model.placement
    assert rules is not None, "fit check is a placed-run question"
    mesh = rules.mesh
    assert isinstance(mesh, Mesh), type(mesh)
    with jax.set_mesh(mesh):
        declared = declared_targeted_run(pd, ci_fn, model, positions, nontarget)
        state, opt_vu, opt_ci = declared.state, declared.opt_vu, declared.opt_ci

        substrate = ForwardSubstrate.of(
            model,
            remat_recon_forwards=remat_recon_forwards,
            remat_ci_fn=remat_ci_fn,
            ci_capture_keys=state.decomposition.ci_fn.capture_keys,
        )
        step_fn = make_targeted_train_step(
            model_static=model,
            substrate=substrate,
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=pd.steps,
        )

        step_key = jax.eval_shape(lambda: random.fold_in(random.PRNGKey(0), 0))
        print("lowering targeted jit_step AOT ...", flush=True)
        outer = jax.jit(step_fn, donate_argnums=(1, 2, 3, 4), compiler_options=compiler_options)
        return outer.lower(model, state, target_batch, nontarget_batch, step_key), declared


def aot_fit_check[TargetIn, Out, PreparedT: ComponentActivations, Conditioning, PreparedMaskingT](
    pd: PDConfig,
    ci_fn: CIFnArchitecture[Conditioning],
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    positions: PositionAxis,
    batch: TargetIn,
    *,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    compiler_options: dict[str, bool | int | str],
    pool_gib: float,
    dump: DumpConfig | None,
) -> FitReport:
    """Compile the run's jit_step AOT (model already abstract, on a compile-only or real
    mesh) and report per-device memory vs the stated pool."""
    options: dict[str, bool | int | str] = dict(compiler_options)
    if dump is not None:
        options.update(dump.compiler_options())
    lowered, declared = lowered_train_step(
        pd,
        ci_fn,
        model,
        positions,
        batch,
        standin_faithfulness_loss(model),
        remat_recon_forwards=remat_recon_forwards,
        remat_ci_fn=remat_ci_fn,
        compiler_options=options,
    )
    argument_audit((model, declared.state, batch), pool_gib)
    print("compiling jit_step AOT ...", flush=True)
    return fit_report_of_compiled(lowered.compile(), pool_gib)


def aot_targeted_fit_check[
    TargetIn,
    Out,
    PreparedT: ComponentActivations,
    Conditioning,
    PreparedMaskingT,
](
    pd: TargetedPDConfig,
    ci_fn: CIFnArchitecture[Conditioning],
    nontarget: NontargetConfig,
    model: PlacedModel[TargetIn, Out, PreparedT, Conditioning, PreparedMaskingT],
    positions: PositionAxis,
    target_batch: TargetIn,
    nontarget_batch: TargetIn,
    *,
    remat_recon_forwards: bool,
    remat_ci_fn: bool,
    compiler_options: dict[str, bool | int | str],
    pool_gib: float,
    dump: DumpConfig | None,
) -> FitReport:
    """`aot_fit_check`'s tPD twin: compile the two-stream targeted jit_step
    AOT and report per-device memory vs the stated pool. `positions` is the TARGET
    stream's waist geometry — the pool's own prompt length, so the receipt prices the
    persistent sources at their true extent."""
    options: dict[str, bool | int | str] = dict(compiler_options)
    if dump is not None:
        options.update(dump.compiler_options())
    lowered, declared = lowered_targeted_train_step(
        pd,
        nontarget,
        ci_fn,
        model,
        positions,
        target_batch,
        nontarget_batch,
        remat_recon_forwards=remat_recon_forwards,
        remat_ci_fn=remat_ci_fn,
        compiler_options=options,
    )
    argument_audit((model, declared.state, target_batch, nontarget_batch), pool_gib)
    print("compiling targeted jit_step AOT ...", flush=True)
    return fit_report_of_compiled(lowered.compile(), pool_gib)
