"""Offline distillation probe: can small factored CI fns REPRESENT a converged chunkwise
CI fn's labeling?

Opens a finished run, freezes its CI fn as the TEACHER, and trains small `FactoredCIFn`
STUDENTS (gate-only + rank-r context variants) to regress the teacher's CI logits over
fresh corpus batches — MSE on a leaky-clipped view of the logits, so capacity goes to the
decision-relevant [0,1] region rather than matching a -50 with a -30. Reports how closely
each student reproduces the teacher's lower/upper squashings and alive set.

    python -m param_decomp_lab.experiments.lm.distill_ci_probe --run_dir runs/<p-id> \
        [--steps 4000] [--batch_size 32] [--seed 0] [--out <path.jsonl>]

Single-process, single-device-friendly: no mesh/sharding beyond what `open_jax_run` does.
"""

import json
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import fire
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray

from param_decomp.built_run import DataConfig
from param_decomp.ci_fn import (
    CIFn,
    FactoredCIArch,
    FactoredCIFn,
    FactoredCtxArch,
    build_ci_fn,
    lower_leaky_hard_sigmoid,
    upper_leaky_hard_sigmoid,
)
from param_decomp.components import DecompVU
from param_decomp.data import BatchSchedule, ShardServer, scan_shards
from param_decomp.lm import DecomposedModel
from param_decomp.train import COMPUTE_DT, cast_floating
from param_decomp_lab.experiments.lm.load_run import open_jax_run

LOG_EVERY = 200
SOFT_LEAK = 0.05
ALIVE_THRESHOLDS = (1e-6, 0.01, 0.1)
"""1e-6 = the L0 counting threshold; 0.1 = the paper's rounding threshold (sub-0.1 CI
carries essentially no performance — the mask-relevant boundary)."""
LR = 1e-3
WARMUP_STEPS = 500
STUDENT_LR = {"d2048-r512": 5e-4}
"""Per-student peak-LR overrides: the d2048 trunk diverges at the shared 1e-3 without
warmup; with warmup 5e-4 holds."""


def _student_archs(
    ctx_taps: tuple[str, ...], input_dim: int, n_sites: int
) -> dict[str, FactoredCIArch]:
    def ctx(
        d_model: int,
        n_blocks: int,
        n_heads: int,
        mlp_hidden: int,
        rank: int,
        summary_k: int | None = None,
        modulate_slopes: bool = False,
    ) -> FactoredCtxArch:
        summary_dim = n_sites * summary_k if summary_k is not None else 0
        return FactoredCtxArch(
            taps=ctx_taps,
            input_dim=input_dim + summary_dim,
            d_model=d_model,
            n_blocks=n_blocks,
            n_heads=n_heads,
            mlp_hidden=mlp_hidden,
            rank=rank,
            summary_k=summary_k,
            modulate_slopes=modulate_slopes,
        )

    return {
        "gate-only": FactoredCIArch(ctx=None),
        "r32-d256x2": FactoredCIArch(ctx=ctx(256, 2, 4, 1024, 32)),
        "d1024-r128": FactoredCIArch(ctx=ctx(1024, 4, 8, 4096, 128)),
        "d1024-full": FactoredCIArch(
            ctx=ctx(1024, 4, 8, 4096, 128, summary_k=16, modulate_slopes=True)
        ),
        "d2048-r512": FactoredCIArch(ctx=ctx(2048, 4, 16, 8192, 512)),
    }


def _soft(x: Array) -> Array:
    """Leaky-clipped view of CI logits: preserves the decision-relevant [0,1] region while
    keeping a monotone, everywhere-differentiable gradient toward far-out-of-range logits."""
    return jnp.clip(x, 0.0, 1.0) + SOFT_LEAK * x


def _distill_loss(
    student: CIFn,
    taps: dict[str, Array],
    teacher_logits: dict[str, Array],
    vu: DecompVU,
) -> Float[Array, ""]:
    """Per-site MSE between soft-clipped student and teacher logits, summed over sites.
    Student forward in bf16 (cast inside, like the trainer), loss arithmetic in fp32."""
    ci = cast_floating(student, COMPUTE_DT)(taps, vu=vu, remat=False)
    per_site = [
        jnp.mean((_soft(ci.logits[site].astype(jnp.float32)) - _soft(t.astype(jnp.float32))) ** 2)
        for site, t in teacher_logits.items()
    ]
    return jnp.sum(jnp.stack(per_site))


@eqx.filter_jit
def _read_taps(
    model: DecomposedModel, tokens: Int[Array, "B T"], names: tuple[str, ...]
) -> dict[str, Array]:
    return model.read_activations(tokens, names)


@eqx.filter_jit
def _read_and_label(
    model: DecomposedModel,
    teacher: CIFn,
    tokens: Int[Array, "B T"],
    merged_names: tuple[str, ...],
) -> tuple[dict[str, Array], dict[str, Array]]:
    """One target forward serving every consumer's taps + the teacher's CI logits (labels).
    `model` is a traced arg, never closed over (the HLO-baking rule)."""
    taps = model.read_activations(tokens, merged_names)
    labels = cast_floating(teacher, COMPUTE_DT)(taps, remat=False).logits
    return taps, jax.lax.stop_gradient(labels)


UpdateFn = Callable[
    [CIFn, optax.OptState, dict[str, Array], dict[str, Array], DecompVU],
    tuple[CIFn, optax.OptState, Float[Array, ""]],
]


def _make_update(optimizer: optax.GradientTransformation) -> UpdateFn:
    @eqx.filter_jit
    def update(
        student: CIFn,
        opt_state: optax.OptState,
        taps: dict[str, Array],
        teacher_logits: dict[str, Array],
        vu: DecompVU,
    ) -> tuple[CIFn, optax.OptState, Float[Array, ""]]:
        loss, grads = eqx.filter_value_and_grad(_distill_loss)(student, taps, teacher_logits, vu)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(student, updates), opt_state, loss

    return update


@eqx.filter_jit
def _metrics(
    student: CIFn,
    taps: dict[str, Array],
    teacher_logits: dict[str, Array],
    vu: DecompVU,
) -> dict[str, Float[Array, ""]]:
    """Real-squashing agreement on the current batch, pooled over every (site, position,
    component) element, in fp32 (bf16 student forward — the dtype it would be consumed at)."""
    ci = cast_floating(student, COMPUTE_DT)(taps, vu=vu, remat=False)
    one = jnp.asarray(1.0, jnp.float32)
    sq_lower = jnp.zeros((), jnp.float32)
    abs_lower = jnp.zeros((), jnp.float32)
    sq_upper = jnp.zeros((), jnp.float32)
    w_hits = jnp.zeros((), jnp.float32)  # teacher-lower mass the student also marks alive
    w_total = jnp.zeros((), jnp.float32)
    by_thr = {
        thr: {k: jnp.zeros((), jnp.float32) for k in ("hits", "t_n", "s_n", "sq")}
        for thr in ALIVE_THRESHOLDS
    }
    n = 0
    for site, t_logits in teacher_logits.items():
        t = t_logits.astype(jnp.float32)
        s = ci.logits[site].astype(jnp.float32)
        t_lower, s_lower = lower_leaky_hard_sigmoid(t), lower_leaky_hard_sigmoid(s)
        sq_lower += jnp.sum((s_lower - t_lower) ** 2)
        abs_lower += jnp.sum(jnp.abs(s_lower - t_lower))
        sq_upper += jnp.sum((upper_leaky_hard_sigmoid(s) - upper_leaky_hard_sigmoid(t)) ** 2)
        # Value-weighted recall: what fraction of the teacher's total CI MASS lands on
        # elements the student also marks alive — immune to a sea of mask-irrelevant
        # boundary elements dominating the count-based recall.
        base_alive = s_lower > ALIVE_THRESHOLDS[0]
        w_hits += jnp.sum(jnp.where(base_alive, t_lower, 0.0))
        w_total += jnp.sum(t_lower)
        for thr, acc in by_thr.items():
            t_alive = t_lower > thr
            s_alive = s_lower > thr
            acc["hits"] += jnp.sum((t_alive & s_alive).astype(jnp.float32))
            acc["t_n"] += jnp.sum(t_alive.astype(jnp.float32))
            acc["s_n"] += jnp.sum(s_alive.astype(jnp.float32))
            acc["sq"] += jnp.sum(jnp.where(t_alive, (s_lower - t_lower) ** 2, 0.0))
        n += t.size
    out: dict[str, Float[Array, ""]] = {
        "mse_lower": sq_lower / n,
        "mae_lower": abs_lower / n,
        "mse_upper": sq_upper / n,
        "weighted_recall": w_hits / jnp.maximum(w_total, one),
    }
    # Base-rate-honest views at a THRESHOLD LADDER: teacher-alive at 1e-6 is ~1% of
    # elements and includes mask-irrelevant boundary values; 0.1 is the paper's rounding
    # threshold (sub-0.1 CI carries essentially no performance).
    for thr, acc in by_thr.items():
        tag = f"@{thr:g}"
        out[f"recall{tag}"] = acc["hits"] / jnp.maximum(acc["t_n"], one)
        out[f"precision{tag}"] = acc["hits"] / jnp.maximum(acc["s_n"], one)
        out[f"mse_on_alive{tag}"] = acc["sq"] / jnp.maximum(acc["t_n"], one)
        out[f"teacher_frac{tag}"] = acc["t_n"] / n
        out[f"student_frac{tag}"] = acc["s_n"] / n
    return out


def _n_params(student: CIFn) -> int:
    return sum(x.size for x in jax.tree.leaves(eqx.filter(student, eqx.is_inexact_array)))


def _wake_context(student: CIFn, key: PRNGKeyArray) -> CIFn:
    """Replace the zero-init ctx out_proj with a small Kaiming draw. The zero init is a
    TRAINING-time feature (context grows in as the game demands it) but a supervised-probe
    pathology: it makes the whole context pathway start at a saddle and crawl, so a short
    probe systematically under-uses context. No-op for gate-only."""
    if not isinstance(student, FactoredCIFn) or student.context is None:
        return student
    d_model = student.context.net.out_w.shape[0]
    fresh = jax.random.normal(key, student.context.net.out_w.shape) * d_model**-0.5
    return eqx.tree_at(lambda s: s.context.net.out_w, student, fresh)


def main(
    run_dir: str, steps: int = 20000, batch_size: int = 32, seed: int = 0, out: str | None = None
) -> None:
    run_path = Path(run_dir)
    out_path = Path(out) if out is not None else run_path / "distill_probe.jsonl"
    assert jax.process_count() == 1, "the distill probe is single-process"

    loaded = open_jax_run(run_path)
    teacher = loaded._state.ci_fn
    assert loaded.lm.leading_axes == ("sequence",) == teacher.expects_axes
    vu = loaded._state.components
    assert isinstance(vu, DecompVU)
    vu_compute: DecompVU = cast_floating(vu, COMPUTE_DT)

    data = loaded.config.data
    assert isinstance(data, DataConfig), "the distill probe is the LM (parquet) data path"
    schedule = BatchSchedule(scan_shards(data.dir), batch_size, seed)
    server = ShardServer(schedule, data.seq_len, jax.process_index(), 1)

    ctx_taps = tuple(sorted(teacher.input_names))
    first_tokens = jnp.asarray(server.local_batch(0))
    tap0 = _read_taps(loaded.lm, first_tokens, ctx_taps)
    widths = {name: int(t.shape[-1]) for name, t in tap0.items()}
    resid_width = widths[ctx_taps[0]]
    assert all(w == resid_width for w in widths.values()), widths
    input_dim = len(ctx_taps) * resid_width

    key = jax.random.PRNGKey(seed)
    archs = _student_archs(ctx_taps, input_dim, len(loaded.lm.sites))
    students: dict[str, CIFn] = {
        name: _wake_context(
            build_ci_fn(arch, loaded.lm.sites, jax.random.fold_in(key, i)),
            jax.random.fold_in(key, 1000 + i),
        )
        for i, (name, arch) in enumerate(archs.items())
    }
    merged_names = tuple(
        sorted({*teacher.input_names, *(n for s in students.values() for n in s.input_names)})
    )

    optimizers = {
        name: optax.adam(
            optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=STUDENT_LR.get(name, LR),
                warmup_steps=WARMUP_STEPS,
                decay_steps=steps,
                end_value=0.1 * STUDENT_LR.get(name, LR),
            )
        )
        for name in students
    }
    opt_states = {
        name: optimizers[name].init(eqx.filter(s, eqx.is_inexact_array))
        for name, s in students.items()
    }
    updates_fns = {name: _make_update(opt) for name, opt in optimizers.items()}

    param_counts = {name: _n_params(s) for name, s in students.items()}
    print(
        f"teacher run {loaded.run_id} @ step {loaded.step} | {len(loaded.lm.sites)} sites | "
        f"ctx_taps={len(ctx_taps)} (width {resid_width}, input_dim {input_dim}) | "
        f"{steps} steps x B={batch_size} seq={data.seq_len}",
        flush=True,
    )
    for name, n in param_counts.items():
        print(f"  student {name:<12} {n:>14,} params", flush=True)

    # Held-out batch (index past the training stream): metrics measure the FUNCTION fit,
    # not memorization of the current training batch.
    heldout_tokens = jnp.asarray(server.local_batch(steps + 7))
    heldout_taps, heldout_labels = _read_and_label(loaded.lm, teacher, heldout_tokens, merged_names)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_record: dict[str, dict[str, float]] = {}
    with out_path.open("a") as sink:
        header = {
            "run_id": loaded.run_id,
            "teacher_step": loaded.step,
            "steps": steps,
            "batch_size": batch_size,
            "seed": seed,
            "ctx_taps": list(ctx_taps),
            "input_dim": input_dim,
            "params": param_counts,
        }
        sink.write(json.dumps(header) + "\n")
        for step in range(steps):
            tokens = first_tokens if step == 0 else jnp.asarray(server.local_batch(step))
            taps, labels = _read_and_label(loaded.lm, teacher, tokens, merged_names)
            losses: dict[str, float] = {}
            for name in students:
                students[name], opt_states[name], loss = updates_fns[name](
                    students[name], opt_states[name], taps, labels, vu_compute
                )
                losses[name] = float(loss)
            if step % LOG_EVERY == 0 or step == steps - 1:
                record: dict[str, dict[str, float]] = {}
                for name, student in students.items():
                    m = _metrics(student, heldout_taps, heldout_labels, vu_compute)
                    record[name] = {k: float(v) for k, v in m.items()} | {
                        "train_loss": losses[name]
                    }
                    brief = {
                        k: record[name][k]
                        for k in ("recall@1e-06", "recall@0.1", "weighted_recall", "train_loss")
                    }
                    stats = " ".join(f"{k}={v:.5f}" for k, v in brief.items())
                    print(f"[{step:>6}] {name:<12} {stats}", flush=True)
                last_record = record
                sink.write(json.dumps({"step": step, "students": record}) + "\n")
                sink.flush()

    print(f"\n=== distill probe summary: teacher {loaded.run_id} @ step {loaded.step} ===")
    cols = (
        "recall@1e-06",
        "recall@0.1",
        "weighted_recall",
        "precision@0.1",
        "mse_on_alive@0.1",
        "train_loss",
    )
    print(f"{'student':<12} {'params':>14} " + " ".join(f"{c:>16}" for c in cols))
    for name in students:
        vals = " ".join(f"{last_record[name][c]:>16.6f}" for c in cols)
        print(f"{name:<12} {param_counts[name]:>14,} {vals}")
    print(f"\nrecords appended to {out_path}")


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
