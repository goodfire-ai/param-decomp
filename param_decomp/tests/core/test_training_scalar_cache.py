"""Sweep magnitudes are consumed from state and reuse fresh-process executables."""

import dataclasses
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from param_decomp.core.configs import (
    AdamWOptimizerConfig,
    AuxiliaryReconstructionConfig,
    CaptureReconstruction,
    FaithfulnessLossConfig,
    FrequencyMinimalityConfig,
    ImportanceMinimalityLossConfig,
    NontargetConfig,
    StochasticReconLossConfig,
)
from param_decomp.core.faithfulness import faithfulness_loss_for
from param_decomp.core.losses import BatchFrequency
from param_decomp.core.objective import (
    MinimalityTerm,
    build_objective,
)
from param_decomp.core.run_state import _adamw_optimizer
from param_decomp.core.runtime_schedule import RuntimeSchedule
from param_decomp.core.schedule import Knot, ScheduleConfig
from param_decomp.core.train import (
    CIScaledWeightDecay,
    ForwardSubstrate,
    PDState,
    PDTrainingState,
    TargetedPDState,
    TrainState,
    make_targeted_train_step,
    make_train_step,
)
from param_decomp.tests.experiments.tms.test_targeted_tms import _tiny_setup

type StepKind = Literal["plain", "targeted"]


@dataclasses.dataclass(frozen=True)
class Sweep[State: (PDState[Array], TargetedPDState[Array])]:
    initial_state: State
    advance: Callable[[State], tuple[State, dict[str, Array]]]


def _curve(magnitude: float) -> ScheduleConfig:
    return ScheduleConfig(
        max_val=magnitude, points=(Knot(at=0.0, frac=0.5), Knot(at=1.0, frac=1.0))
    )


@overload
def _sweep(kind: Literal["plain"], lr_scale: float, loss_scale: float) -> Sweep[PDState[Array]]: ...


@overload
def _sweep(
    kind: Literal["targeted"], lr_scale: float, loss_scale: float
) -> Sweep[TargetedPDState[Array]]: ...


def _sweep(
    kind: StepKind, lr_scale: float, loss_scale: float
) -> Sweep[PDState[Array]] | Sweep[TargetedPDState[Array]]:
    losses = (
        ImportanceMinimalityLossConfig(
            coeff=_curve(0.03 * loss_scale),
            gamma=ScheduleConfig.constant(1.0),
            frequency=FrequencyMinimalityConfig(
                coeff=_curve(0.02 * loss_scale), reference_datapoint_count=4
            ),
        ),
        StochasticReconLossConfig(
            coeff=loss_scale,
            auxiliaries=(
                AuxiliaryReconstructionConfig(
                    name="hidden",
                    coeff=_curve(0.2 * loss_scale),
                    comparisons=(
                        CaptureReconstruction(
                            capture="linear1.out", distance="relative_squared_error"
                        ),
                    ),
                ),
            ),
        ),
    )
    nontarget = NontargetConfig(
        batch_size=4,
        impmin_coeff=_curve(0.03 * loss_scale),
        recon=[StochasticReconLossConfig(coeff=loss_scale)],
    )
    _, model, state, _ = _tiny_setup(losses, nontarget)
    optimizer = _adamw_optimizer(AdamWOptimizerConfig(lr_schedule=_curve(1e-3 * lr_scale)), 20)
    substrate = ForwardSubstrate.of(
        model,
        remat_recon_forwards=False,
        remat_ci_fn=False,
        ci_capture_keys=state.decomposition.ci_fn.capture_keys,
    )
    batch = jax.random.uniform(jax.random.PRNGKey(10), (4, 5))
    key = jax.random.PRNGKey(11)
    components_opt_state = optimizer.init(eqx.filter(state.decomposition.components, eqx.is_array))
    ci_fn_opt_state = optimizer.init(eqx.filter(state.decomposition.ci_fn, eqx.is_array))
    match kind:
        case "plain":
            plain_objective = build_objective(
                (FaithfulnessLossConfig(coeff=0.7 * loss_scale), *losses), model.model.sites
            )
            step = jax.jit(
                make_train_step(
                    model,
                    substrate=substrate,
                    components_optimizer=optimizer,
                    ci_fn_optimizer=optimizer,
                    total_steps=20,
                    faithfulness=faithfulness_loss_for(model),
                )
            )
            plain_state = TrainState(
                decomposition=state.decomposition,
                training=PDTrainingState(
                    frequency=BatchFrequency(),
                    objective=plain_objective,
                    components_opt_state=components_opt_state,
                    ci_fn_opt_state=ci_fn_opt_state,
                    adversaries=state.training.adversaries,
                    step=state.training.step,
                ),
            )
            return Sweep(plain_state, lambda value: step(model, value, batch.copy(), key.copy()))
        case "targeted":
            targeted_step = jax.jit(
                make_targeted_train_step(
                    model,
                    substrate=substrate,
                    components_optimizer=optimizer,
                    ci_fn_optimizer=optimizer,
                    total_steps=20,
                )
            )
            targeted_state = dataclasses.replace(
                state,
                training=dataclasses.replace(
                    state.training,
                    ci_scaled_weight_decay=CIScaledWeightDecay(
                        coeff=jnp.asarray(0.2 * loss_scale, jnp.float32)
                    ),
                    components_opt_state=components_opt_state,
                    ci_fn_opt_state=ci_fn_opt_state,
                ),
            )
            return Sweep(
                targeted_state,
                lambda value: targeted_step(model, value, batch.copy(), batch * 0.3, key.copy()),
            )


def run_sweep(kind: StepKind, lr_scale: float, loss_scale: float) -> tuple[float, float]:
    match kind:
        case "plain":
            return _advance_sweep(_sweep("plain", lr_scale, loss_scale), loss_scale)
        case "targeted":
            return _advance_sweep(_sweep("targeted", lr_scale, loss_scale), loss_scale)


def _advance_sweep[State: (PDState[Array], TargetedPDState[Array])](
    sweep: Sweep[State], loss_scale: float
) -> tuple[float, float]:
    before = jax.tree.map(
        np.array, eqx.filter(sweep.initial_state.decomposition.ci_fn, eqx.is_array)
    )
    state, first_metrics = sweep.advance(sweep.initial_state)
    update_norm = np.sqrt(
        sum(
            np.sum((np.asarray(after) - old) ** 2)
            for old, after in zip(
                jax.tree.leaves(before), jax.tree.leaves(state.decomposition.ci_fn), strict=True
            )
        )
    )
    state, metrics = sweep.advance(state)
    assert int(state.training.step) == 2
    assert np.isfinite(float(metrics["total"]))
    assert float(metrics["schedules/coeff/ImportanceMinimalityLoss"]) == pytest.approx(
        0.03 * loss_scale * (0.5 + 0.5 / 19)
    )
    assert float(metrics["schedules/coeff/ImportanceMinimalityLoss/frequency"]) == pytest.approx(
        0.02 * loss_scale * (0.5 + 0.5 / 19)
    )
    assert float(metrics["schedules/coeff/StochasticReconLoss/hidden"]) == pytest.approx(
        0.2 * loss_scale * (0.5 + 0.5 / 19)
    )
    assert "schedules/coeff/StochasticReconLoss" not in metrics
    return float(first_metrics["total"]), float(update_norm)


def _zero_coefficient_check[State: (PDState[Array], TargetedPDState[Array])](
    sweep: Sweep[State],
) -> tuple[Callable[[Callable[[State], RuntimeSchedule], Array], None], dict[str, Array]]:
    initial = sweep.initial_state
    _, metrics = sweep.advance(jax.tree.map(jnp.copy, initial))

    def check_zero(select: Callable[[State], RuntimeSchedule], derivative: Array) -> None:
        coefficient = select(initial).at(jnp.asarray(0.0, jnp.float32))
        assert float(derivative) > 0
        changed = eqx.tree_at(
            lambda state: select(state).magnitude, initial, jnp.zeros((), jnp.float32)
        )
        _, result = sweep.advance(jax.tree.map(jnp.copy, changed))
        assert float(result["total"]) == pytest.approx(
            float(metrics["total"] - coefficient * derivative), rel=1e-5, abs=1e-7
        )

    return check_zero, metrics


def _frequency(imp: MinimalityTerm) -> RuntimeSchedule:
    assert imp.frequency is not None
    return imp.frequency.coeff


def test_each_plain_coefficient_controls_its_loss_including_zero():
    check_zero, metrics = _zero_coefficient_check(_sweep("plain", 1.0, 1.0))
    check_zero(lambda state: state.training.objective.faith.coeff, metrics["faith"])
    check_zero(lambda state: state.training.objective.minimality.activity_coeff, metrics["imp"])
    check_zero(lambda state: _frequency(state.training.objective.minimality), metrics["freq"])
    check_zero(
        lambda state: state.training.objective.recon[0].coeff, metrics["loss/StochasticReconLoss"]
    )
    check_zero(
        lambda state: state.training.objective.recon[0].auxiliaries[0].coeff,
        metrics["loss/StochasticReconLoss/hidden/linear1.out"],
    )


def test_each_targeted_coefficient_controls_its_loss_including_zero():
    check_zero, metrics = _zero_coefficient_check(_sweep("targeted", 1.0, 1.0))
    assert float(metrics["schedules/coeff/nontarget/impmin"]) == pytest.approx(0.015)
    check_zero(
        lambda state: state.training.objective.target.minimality.activity_coeff, metrics["imp"]
    )
    check_zero(
        lambda state: _frequency(state.training.objective.target.minimality),
        metrics["freq"],
    )
    check_zero(
        lambda state: _frequency(state.training.objective.nontarget.minimality),
        metrics["loss/nontarget/freq"],
    )
    check_zero(
        lambda state: state.training.objective.target.recon[0].coeff,
        metrics["loss/StochasticReconLoss"],
    )
    check_zero(
        lambda state: state.training.objective.target.recon[0].auxiliaries[0].coeff,
        metrics["loss/StochasticReconLoss/hidden/linear1.out"],
    )
    check_zero(
        lambda state: state.training.objective.nontarget.recon[0].coeff,
        metrics["loss/nontarget/StochasticReconLoss"],
    )
    check_zero(
        lambda state: state.training.objective.nontarget.minimality.activity_coeff,
        metrics["loss/nontarget/imp"],
    )


@pytest.mark.parametrize("kind", ["plain", "targeted"])
def test_fresh_sweep_process_hits_persistent_cache(tmp_path: Path, kind: str):
    script = """
import jax, json, sys
from param_decomp.tests.core.test_training_scalar_cache import run_sweep
jax.config.update("jax_compilation_cache_dir", sys.argv[4])
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_explain_cache_misses", True)
jax.config.update("jax_log_compiles", True)
print(json.dumps(run_sweep(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))))
"""
    runs = [
        subprocess.run(
            [sys.executable, "-c", script, kind, str(lr), str(loss), str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        for lr, loss in ((1, 1), (3, 1), (1, 3))
    ]
    entry = "jit_step" if kind == "plain" else "jit_targeted_step"
    events = ["\n".join(line for line in run.stderr.splitlines() if entry in line) for run in runs]
    assert f"PERSISTENT COMPILATION CACHE MISS for '{entry}'" in events[0], events[0]
    for event in events[1:]:
        assert f"Persistent compilation cache hit for '{entry}'" in event, event
    baseline, lr_only, loss_only = [json.loads(run.stdout) for run in runs]
    assert baseline[1] > 0
    assert lr_only[0] == pytest.approx(baseline[0])
    assert lr_only[1] == pytest.approx(3 * baseline[1], rel=2e-3)
    assert loss_only[0] != pytest.approx(baseline[0])
