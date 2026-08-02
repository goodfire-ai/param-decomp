"""Pure lifecycle surgery primitives for capacity birth / column probes (task #820).

Everything here is a pure function on `(TrainState, site, slot)` — no run-loop, config,
or model coupling. The caller (the #811 controller harness) supplies the per-site
represented-matrix reconstruction gradient `G_s = d r_adv / d M_s` for direction
selection; the recommended zero-plumbing probe is documented on
`birth_direction_from_grad`. Spec: lore
2026-08-02--design--reconstruction-budget-control-and-demand-triggered-capacity-birth
(+ the column-generation addendum). Fail-closed everywhere: unsupported CI
architectures and optimizer states raise, they never no-op.
"""

from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from param_decomp.core.ci_fn import CIFn, GlobalMLPCIFn, LayerwiseMLPCIFn
from param_decomp.core.components import ComponentStacks
from param_decomp.core.train import Decomposition, TrainingItem, TrainState

# ----------------------------- slot discovery -----------------------------


def find_inactive_slot(components: ComponentStacks, site: str) -> int | None:
    """Lowest slot whose U row is EXACTLY zero (the exact-null convention of
    svd_null_tail and of post-rollback slots). None when the site is full — the
    controller's CAPACITY_EXHAUSTED input, never an exception."""
    _, U = components.site(site)
    null_rows = jnp.all(U == 0.0, axis=1)
    idx = int(jnp.argmax(null_rows))
    return idx if bool(null_rows[idx]) else None


# ----------------------------- birth direction -----------------------------


def birth_direction_from_grad(
    G: Float[Array, "d_in d_out"], n_iters: int = 64
) -> tuple[Array, Array]:
    """Leading left singular direction `p` (unit) and value `sigma` of the represented-
    matrix gradient `G = d r_adv / d M_s` — the GradMax-optimal rank-1 birth direction:
    with `V[:, c] = p, U[c, :] = 0` the represented matrix is unchanged exactly, while
    `d r_adv / d U[c, :] = p^T G = sigma q^T` has the maximum first-order norm among
    unit-norm choices of the live factor.

    How the caller gets `G` with zero model plumbing: for a null slot (`U_c = 0`),
    `dL/dU_c` evaluated at the unperturbed function with `V_c = v` and the slot's mask
    forced 1 equals `v^T G`; symmetrically `dL/dV_c` at `V_c = 0, U_c = u` equals
    `G u` — so 2–3 alternating probe backwards on one scratch slot power-iterate the
    same pair this function computes, when materializing `G` is too big. Here At LM scale
    use that alternating factor-gradient path rather than materializing dense `G`."""
    g = G.astype(jnp.float32)
    v = jnp.ones((g.shape[1],), jnp.float32) / jnp.sqrt(g.shape[1])
    p = g @ v
    sigma = jnp.linalg.norm(p)
    for _ in range(n_iters):
        p = g @ v
        p = p / (jnp.linalg.norm(p) + 1e-30)
        v = g.T @ p
        sigma = jnp.linalg.norm(v)
        v = v / (sigma + 1e-30)
    return p, sigma


def select_birth_site(grads_by_site: dict[str, Array]) -> tuple[str, Array, Array]:
    """The site with the largest leading singular value of its represented-matrix
    gradient, normalized by sqrt(numel) so heterogeneous site shapes compare fairly.
    Returns `(site, p, sigma)`."""
    assert grads_by_site, "no candidate sites"
    best: tuple[str, Array, Array] | None = None
    best_score = -jnp.inf
    for site, G in grads_by_site.items():
        p, sigma = birth_direction_from_grad(G)
        score = sigma / jnp.sqrt(jnp.asarray(G.size, jnp.float32))
        if bool(score > best_score):
            best, best_score = (site, p, sigma), score
    assert best is not None
    return best


# ----------------------------- state surgery -----------------------------


def _edited_stacks(
    components: ComponentStacks, site: str, slot: int, v_col: Array, u_row: Array
) -> ComponentStacks:
    shape, stack_idx = {name: (s, i) for name, s, i in components.site_slots}[site]
    Vs, Us = components.stacks[shape]
    assert 0 <= slot < Vs.shape[2], (slot, Vs.shape)
    new_stacks = dict(components.stacks)
    new_stacks[shape] = (
        Vs.at[stack_idx, :, slot].set(v_col.astype(Vs.dtype)),
        Us.at[stack_idx, slot, :].set(u_row.astype(Us.dtype)),
    )
    return ComponentStacks(stacks=new_stacks, site_slots=components.site_slots)


def _ci_head_leaves(ci_fn: CIFn, site: str) -> tuple[Array, Array, int]:
    """(final weights, final bias, column offset of `site` slot 0) — the two leaves CI
    surgery edits. Fail-closed on any other architecture."""
    match ci_fn:
        case LayerwiseMLPCIFn():
            mlp = ci_fn.site_mlps[site]
            return mlp.weights[-1], mlp.biases[-1], 0
        case GlobalMLPCIFn():
            offset = 0
            for name, c in zip(ci_fn.output_names, ci_fn.c_sizes, strict=True):
                if name == site:
                    return ci_fn.mlp.weights[-1], ci_fn.mlp.biases[-1], offset
                offset += c
            raise AssertionError(f"site {site!r} not in {ci_fn.output_names}")
        case _:
            raise NotImplementedError(
                f"slot surgery supports the MLP CI fns only, not {type(ci_fn).__name__}"
            )


def _with_ci_head(ci_fn: CIFn, site: str, weights: Array, bias: Array) -> CIFn:
    import equinox as eqx

    match ci_fn:
        case LayerwiseMLPCIFn():
            return eqx.tree_at(
                lambda f: (f.site_mlps[site].weights[-1], f.site_mlps[site].biases[-1]),
                ci_fn,
                (weights, bias),
            )
        case GlobalMLPCIFn():
            return eqx.tree_at(
                lambda f: (f.mlp.weights[-1], f.mlp.biases[-1]), ci_fn, (weights, bias)
            )
        case _:
            raise NotImplementedError(type(ci_fn).__name__)


@dataclass(frozen=True)
class TrialSnapshot:
    """The ENTIRE pre-trial `TrainState` (independently-owned buffers) plus the trial's
    identity. During a COLUMN_PROBE every other parameter, both optimizers' moments, and
    the persistent adversaries keep training — restoring only the trial slot would
    accept/reject on contaminated state, so rollback returns to the full pre-probe
    frontier. The buffers are COPIES: immutability alone is not a snapshot under the
    train step's buffer donation, which deletes the referenced buffers on the next
    step (found live in the first acceptance run)."""

    state: TrainState
    site: str
    slot: int


def snapshot_trial(state: TrainState, site: str, slot: int) -> TrialSnapshot:
    import equinox as eqx

    owned = jax.tree_util.tree_map(
        lambda leaf: jnp.copy(leaf) if eqx.is_array(leaf) else leaf, state
    )
    return TrialSnapshot(state=owned, site=site, slot=slot)


def rollback_trial(snapshot: TrialSnapshot) -> TrainState:
    """Bitwise return to the pre-trial frontier: every leaf of the snapshot's state —
    all V/U, the whole CI fn, both optimizer states, adversaries, and the step counter."""
    return snapshot.state


def _adam_state(opt_state: Any) -> optax.ScaleByAdamState:
    holder: list[optax.ScaleByAdamState] = []

    def visit(node: Any) -> None:
        if isinstance(node, optax.ScaleByAdamState):
            holder.append(node)
        elif type(node) is tuple:
            for x in node:
                visit(x)

    visit(opt_state)
    assert len(holder) == 1, f"expected exactly one ScaleByAdamState, found {len(holder)}"
    return holder[0]


def _ci_head_leaves_like(tree: Any, ci_fn: CIFn, site: str) -> tuple[Array, Array, int]:
    """The final-layer leaves of a CI-fn-SHAPED tree (an Adam moment tree), located via
    the same structure as `_ci_head_leaves`."""
    match ci_fn:
        case LayerwiseMLPCIFn():
            return tree.site_mlps[site].weights[-1], tree.site_mlps[site].biases[-1], 0
        case GlobalMLPCIFn():
            offset = 0
            for name, c in zip(ci_fn.output_names, ci_fn.c_sizes, strict=True):
                if name == site:
                    return tree.mlp.weights[-1], tree.mlp.biases[-1], offset
                offset += c
            raise AssertionError(site)
        case _:
            raise NotImplementedError(type(ci_fn).__name__)


def _edit_slot_everywhere(
    state: TrainState,
    site: str,
    slot: int,
    v_col: Array,
    u_row: Array,
    ci_w_col: Array,
    ci_b_val: Array,
    vu_moments: tuple[Array, Array, Array, Array],
    ci_moments: tuple[Array, Array, Array, Array],
) -> TrainState:
    """Write the slot's factor slices, CI-head column/bias, and Adam-moment slices —
    the ONE mutation path shared by birth and rollback so they cannot drift apart.
    `vu_moments = (mu_v, mu_u, nu_v, nu_u)`, `ci_moments = (mu_w_col, mu_b, nu_w_col, nu_b)`."""
    import equinox as eqx

    decomposition = state.decomposition
    ci_fn = decomposition.ci_fn
    ci_w, ci_b, offset = _ci_head_leaves(ci_fn, site)
    col = offset + slot

    new_components = _edited_stacks(decomposition.components, site, slot, v_col, u_row)
    new_ci = _with_ci_head(
        ci_fn, site,
        ci_w.at[:, col].set(ci_w_col.astype(ci_w.dtype)),
        ci_b.at[col].set(jnp.asarray(ci_b_val, ci_b.dtype)),
    )  # fmt: skip

    mu_v, mu_u, nu_v, nu_u = vu_moments
    ci_mu_w, ci_mu_b, ci_nu_w, ci_nu_b = ci_moments

    vu_adam = _adam_state(state.training.components_opt_state)
    new_vu_opt = _replace_adam(
        state.training.components_opt_state,
        vu_adam._replace(
            mu=_edited_stacks(
                cast(ComponentStacks, cast(object, vu_adam.mu)), site, slot, mu_v, mu_u
            ),
            nu=_edited_stacks(
                cast(ComponentStacks, cast(object, vu_adam.nu)), site, slot, nu_v, nu_u
            ),
        ),
    )

    def edit_ci_moment_tree(tree: Any, w_col: Array, b_val: Array) -> Any:
        w, b, off = _ci_head_leaves_like(tree, ci_fn, site)
        new_w = w.at[:, off + slot].set(w_col.astype(w.dtype))
        new_b = b.at[off + slot].set(jnp.asarray(b_val, b.dtype))
        match ci_fn:
            case LayerwiseMLPCIFn():
                return eqx.tree_at(
                    lambda t: (t.site_mlps[site].weights[-1], t.site_mlps[site].biases[-1]),
                    tree, (new_w, new_b),
                )  # fmt: skip
            case GlobalMLPCIFn():
                return eqx.tree_at(
                    lambda t: (t.mlp.weights[-1], t.mlp.biases[-1]), tree, (new_w, new_b)
                )
            case _:
                raise NotImplementedError(type(ci_fn).__name__)

    ci_adam = _adam_state(state.training.ci_fn_opt_state)
    new_ci_opt = _replace_adam(
        state.training.ci_fn_opt_state,
        ci_adam._replace(
            mu=edit_ci_moment_tree(ci_adam.mu, ci_mu_w, ci_mu_b),
            nu=edit_ci_moment_tree(ci_adam.nu, ci_nu_w, ci_nu_b),
        ),
    )

    return TrainState(
        decomposition=Decomposition(components=new_components, ci_fn=new_ci),
        training=TrainingItem(
            components_opt_state=new_vu_opt,
            ci_fn_opt_state=new_ci_opt,
            adversaries=state.training.adversaries,
            step=state.training.step,
        ),
    )


def _replace_adam(opt_state: Any, new_adam: optax.ScaleByAdamState) -> Any:
    replaced = 0

    def rewrite(node: Any) -> Any:
        nonlocal replaced
        if isinstance(node, optax.ScaleByAdamState):
            replaced += 1
            return new_adam
        if type(node) is tuple:
            return tuple(rewrite(x) for x in node)
        return node

    out = rewrite(opt_state)
    assert replaced == 1, replaced
    return out


def birth_slot(state: TrainState, site: str, slot: int, direction: Array) -> TrainState:
    """Function-preserving GradMax birth: `V[:, slot] = direction` (unit-norm asserted),
    `U[slot, :] = 0` — the represented matrix is unchanged exactly — with the slot's
    CI-head column zeroed, its bias set to 1 (open after protection), and the slot's
    Adam moments in BOTH optimizers reset to zero (a newborn carries no history).
    Gate protection during settling is the caller's job via `protected_mask` +
    `StepControls`."""
    V, _ = state.decomposition.components.site(site)
    assert direction.shape == (V.shape[0],), (direction.shape, V.shape)
    norm = jnp.linalg.norm(direction.astype(jnp.float32))
    assert bool(jnp.isfinite(norm)) and float(norm) > 0.0
    unit = direction.astype(jnp.float32) / norm
    _, U = state.decomposition.components.site(site)
    assert bool(jnp.all(U[slot, :] == 0.0)), f"slot {slot} of {site} is not an exact null"
    ci_w, _, _ = _ci_head_leaves(state.decomposition.ci_fn, site)
    zeros_w = jnp.zeros_like(ci_w[:, 0])
    zero = jnp.zeros(())
    return _edit_slot_everywhere(
        state, site, slot,
        v_col=unit, u_row=jnp.zeros((U.shape[1],)),
        ci_w_col=zeros_w, ci_b_val=jnp.ones(()),
        vu_moments=(jnp.zeros_like(unit), jnp.zeros((U.shape[1],)),
                    jnp.zeros_like(unit), jnp.zeros((U.shape[1],))),
        ci_moments=(zeros_w, zero, zeros_w, zero),
    )  # fmt: skip


def protected_mask(components: ComponentStacks, site: str, slot: int) -> dict[str, Array]:
    """The `StepControls.protected` entry holding exactly this slot's gate open."""
    _, U = components.site(site)
    C = U.shape[0]
    return {site: jnp.zeros((C,), bool).at[slot].set(True)}


def truncate_active_prefix(state: TrainState, active_by_site: dict[str, int]) -> TrainState:
    """Turn every slot at or beyond `active_by_site[site]` into an exact inactive null:
    U row zero, CI-head column zero with bias 0 (closed, NOT protected-open — these are
    dead capacity, not newborns), and the slot's Adam moments cleared in both
    optimizers. V columns keep their values (null-ness is defined by the U row; see
    `find_inactive_slot`). Idempotent. This is how an acceptance case authors
    'physical Cmax with a smaller logical active width' without a second initializer."""
    for site, k in active_by_site.items():
        V, U = state.decomposition.components.site(site)
        assert 0 < k <= U.shape[0], (site, k, U.shape)
        zeros_w = jnp.zeros_like(_ci_head_leaves(state.decomposition.ci_fn, site)[0][:, 0])
        zero = jnp.zeros(())
        for slot in range(k, U.shape[0]):
            state = _edit_slot_everywhere(
                state, site, slot,
                v_col=V[:, slot], u_row=jnp.zeros((U.shape[1],)),
                ci_w_col=zeros_w, ci_b_val=zero,
                vu_moments=(jnp.zeros((V.shape[0],)), jnp.zeros((U.shape[1],)),
                            jnp.zeros((V.shape[0],)), jnp.zeros((U.shape[1],))),
                ci_moments=(zeros_w, zero, zeros_w, zero),
            )  # fmt: skip
    return state
