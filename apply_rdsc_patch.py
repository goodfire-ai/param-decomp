import pathlib

def patch(path, old, new, desc):
    p = pathlib.Path(path); s = p.read_text()
    if new in s:
        print(f"  already applied: {desc}"); return
    assert old in s, f"ANCHOR NOT FOUND: {desc}"
    assert s.count(old) == 1, f"AMBIGUOUS ANCHOR ({s.count(old)}x): {desc}"
    p.write_text(s.replace(old, new, 1)); print(f"  patched: {desc}")

CFG = """    gamma_anneal_start_frac: Probability = 1.0
    gamma_anneal_final_gamma: PositiveFloat | None = None
    gamma_anneal_end_frac: Probability = 1.0"""

# ---------------- losses.py ----------------
L = "param_decomp/losses.py"
patch(L,
"""    AnyImportanceMinimalityLossConfig,
    ImportanceMinimalityLossConfig,
    SmoothL0ImportanceMinimalityLossConfig,""",
"""    AnyImportanceMinimalityLossConfig,
    ArctanImportanceMinimalityLossConfig,
    FractionImportanceMinimalityLossConfig,
    ImportanceMinimalityLossConfig,
    MCPImportanceMinimalityLossConfig,
    SmoothL0ImportanceMinimalityLossConfig,""",
"losses import")

patch(L,
"    return _imp_min_terms(ci_upper, lambda ci: ci**2 / (ci**2 + gamma_sq))\n",
'''    return _imp_min_terms(ci_upper, lambda ci: ci**2 / (ci**2 + gamma_sq))


@jaxtyped(typechecker=beartype)
def fraction_importance_minimality_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]], gamma: Float[Array, ""]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Fractional (q=1) penalty `c / (c + gamma)`: `phi'(0) = 1/gamma > 0` (a kill-force at
    the origin that drives small CI to exactly 0 via the hard floor), redescends for large
    c. Bounded, no singularity."""
    return _imp_min_terms(ci_upper, lambda ci: ci / (ci + gamma))


@jaxtyped(typechecker=beartype)
def mcp_importance_minimality_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]], gamma: Float[Array, ""]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """MCP penalty: `(c/gamma)(2 - c/gamma)` up to the knee `c = gamma`, flat (=1) above.
    `phi'(0) = 2/gamma > 0`; `phi' = 0` for `c > gamma` (zero bias on clearly-on)."""
    return _imp_min_terms(
        ci_upper, lambda ci: jnp.where(ci < gamma, (ci / gamma) * (2.0 - ci / gamma), 1.0)
    )


@jaxtyped(typechecker=beartype)
def arctan_importance_minimality_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]], gamma: Float[Array, ""]
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Arctan penalty `arctan(c/gamma)/(pi/2)`: `phi'(0) = 2/(pi*gamma) > 0` (gentler than
    fraction/MCP), smooth, redescending."""
    return _imp_min_terms(ci_upper, lambda ci: jnp.arctan(ci / gamma) / (jnp.pi / 2.0))
''',
"losses term fns")

patch(L,
"""        case SmoothL0ImportanceMinimalityLossConfig():
            return annealed_gamma(step_f32, total_steps, cfg)""",
"""        case (
            SmoothL0ImportanceMinimalityLossConfig()
            | FractionImportanceMinimalityLossConfig()
            | MCPImportanceMinimalityLossConfig()
            | ArctanImportanceMinimalityLossConfig()
        ):
            return annealed_gamma(step_f32, total_steps, cfg)""",
"losses annealed dispatch")

patch(L,
"""        case SmoothL0ImportanceMinimalityLossConfig():
            return smooth_l0_importance_minimality_terms(ci_upper, annealed_param)""",
"""        case SmoothL0ImportanceMinimalityLossConfig():
            return smooth_l0_importance_minimality_terms(ci_upper, annealed_param)
        case FractionImportanceMinimalityLossConfig():
            return fraction_importance_minimality_terms(ci_upper, annealed_param)
        case MCPImportanceMinimalityLossConfig():
            return mcp_importance_minimality_terms(ci_upper, annealed_param)
        case ArctanImportanceMinimalityLossConfig():
            return arctan_importance_minimality_terms(ci_upper, annealed_param)""",
"losses imp_min_terms dispatch")

# ---------------- configs.py ----------------
C = "param_decomp/configs.py"
patch(C,
"""# The two importance-minimality penalties share the `coeff`/`beta` surface and the
# `lp + beta * entropy` aggregation; they differ only in the per-value penalty shape and
# its annealed parameter (`p` vs `gamma`). The trainer's imp-min slot accepts either.
AnyImportanceMinimalityLossConfig = (
    ImportanceMinimalityLossConfig | SmoothL0ImportanceMinimalityLossConfig
)""",
f'''class FractionImportanceMinimalityLossConfig(LossMetricConfig):
    """Fractional (q=1) imp-min penalty `phi(c) = c / (c + gamma)` on upper-leaky CI. Like
    smooth-L0 but first-order in c, so `phi'(0) = 1/gamma > 0`: a nonzero kill-force at the
    origin that drives small-but-positive CI to exactly 0 (via the hard floor), while still
    redescending for clearly-on components. `gamma` annealed as in smooth-L0."""

    type: Literal["FractionImportanceMinimalityLoss"] = "FractionImportanceMinimalityLoss"
    gamma: PositiveFloat
    beta: NonNegativeFloat
{CFG}


class MCPImportanceMinimalityLossConfig(LossMetricConfig):
    """Minimax-concave imp-min penalty: `phi(c) = (c/gamma)(2 - c/gamma)` up to the knee
    `c = gamma`, flat (=1) above. `phi'(0) = 2/gamma > 0` (kills the tail), `phi' = 0` for
    `c > gamma` (zero bias on clearly-on components), bounded throughout."""

    type: Literal["MCPImportanceMinimalityLoss"] = "MCPImportanceMinimalityLoss"
    gamma: PositiveFloat
    beta: NonNegativeFloat
{CFG}


class ArctanImportanceMinimalityLossConfig(LossMetricConfig):
    """Arctan imp-min penalty `phi(c) = arctan(c/gamma)/(pi/2)`. `phi'(0) = 2/(pi*gamma) > 0`
    (a gentler kill-force than fraction/MCP), smooth, redescending. `gamma` annealed as in
    smooth-L0."""

    type: Literal["ArctanImportanceMinimalityLoss"] = "ArctanImportanceMinimalityLoss"
    gamma: PositiveFloat
    beta: NonNegativeFloat
{CFG}


# The importance-minimality penalties share the `coeff`/`beta` surface and the
# `lp + beta * entropy` aggregation; they differ only in the per-value penalty shape and
# its annealed parameter. The trainer's imp-min slot accepts any of them.
AnyImportanceMinimalityLossConfig = (
    ImportanceMinimalityLossConfig
    | SmoothL0ImportanceMinimalityLossConfig
    | FractionImportanceMinimalityLossConfig
    | MCPImportanceMinimalityLossConfig
    | ArctanImportanceMinimalityLossConfig
)''',
"configs new classes + Any union")

patch(C,
"    | SmoothL0ImportanceMinimalityLossConfig\n",
"""    | SmoothL0ImportanceMinimalityLossConfig
    | FractionImportanceMinimalityLossConfig
    | MCPImportanceMinimalityLossConfig
    | ArctanImportanceMinimalityLossConfig
""",
"configs AnyLossMetricConfig union")

# ---------------- recon.py ----------------
R = "param_decomp/recon.py"
patch(R,
"    SmoothL0ImportanceMinimalityLossConfig,\n",
"""    SmoothL0ImportanceMinimalityLossConfig,
    FractionImportanceMinimalityLossConfig,
    MCPImportanceMinimalityLossConfig,
    ArctanImportanceMinimalityLossConfig,
""",
"recon import")

patch(R,
"""            case SmoothL0ImportanceMinimalityLossConfig():
                assert imp_min is None
                assert cfg.gamma_anneal_final_gamma is not None
                imp_min = cfg""",
"""            case (
                SmoothL0ImportanceMinimalityLossConfig()
                | FractionImportanceMinimalityLossConfig()
                | MCPImportanceMinimalityLossConfig()
                | ArctanImportanceMinimalityLossConfig()
            ):
                assert imp_min is None
                assert cfg.gamma_anneal_final_gamma is not None
                imp_min = cfg""",
"recon build_recon_terms case")

print("ALL PATCHES APPLIED")
