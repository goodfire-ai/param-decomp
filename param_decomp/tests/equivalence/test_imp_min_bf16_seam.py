"""Bounds the N3 bf16-input imp-min cast-point seam.

The fp32-only equivalence fixtures never push a bf16 ci through the power, so the one
remaining imp-min asymmetry is unmeasured: under autocast torch forms
`(ci_bf16 + eps) ** pnorm` as a **bf16 intermediate** then sums in fp32
(`importance_minimality.py` `per_component_lp_sums` + `lp_and_entropy_terms`), whereas
JAX casts `ci -> fp32` BEFORE the power (`losses.py::importance_minimality_terms`).

This feeds the byte-identical bf16 ci to both reductions and bounds the disagreement.
`imp_min_bf16_fixture.npz` stores ci as raw bf16 bits (uint16) so JAX reconstructs the
exact values torch reduced; `imp_min_bf16_reference.json` holds the frozen torch-oracle
`lp`/`entropy`. The JAX side reconstructs the bits and runs the real
`importance_minimality_terms`. Regenerate the golden from the `torch-oracle` tag only
when the imp-min math or the fixture changes (`param_decomp` imports no torch).

Measured worst-case relative error (this fixture): lp 2.06e-4, entropy 2.76e-4 — both
well under the 1e-3 tolerance pinned below. Recorded in the N3 spec note.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from param_decomp.losses import importance_minimality_terms

HERE = Path(__file__).resolve().parent
SEAM_RTOL = 1e-3


def _load_bf16_ci() -> dict[str, jnp.ndarray]:
    npz = np.load(HERE / "imp_min_bf16_fixture.npz")
    return {key.split("::", 1)[1]: jnp.asarray(npz[key]).view(jnp.bfloat16) for key in npz.files}


def test_imp_min_bf16_input_seam_within_tolerance():
    reference = json.loads((HERE / "imp_min_bf16_reference.json").read_text())
    ci_upper = _load_bf16_ci()

    lp, entropy = importance_minimality_terms(
        ci_upper, jnp.asarray(reference["pnorm"]), reference["eps"]
    )

    worst_rel = 0.0
    for name, jax_value, torch_value in (
        ("lp", float(lp), reference["torch_lp"]),
        ("entropy", float(entropy), reference["torch_entropy"]),
    ):
        rel = abs(jax_value - torch_value) / abs(torch_value)
        worst_rel = max(worst_rel, rel)
        assert rel <= SEAM_RTOL, (
            f"imp-min {name} bf16-input seam exceeded tolerance: "
            f"jax (fp32-first) {jax_value!r} vs torch (bf16 pow intermediate) "
            f"{torch_value!r} rel {rel:.3e} > {SEAM_RTOL:.0e}"
        )

    print(f"\nN3 imp-min bf16-input seam worst-case rel error: {worst_rel:.3e}")
