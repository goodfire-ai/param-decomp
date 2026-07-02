"""CPU tests for the hidden-acts recon eval steps (SPEC S31) at LM shapes.

Pins that both step factories derive the waist `*leading` from the CI output, not from
the model input — for an LM the input is token ids `(B, T)` with no trailing d, so a
`residual.shape[:-1]` leading came out `(B,)` and the masked forward crashed on the
first in-loop slow eval (caught by the p-afa1e91c multi-host smoke).
"""

import types

import jax

from param_decomp.components import init_decomp_vu
from param_decomp.slow_eval import compute_hidden_acts_metrics
from param_decomp.targets.llama8b import llama_site_specs, mlp_family_site_cs
from param_decomp.tests.test_llama8b import _tiny_cfg, _tiny_decomposed_lm
from param_decomp.tests.test_slow_eval import _build_ci_fn


def test_hidden_acts_metrics_at_lm_token_input_shapes():
    cfg = _tiny_cfg()
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 5, 8))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    ci_fn = _build_ci_fn(lm, cfg.n_embd, jax.random.PRNGKey(2))
    components = init_decomp_vu(sites, jax.random.PRNGKey(1))
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)

    state = types.SimpleNamespace(components=components, ci_fn=ci_fn)
    metrics = compute_hidden_acts_metrics(
        lm, state, [tokens], n_mask_samples=2, base_key=jax.random.PRNGKey(7)
    )

    for class_name in ("CIHiddenActsReconLoss", "StochasticHiddenActsReconLoss"):
        assert class_name in metrics
        per_site = [k for k in metrics if k.startswith(f"{class_name}/")]
        assert len(per_site) == len(lm.site_names)
    assert all(v >= 0.0 for v in metrics.values())
