import jax
import jax.numpy as jnp

from param_decomp.built_run import EvalPGDConfig
from param_decomp.components import init_decomp_vu
from param_decomp.eval import make_eval_step
from param_decomp.sharding import hsdp_mesh
from param_decomp.targets.llama8b import llama_site_specs, mlp_family_site_cs
from param_decomp.tests.test_eval import _build_ci_fn
from param_decomp.tests.test_llama8b import _tiny_cfg, _tiny_decomposed_lm
from param_decomp_lab.experiments.lm.replay_eval import _make_replay_eval_step


def test_replay_eval_matches_single_restart_pgd_probe():
    mesh = hsdp_mesh()
    cfg = _tiny_cfg()
    sites = llama_site_specs(cfg, mlp_family_site_cs(4, 4, 8))
    lm = _tiny_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(lm, cfg.n_embd, jax.random.PRNGKey(2))
    token_ids = jax.random.randint(
        jax.random.PRNGKey(3), (2 * mesh.devices.size, 16), 0, cfg.vocab_size
    )
    key = jax.random.PRNGKey(5)

    reference_step = make_eval_step(
        lm,
        rounding_threshold=0.0,
        ci_alive_threshold=0.0,
        l0_group_patterns={"total": ("*",)},
        pgd=EvalPGDConfig(n_steps=8, step_size=0.1),
        mesh=mesh,
    )
    replay_step = _make_replay_eval_step(
        lm,
        pgd_steps=(8,),
        n_restarts=1,
        density_n_bins=8,
        mesh=mesh,
        compiler_options=None,
    )

    reference = reference_step(lm, vu, ci_fn, token_ids, key)
    replay = replay_step(lm, vu, ci_fn, token_ids, key)

    assert jnp.allclose(replay.pgd_losses[0, 0], reference["loss/PGDReconLoss"], rtol=1e-6)
    assert jnp.allclose(replay.l0, reference["l0/0.0_total"], rtol=1e-6)
    assert jnp.allclose(replay.ci_masked_kl, reference["ce_kl/kl_ci_masked"], rtol=1e-6)
    assert jnp.allclose(
        replay.ci_masked_ce_difference,
        reference["ce_kl/ce_difference_ci_masked"],
        rtol=1e-6,
    )
    assert sum(int(hist.sum()) for hist in replay.density_hist.values()) == (
        token_ids.size * sum(site.C for site in sites)
    )
