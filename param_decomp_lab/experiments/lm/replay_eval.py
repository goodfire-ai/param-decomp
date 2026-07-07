"""Fixed-batch, multi-restart PGD evaluation for replay experiments."""

import gc
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Int, PRNGKeyArray

from param_decomp.built_run import DataConfig
from param_decomp.ci_fn import CIFn
from param_decomp.components import DecompVU
from param_decomp.data import BatchSchedule, ShardServer, scan_shards
from param_decomp.eval import next_token_cross_entropy
from param_decomp.jit_util import filter_jit
from param_decomp.lm import DecomposedModel
from param_decomp.losses import kl_per_position
from param_decomp.sharding import batch_shard_leading, hsdp_mesh
from param_decomp.slow_eval import _per_component_ci_hist
from param_decomp.train import COMPUTE_DT, cast_floating
from param_decomp_lab.experiments.lm.config import load_run_dir_config
from param_decomp_lab.experiments.lm.load_run import open_jax_run
from param_decomp_lab.experiments.lm.run import _global_token_batch


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplayEvalOutput:
    pgd_losses: Array
    l0: Array
    mean_ci: Array
    ci_masked_kl: Array
    ci_masked_ce_difference: Array
    density_hist: dict[str, Array]


ReplayEvalStep = Callable[
    [DecomposedModel, DecompVU, CIFn, Int[Array, "B T"], PRNGKeyArray], ReplayEvalOutput
]


def _make_replay_eval_step(
    lm: DecomposedModel,
    pgd_steps: tuple[int, ...],
    n_restarts: int,
    density_n_bins: int,
    mesh: Mesh,
    compiler_options: dict[str, bool | int | str] | None,
) -> ReplayEvalStep:
    assert pgd_steps == tuple(sorted(set(pgd_steps))), pgd_steps
    assert pgd_steps[0] > 0 and n_restarts > 0 and density_n_bins > 0
    max_pgd_steps = pgd_steps[-1]
    site_names = lm.site_names
    component_counts = {site.name: site.C for site in lm.sites}

    def batch_sharded(x: Array) -> Array:
        return batch_shard_leading(x, mesh)

    def ci_sharded(x: Array) -> Array:
        return jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P(("replicate", "fsdp"), *((None,) * (x.ndim - 1))))
        )

    def replay_eval_step(
        model: DecomposedModel,
        components: DecompVU,
        ci_fn: CIFn,
        token_ids: Int[Array, "B T"],
        key: PRNGKeyArray,
    ) -> ReplayEvalOutput:
        token_ids = batch_sharded(token_ids)
        clean_output = batch_sharded(model.clean_output(token_ids))
        taps = model.read_activations(token_ids, ci_fn.input_names)
        prepared = model.prepare_compute_weights(cast_floating(components, COMPUTE_DT))
        ci_lower = {
            site: ci_sharded(value)
            for site, value in cast_floating(ci_fn, COMPUTE_DT)(taps, remat=False).lower.items()
        }
        leading = token_ids.shape

        def masked_output(sources: dict[str, Array]) -> Array:
            masks = {}
            delta_masks = {}
            for site in site_names:
                source = sources[site].astype(COMPUTE_DT)
                ci_site = ci_lower[site]
                masks[site] = ci_site + (1.0 - ci_site) * source[..., :-1]
                delta_masks[site] = source[..., -1]
            return batch_sharded(
                model.masked_output(
                    prepared,
                    token_ids,
                    masks,
                    delta_masks,
                    None,
                    site_names,
                    True,
                    remat=False,
                )
            )

        def pgd_loss(sources: dict[str, Array]) -> Array:
            return kl_per_position(masked_output(sources), clean_output)

        pgd_key = random.split(key, 3)[2]
        restart_keys = jnp.stack(
            (
                pgd_key,
                *(random.fold_in(pgd_key, restart_idx) for restart_idx in range(1, n_restarts)),
            )
        )
        initial_sources = {
            site: jax.vmap(
                lambda restart_key, site_idx=site_idx, site=site: random.uniform(
                    random.fold_in(restart_key, site_idx),
                    (1, 1, component_counts[site] + 1),
                    jnp.float32,
                )
            )(restart_keys)
            for site_idx, site in enumerate(site_names)
        }

        def attack(initial: dict[str, Array]) -> Array:
            snapshots = tuple(
                {site: jnp.zeros_like(initial[site]) for site in site_names} for _ in pgd_steps
            )

            def ascend(
                carry: tuple[dict[str, Array], tuple[dict[str, Array], ...]],
                ascent_idx: Array,
            ) -> tuple[
                tuple[dict[str, Array], tuple[dict[str, Array], ...]],
                None,
            ]:
                sources, saved = carry
                source_grads = jax.grad(pgd_loss)(sources)
                ascended = {
                    site: jnp.clip(sources[site] + 0.1 * jnp.sign(source_grads[site]), 0.0, 1.0)
                    for site in site_names
                }
                completed_steps = ascent_idx + 1
                new_saved = tuple(
                    {
                        site: jnp.where(completed_steps == milestone, ascended[site], snap[site])
                        for site in site_names
                    }
                    for milestone, snap in zip(pgd_steps, saved, strict=True)
                )
                return (ascended, new_saved), None

            (_, final_snapshots), _ = jax.lax.scan(
                ascend,
                (initial, snapshots),
                jnp.arange(max_pgd_steps),
            )
            return jnp.stack(tuple(pgd_loss(snapshot) for snapshot in final_snapshots))

        pgd_losses = jax.lax.map(attack, initial_sources)

        zeros_delta = {site: jnp.zeros(leading, COMPUTE_DT) for site in site_names}
        ci_masked_output = batch_sharded(
            model.masked_output(
                prepared,
                token_ids,
                ci_lower,
                zeros_delta,
                None,
                site_names,
                True,
                remat=False,
            )
        )
        target_ce = next_token_cross_entropy(clean_output, token_ids)
        ci_masked_ce = next_token_cross_entropy(ci_masked_output, token_ids)
        n_positions = math.prod(leading)
        total_components = sum(component_counts.values())
        density_hist = {
            site: _per_component_ci_hist(ci_lower[site], density_n_bins).sum(0)
            for site in site_names
        }
        l0 = (
            sum(
                ((ci_lower[site] > 0).astype(jnp.float32).sum() for site in site_names),
                start=jnp.zeros((), jnp.float32),
            )
            / n_positions
        )
        mean_ci = sum(
            (ci_lower[site].astype(jnp.float32).sum() for site in site_names),
            start=jnp.zeros((), jnp.float32),
        ) / (n_positions * total_components)
        return ReplayEvalOutput(
            pgd_losses=pgd_losses,
            l0=l0,
            mean_ci=mean_ci,
            ci_masked_kl=kl_per_position(ci_masked_output, clean_output),
            ci_masked_ce_difference=ci_masked_ce - target_ce,
            density_hist=density_hist,
        )

    return filter_jit(replay_eval_step, compiler_options=compiler_options)


def _scalar_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    assert array.ndim == 1 and array.size > 0
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "se": float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.1)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _write_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def evaluate(
    *,
    run_dirs: str,
    out_path: str,
    step: int = 400_000,
    n_batches: int = 32,
    n_restarts: int = 4,
    pgd_steps: str = "20,50",
    batch_size: int = 128,
    density_n_bins: int = 32,
    seed: int = 0,
) -> None:
    """Evaluate comma-separated run directories on identical batches and PGD starts."""
    paths = tuple(Path(value) for value in run_dirs.split(","))
    milestones = tuple(int(value) for value in pgd_steps.split(","))
    assert paths and n_batches > 0
    out = Path(out_path)
    jax.config.update("jax_compilation_cache_dir", str(out.parent / "xla_cache"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    configs = tuple(load_run_dir_config(path) for path in paths)
    first_data = configs[0].data
    assert isinstance(first_data, DataConfig)
    for config in configs[1:]:
        assert isinstance(config.data, DataConfig)
        assert (config.data.dir, config.data.seq_len) == (first_data.dir, first_data.seq_len)
        assert tuple((site.name, site.C) for site in config.target.sites) == tuple(
            (site.name, site.C) for site in configs[0].target.sites
        )
    for config in configs:
        assert config.eval is not None and config.eval.pgd is not None
        assert config.eval.pgd.step_size == 0.1, config.eval.pgd

    mesh = hsdp_mesh()
    assert mesh.devices.size <= batch_size and batch_size % mesh.devices.size == 0
    schedule = BatchSchedule(scan_shards(first_data.dir), batch_size, configs[0].pd.seed + 1)
    server = ShardServer(schedule, first_data.seq_len, process_index=0, process_count=1)
    batches = tuple(
        _global_token_batch(server.local_batch(batch_idx), mesh, batch_size)
        for batch_idx in range(n_batches)
    )

    results: dict[str, Any] = {
        "config": {
            "step": step,
            "n_batches": n_batches,
            "n_restarts": n_restarts,
            "pgd_steps": milestones,
            "batch_size": batch_size,
            "density_n_bins": density_n_bins,
            "seed": seed,
        },
        "runs": {},
    }
    _write_results(out, results)

    for run_idx, run_dir in enumerate(paths):
        loaded = open_jax_run(run_dir, step)
        assert isinstance(loaded._state.ci_fn, CIFn)
        eval_step = _make_replay_eval_step(
            loaded.lm,
            milestones,
            n_restarts,
            density_n_bins,
            mesh,
            loaded.config.runtime.compiler_options,
        )
        raw: list[dict[str, Any]] = []
        density_hist = np.zeros(density_n_bins + 1, np.int64)
        for batch_idx, batch in enumerate(batches):
            batch_key = random.fold_in(random.PRNGKey(seed), batch_idx)
            record = eval_step(
                loaded.lm,
                loaded._state.components,
                loaded._state.ci_fn,
                batch,
                batch_key,
            )
            pgd_losses = np.asarray(record.pgd_losses)
            assert pgd_losses.shape == (n_restarts, len(milestones))
            for site_hist in record.density_hist.values():
                density_hist += np.asarray(site_hist, np.int64)
            batch_record: dict[str, Any] = {
                "batch": batch_idx,
                "l0": float(record.l0),
                "mean_ci": float(record.mean_ci),
                "ci_masked_kl": float(record.ci_masked_kl),
                "ci_masked_ce_difference": float(record.ci_masked_ce_difference),
                "pgd": {
                    str(pgd_step): {
                        "restarts": pgd_losses[:, milestone_idx].tolist(),
                        "worst": float(pgd_losses[:, milestone_idx].max()),
                    }
                    for milestone_idx, pgd_step in enumerate(milestones)
                },
            }
            raw.append(batch_record)
            print(
                f"[{run_idx + 1}/{len(paths)} {loaded.run_id}] "
                f"batch {batch_idx + 1}/{n_batches}: "
                + ", ".join(
                    f"PGD-{pgd_step}={batch_record['pgd'][str(pgd_step)]['worst']:.4f}"
                    for pgd_step in milestones
                ),
                flush=True,
            )

        summary: dict[str, Any] = {
            metric: _scalar_summary([record[metric] for record in raw])
            for metric in ("l0", "mean_ci", "ci_masked_kl", "ci_masked_ce_difference")
        }
        summary["pgd"] = {
            str(pgd_step): _scalar_summary(
                [record["pgd"][str(pgd_step)]["worst"] for record in raw]
            )
            for pgd_step in milestones
        }
        results["runs"][loaded.run_id] = {
            "run_name": loaded.config.run.run_name,
            "checkpoint_step": loaded.step,
            "summary": summary,
            "density_hist": density_hist.tolist(),
            "batches": raw,
        }
        _write_results(out, results)
        del eval_step, loaded
        gc.collect()


def cli() -> None:
    fire.Fire(evaluate)


if __name__ == "__main__":
    cli()
