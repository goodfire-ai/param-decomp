"""Real one-backward CI-fn gradient check: single-pool vs 3-pool.

Runs ONE forward+backward on a tiny gpt2-attn decomposition with a FIXED,
deterministic batch (no dataloader nondeterminism) and dumps the CI fn
parameters' ``.grad`` to disk. A companion ``compare`` mode loads two dumps and
reports max-abs / relative error.

This is the *real* gradient check the loss-curve harness can't do: it compares
the actual autograd gradient that lands on the CI fn params, at the same point
in both code paths —

  * single-pool: right after ``total_loss.backward()``, before grad clip / step.
  * 3-pool CI pool: right after the CI in-pool AVG-reduce
    (``all_reduce_ci_fn_grads``), before cross-pool clip / step. Every CI rank
    then holds the global grad, so any one CI rank's dump is canonical.

Both paths have ``grad_clip_norm_ci_fn = None`` in the config, so no clip
perturbs the dumped grad — it's the pure backward (+ AVG-reduce) gradient.

Determinism: the CI fn is seeded identically (``pd.seed`` drives V/U + CI fn
init in both trainers via ``seed_all_ranks``), the batch is a fixed tensor, and
we run exactly one step (``current_frac_of_training = 0``).

Usage (single torchrun launch per mode):

  # single-pool reference (1 GPU)
  torchrun --standalone --nproc_per_node=1 grad_check.py run \
      --mode singlepool --n_ci 2 --n_per_block 4 --n_ppgd 2 --out <dir>

  # 3-pool (8 GPU); topology built from n_ci/n_per_block/n_ppgd
  torchrun --standalone --nproc_per_node=8 grad_check.py run \
      --mode threepool --n_ci 2 --n_per_block 4 --n_ppgd 2 --out <dir>

  # compare (no GPU needed)
  python grad_check.py compare --a <dir_1p> --b <dir_3p>

The ``--n_ci/--n_per_block/--n_ppgd`` only affect the 3-pool topology; single
pool ignores them for compute but records them so the dumps pair up. The fix
itself is NOT toggled here — run against the with-fix / without-fix source tree
(the driver checks out ``step_layerwise.py`` versions).
"""

import argparse
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from param_decomp.configs import Cadence
from param_decomp.optimize import Trainer
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.experiments.lm.run import (
    LMExperimentConfig,
    build_target,
    make_run_batch,
)
from param_decomp_lab.three_pool import ThreePoolConfig, ThreePoolTrainer

BASE_YAML = Path(__file__).parent / "base.yaml"
SEQ_LEN = 128

# Constant the stochastic-mask uniform draw is pinned to during a grad check.
# The stochastic recon mask is ``ci + (1 - ci) * u`` with ``u ~ Uniform[0,1)``
# drawn via torch.rand / torch.rand_like — at the SHARDED batch granularity, so
# 1p (full batch) and 3p (per-rank shards) draw different-shaped tensors and can
# never agree even with the same seed. Pinning ``u`` to a constant makes the mask
# deterministic and identical across both code paths, isolating the gradient
# SCALING (the bug the fix targets) from this irreducible RNG mismatch. The mask
# stays a valid interpolation; only its randomness is removed.
_DETERMINISTIC_U = 0.5


class _PinRandToConstant:
    """Context manager: make ``torch.rand`` / ``torch.rand_like`` return a constant.

    Used only inside the single grad-check step so the stochastic mask is the
    same in 1p and 3p regardless of batch sharding. Restores the originals on
    exit. Does NOT touch ``randint`` / ``randn`` (PPGD sources are zeroed out via
    coeff=0 in isolate mode, so their RNG doesn't reach the dumped grad)."""

    def __enter__(self) -> "_PinRandToConstant":
        self._rand = torch.rand
        self._rand_like = torch.rand_like

        def const_rand(*size, **kw):  # noqa: ANN002, ANN003
            if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
                size = tuple(size[0])
            return torch.full(tuple(size), _DETERMINISTIC_U, **kw)

        def const_rand_like(t, **kw):  # noqa: ANN001, ANN003
            return torch.full_like(t, _DETERMINISTIC_U, **kw)

        torch.rand = const_rand  # pyright: ignore[reportAttributeAccessIssue]
        torch.rand_like = const_rand_like  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        torch.rand = self._rand  # pyright: ignore[reportAttributeAccessIssue]
        torch.rand_like = self._rand_like  # pyright: ignore[reportAttributeAccessIssue]


def _fixed_batch(batch_size: int, vocab_size: int, device: str) -> torch.Tensor:
    """Deterministic token batch — identical across modes / ranks for a given
    (batch_size, vocab_size). CPU tensor; trainers move it to device."""
    g = torch.Generator()
    g.manual_seed(1234)
    return torch.randint(0, vocab_size, (batch_size, SEQ_LEN), generator=g)


class _FixedBatchLoader:
    """Yields the same fixed batch forever. Mimics a DataLoader closely enough
    for the trainers' ``loop_dataloader`` (it just needs ``__iter__``)."""

    def __init__(self, batch: torch.Tensor):
        self._batch = batch

    def __iter__(self):
        while True:
            yield self._batch


def _topology(n_ci: int, n_per_block: int, n_ppgd: int, sites: list[str]) -> ThreePoolConfig:
    """Single LW block of n_per_block ranks (owns all sites), then CI, then PPGD.
    Rank 0 must be the LW block-0 leader (3-pool convention)."""
    lw_ranks = list(range(n_per_block))
    ci_ranks = list(range(n_per_block, n_per_block + n_ci))
    ppgd_ranks = list(range(n_per_block + n_ci, n_per_block + n_ci + n_ppgd))
    return ThreePoolConfig(
        ci_ranks=ci_ranks,
        layerwise_block_groups=[{"ranks": lw_ranks, "owned_sites": sites}],
        ppgd_ranks=ppgd_ranks,
        use_fused_kl=True,
    )


def _load_cfg(isolate_stoch: bool) -> LMExperimentConfig:
    cfg = LMExperimentConfig.from_file(BASE_YAML)
    loss_metrics = cfg.pd.loss_metrics
    if isolate_stoch:
        # Zero the imp + ppgd coefficients so the ONLY contribution to the CI-fn
        # gradient is the layerwise stochastic-recon seed — the exact term the
        # fix re-normalizes. This removes the two RNG/positioning confounds that
        # would otherwise prevent a clean 3p-vs-1p match even when correct:
        #   * PPGD adversarial sources are seeded per-rank and shaped per the
        #     rank-local batch, so they DIFFER between 1p and 3p. coeff_ppgd=0
        #     zeros the PPGD CI-grad seed regardless of source values.
        #   * imp is exact across the CI pool but its contribution is unrelated
        #     to the stoch fix; zeroing it keeps the comparison single-purpose.
        # With sampling="continuous" the stoch term itself is deterministic, so
        # the isolated CI grad is fully reproducible across both code paths.
        loss_metrics = [
            m.model_copy(update={"coeff": 0.0})
            if m.type in ("ImportanceMinimalityLoss", "PersistentPGDReconLoss")
            else m
            for m in loss_metrics
        ]
    # One step, no eval / save. Frac-of-training = 0 on the single step.
    cfg = cfg.model_copy(
        update={
            "pd": cfg.pd.model_copy(update={"steps": 1, "loss_metrics": loss_metrics}),
            "cadence": Cadence(train_log_every=1, save_every=None),
            "eval": None,
            "wandb": None,
        }
    )
    return cfg


def _vocab_size(target_model: nn.Module) -> int:
    emb = target_model.wte  # GPT2Simple token-embedding table [vocab, d]
    assert isinstance(emb, nn.Embedding)
    return emb.num_embeddings


def run_singlepool(out: Path, isolate_stoch: bool) -> None:
    from param_decomp_lab.distributed import get_device, init_distributed
    from param_decomp_lab.seed import set_seed

    init_distributed()  # registers DistributedState (torchrun sets WORLD_SIZE=1)
    cfg = _load_cfg(isolate_stoch)
    device = get_device()
    set_seed(cfg.pd.seed)
    target_model = build_target(cfg.target)
    pd_config = cfg.pd
    runtime_config = cfg.runtime.model_copy(update={"device": device, "dp": None})

    vocab = _vocab_size(target_model)
    batch = _fixed_batch(pd_config.batch_size, vocab, device)
    loader = _FixedBatchLoader(batch)

    trainer = Trainer(
        target_model=target_model,
        run_batch=make_run_batch(cfg.target),
        reconstruction_loss=recon_loss_kl,
        pd_config=pd_config,
        runtime_config=runtime_config,
    )
    # One step. steps=1 means the loop runs step 0 (compute + backward), then the
    # loop body at `step == pd_config.steps` is the final logging/checkpoint step
    # that SKIPS the gradient update — so the post-backward grad from step 0
    # survives on the params (AdamW.step ran on step 0 but does not zero grad;
    # zero_grad fires at the TOP of step 1 which we stop before). To be safe we
    # dump immediately after a manual single step instead of trusting loop edges.
    _single_step_and_dump(trainer, loader, out)


def _single_step_and_dump(trainer: Trainer, loader: DataLoader, out: Path) -> None:
    """Run exactly one forward+backward on the single-pool trainer and dump the
    CI fn grads BEFORE any clip / optimizer step. Replicates the relevant slice
    of ``Trainer.run`` step 0 to capture the grad at the exact comparison point."""
    from typing import cast

    from param_decomp.metrics.base import LossMetricConfig
    from param_decomp.optimize import (
        _build_metric_context,  # pyright: ignore[reportPrivateImportUsage]
    )
    from param_decomp.torch_helpers import bf16_autocast

    pd_config = trainer.pd_config
    runtime_config = trainer.runtime_config
    device = runtime_config.device
    it = iter(loader)

    trainer.components_optimizer.zero_grad()
    trainer.ci_fn_optimizer.zero_grad()

    with _PinRandToConstant(), bf16_autocast(enabled=runtime_config.autocast_bf16):
        ctx = _build_metric_context(
            next(it),
            step=0,
            is_eval=False,
            device=device,
            wrapped_model=trainer._wrapped_model,  # pyright: ignore[reportPrivateUsage]
            component_model=trainer.component_model,
            config=pd_config,
            reconstruction_loss=trainer.reconstruction_loss,
        )
        losses = {name: m.update(ctx) for name, m in trainer.loss_metrics.items()}

    total_loss = torch.zeros((), device=device)
    for metric_name, loss_val in losses.items():
        if loss_val is None:
            continue
        cfg_m = cast(LossMetricConfig, trainer.loss_metrics[metric_name].cfg)
        assert cfg_m.coeff is not None
        total_loss = total_loss + cfg_m.coeff * loss_val

    for metric_name, m in trainer.loss_metrics.items():
        m.before_backward(losses[metric_name])
    total_loss.backward()
    for m in trainer.loss_metrics.values():
        m.after_backward()

    _dump_ci_fn_grads(trainer._ci_fn_params, trainer.component_model, out)  # pyright: ignore[reportPrivateUsage]


def run_threepool(out: Path, n_ci: int, n_per_block: int, n_ppgd: int, isolate_stoch: bool) -> None:
    from param_decomp_lab.distributed import get_device, init_distributed
    from param_decomp_lab.seed import set_seed

    init_distributed()  # registers DistributedState + inits the process group
    rank = dist.get_rank()
    device = get_device()

    cfg = _load_cfg(isolate_stoch)
    set_seed(cfg.pd.seed)
    target_model = build_target(cfg.target)
    pd_config = cfg.pd
    runtime_config = cfg.runtime.model_copy(update={"device": device, "dp": None})
    sites = [t.module_pattern for t in pd_config.decomposition_targets]
    three_pool_config = _topology(n_ci, n_per_block, n_ppgd, sites)

    vocab = _vocab_size(target_model)
    batch = _fixed_batch(pd_config.batch_size, vocab, device)
    loader = _FixedBatchLoader(batch)

    trainer = ThreePoolTrainer(
        target_model=target_model,
        run_batch=make_run_batch(cfg.target),
        reconstruction_loss=recon_loss_kl,
        pd_config=pd_config,
        runtime_config=runtime_config,
        three_pool_config=three_pool_config,
    )

    _threepool_single_step_and_dump(trainer, loader, out, rank)
    dist.barrier()
    dist.destroy_process_group()


def _threepool_single_step_and_dump(
    trainer: ThreePoolTrainer, loader: DataLoader, out: Path, rank: int
) -> None:
    """Run one 3-pool step (all pools, fully comm-coupled) and have the FIRST CI
    rank dump the CI fn grad right after the in-pool AVG-reduce.

    We hook ``all_reduce_ci_fn_grads`` so the dump lands at the precise
    comparison point (post-reduce, pre-clip, pre-step) regardless of what
    ``step_ci`` does afterwards."""
    import param_decomp_lab.three_pool.step_ci as step_ci_mod
    from param_decomp_lab.three_pool.context import CIContext

    ctx = trainer.ctx
    is_first_ci = isinstance(ctx, CIContext) and rank == min(ctx.world.ci_ranks)

    orig_all_reduce = step_ci_mod.all_reduce_ci_fn_grads

    def _hooked(world, params):  # noqa: ANN001
        params = list(params)
        orig_all_reduce(world, params)
        if is_first_ci:
            _dump_ci_fn_grads(params, trainer.component_model, out)

    step_ci_mod.all_reduce_ci_fn_grads = _hooked  # pyright: ignore[reportAttributeAccessIssue]
    try:
        _run_one_threepool_step(trainer, loader)
    finally:
        step_ci_mod.all_reduce_ci_fn_grads = orig_all_reduce  # pyright: ignore[reportAttributeAccessIssue]


def _run_one_threepool_step(trainer: ThreePoolTrainer, loader: DataLoader) -> None:
    """Drive exactly one step of each pool's step fn — the slice of
    ``ThreePoolTrainer.run`` that does compute+backward for step 0, without the
    snapshot / eval / loader-prefetch bookkeeping."""
    from param_decomp.masks import AllLayersRouter
    from param_decomp.metrics.persistent_pgd_state import (
        PerBatchPerPositionScope,
        PersistentPGDState,
    )
    from param_decomp.torch_helpers import loop_dataloader
    from param_decomp_lab.three_pool.context import CIContext, LWContext, PPGDContext
    from param_decomp_lab.three_pool.optimize import (
        _seq_dims_from_batch,  # pyright: ignore[reportPrivateUsage]
    )
    from param_decomp_lab.three_pool.step_ci import step_ci
    from param_decomp_lab.three_pool.step_layerwise import step_layerwise
    from param_decomp_lab.three_pool.step_ppgd import step_ppgd

    ctx = trainer.ctx
    world = ctx.world
    runtime = trainer.runtime
    device = torch.device(trainer.runtime_config.device)

    it = loop_dataloader(loader)
    batch_T = next(it).to(device)

    if isinstance(ctx, PPGDContext) and trainer.ppgd_state is None:
        ppgd_cfg = runtime.ppgd_cfg
        assert isinstance(ppgd_cfg.scope, PerBatchPerPositionScope)
        trainer.ppgd_state = PersistentPGDState(
            module_to_c=runtime.c_per_site,
            batch_dims=(world.batch_local_ppgd, *_seq_dims_from_batch(batch_T)),
            device=device,
            use_delta_component=True,
            optimizer_cfg=ppgd_cfg.optimizer,
            scope=ppgd_cfg.scope,
            use_sigmoid_parameterization=ppgd_cfg.use_sigmoid_parameterization,
            n_warmup_steps=ppgd_cfg.n_warmup_steps,
            n_samples=ppgd_cfg.n_samples,
            router=AllLayersRouter(),
            reconstruction_loss=trainer.strategy.recon_loss,
        )

    n_steps = 1
    # Pin the stochastic-mask uniform draw to a constant in every pool so the
    # mask is identical to single-pool regardless of batch sharding (see
    # _PinRandToConstant). Every pool must apply it: the draw happens on LW
    # (stoch mask) and on PPGD (warmup sources, zeroed by coeff=0 but still
    # drawn) — wrapping uniformly keeps the RNG-side effects symmetric.
    with _PinRandToConstant():
        match ctx:
            case CIContext():
                assert trainer.optimizer is not None
                step_ci(
                    ctx,
                    trainer.component_model,
                    trainer.optimizer,
                    trainer._ci_fn_params,  # pyright: ignore[reportPrivateUsage]
                    batch_T=batch_T,
                    batch_T_plus_1=None,
                    h_cache_T=None,
                    cfg=runtime,
                    current_frac_of_training=0.0,
                    should_log=False,
                )
            case LWContext():
                assert trainer.optimizer is not None
                step_layerwise(
                    ctx,
                    trainer.component_model,
                    trainer.optimizer,
                    trainer._all_params,  # pyright: ignore[reportPrivateUsage]
                    batch_T,
                    runtime,
                    trainer.strategy,
                    should_log=False,
                )
            case PPGDContext():
                assert trainer.ppgd_state is not None
                step_ppgd(
                    ctx,
                    trainer.component_model,
                    trainer.ppgd_state,
                    batch_T,
                    runtime,
                    trainer.strategy,
                    step=0,
                    n_steps=n_steps,
                    should_log=False,
                )


def _dump_ci_fn_grads(
    ci_fn_params: list[nn.Parameter], component_model: nn.Module, out: Path
) -> None:
    assert component_model.ci_fn is not None  # pyright: ignore[reportAttributeAccessIssue]
    named = list(component_model.ci_fn.named_parameters())  # pyright: ignore[reportAttributeAccessIssue]
    assert len(named) == len(ci_fn_params)
    grads: dict[str, torch.Tensor] = {}
    for (name, p), p2 in zip(named, ci_fn_params, strict=True):
        assert p is p2, f"param order mismatch at {name}"
        assert p.grad is not None, f"no grad on ci_fn param {name}"
        grads[name] = p.grad.detach().float().cpu().clone()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(grads, out / "ci_fn_grads.pth")
    print(f"[grad_check] dumped {len(grads)} CI-fn grad tensors to {out}")


def compare(a: Path, b: Path) -> None:
    ga = torch.load(a / "ci_fn_grads.pth", weights_only=False)
    gb = torch.load(b / "ci_fn_grads.pth", weights_only=False)
    assert set(ga) == set(gb), f"param name mismatch:\n  {set(ga) ^ set(gb)}"
    flat_a = torch.cat([ga[k].flatten() for k in sorted(ga)])
    flat_b = torch.cat([gb[k].flatten() for k in sorted(gb)])
    abs_err = (flat_a - flat_b).abs()
    max_abs = abs_err.max().item()
    denom = flat_a.abs().max().item()
    max_rel = max_abs / denom if denom > 0 else float("nan")
    # Relative L2 over the whole grad vector — robust headline number.
    rel_l2 = (flat_a - flat_b).norm().item() / (flat_a.norm().item() + 1e-30)
    cos = torch.nn.functional.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()
    print(f"\n===== CI-fn grad compare: {a.name} vs {b.name} =====")
    print(f"  n params           : {len(ga)}")
    print(f"  n elements         : {flat_a.numel()}")
    print(f"  max |a|            : {denom:.6e}")
    print(f"  max abs err        : {max_abs:.6e}")
    print(f"  max abs err / max|a|: {max_rel:.6e}")
    print(f"  rel L2 (||a-b||/||a||): {rel_l2:.6e}")
    print(f"  cosine similarity  : {cos:.8f}")
    # Per-param worst offender
    worst_name, worst_val = "", -1.0
    for k in ga:
        e = (ga[k] - gb[k]).abs().max().item()
        if e > worst_val:
            worst_name, worst_val = k, e
    print(f"  worst param        : {worst_name} (max abs err {worst_val:.6e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=["singlepool", "threepool"], required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--n_ci", type=int, default=2)
    r.add_argument("--n_per_block", type=int, default=4)
    r.add_argument("--n_ppgd", type=int, default=2)
    r.add_argument(
        "--isolate-stoch",
        action="store_true",
        help="Zero imp + ppgd coeffs so the dumped CI grad is the stoch seed only.",
    )
    c = sub.add_parser("compare")
    c.add_argument("--a", type=Path, required=True)
    c.add_argument("--b", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "run":
        if args.mode == "singlepool":
            run_singlepool(args.out, args.isolate_stoch)
        else:
            run_threepool(args.out, args.n_ci, args.n_per_block, args.n_ppgd, args.isolate_stoch)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
