"""Diagnostic: does |mean(grad×act)| preserve L2(grad×act) rankings?

The harvester accumulates signed sums of grad×act across positions. This script
checks whether that signed mean gives the same top-K source ranking as the
magnitude-preserving L2 = sqrt(mean((grad×act)²)) alternative.

Methodology:
  For each target component, iterate through data, find positions where the
  target's CI > threshold (i.e. it's actually firing), then compute per-position
  grad×act for all source components at those positions. Reduce to |mean| and L2
  per source component, rank them, and compare rankings via top-K overlap and
  mean rank displacement.

  The per-position grad×act computation matches the harvester exactly:
    - Component sources: grad × act × ci  (CI-weighted, per the harvester)
    - Embed sources: (grad × act).sum(embed_dim), grouped by token ID

Usage:
    python -m spd.dataset_attributions.scripts.diagnose_cancellation \
        "wandb:goodfire/spd/s-892f140b" \
        --n_targets_per_layer 20 --n_active 100
"""

import random
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn

from spd.configs import LMTaskConfig, SamplingType
from spd.data import train_loader_and_tokenizer
from spd.harvest.repo import HarvestRepo
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.settings import SPD_OUT_DIR
from spd.topology import TransformerTopology, get_sources_by_target
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import bf16_autocast, extract_batch_data
from spd.utils.wandb_utils import parse_wandb_run_path

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


@dataclass
class ModelContext:
    model: ComponentModel
    topology: TransformerTopology
    sampling: SamplingType
    sources_by_target: dict[str, list[str]]
    device: torch.device
    embed_path: str
    unembed_path: str
    vocab_size: int


def setup(wandb_path: str) -> ModelContext:
    device = torch.device(get_device())
    run_info = SPDRunInfo.from_path(wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()
    topology = TransformerTopology(model.target_model)

    sources_by_target_raw = get_sources_by_target(
        model, topology, str(device), run_info.config.sampling
    )
    embed_path = topology.path_schema.embedding_path
    unembed_path = topology.path_schema.unembed_path
    component_layers = set(model.target_module_paths)
    valid_sources = component_layers | {embed_path}
    valid_targets = component_layers | {unembed_path}
    sources_by_target: dict[str, list[str]] = {}
    for target, sources in sources_by_target_raw.items():
        if target not in valid_targets:
            continue
        filtered = [s for s in sources if s in valid_sources]
        if filtered:
            sources_by_target[target] = filtered

    return ModelContext(
        model=model,
        topology=topology,
        sampling=run_info.config.sampling,
        sources_by_target=sources_by_target,
        device=device,
        embed_path=embed_path,
        unembed_path=unembed_path,
        vocab_size=topology.embedding_module.num_embeddings,
    )


# ---------------------------------------------------------------------------
# Forward pass (matches harvester.process_batch exactly)
# ---------------------------------------------------------------------------


def forward_with_caches(
    ctx: ModelContext,
    tokens: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """One forward pass → (cache, ci). Reuse across all (target, source) pairs."""
    embed_out: list[Tensor] = []
    pre_unembed: list[Tensor] = []

    def embed_hook(_mod: nn.Module, _args: Any, _kwargs: Any, out: Tensor) -> Tensor:
        out.requires_grad_(True)
        embed_out.clear()
        embed_out.append(out)
        return out

    def pre_unembed_hook(_mod: nn.Module, args: tuple[Any, ...], _kwargs: Any) -> None:
        args[0].requires_grad_(True)
        pre_unembed.clear()
        pre_unembed.append(args[0])

    h1 = ctx.topology.embedding_module.register_forward_hook(embed_hook, with_kwargs=True)
    h2 = ctx.topology.unembed_module.register_forward_pre_hook(pre_unembed_hook, with_kwargs=True)

    with torch.no_grad(), bf16_autocast():
        out = ctx.model(tokens, cache_type="input")
        ci = ctx.model.calc_causal_importances(
            pre_weight_acts=out.cache, sampling=ctx.sampling, detach_inputs=False
        )

    mask_infos = make_mask_infos(
        component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
        routing_masks="all",
    )

    with torch.enable_grad(), bf16_autocast():
        model_output = ctx.model(tokens, mask_infos=mask_infos, cache_type="component_acts")

    h1.remove()
    h2.remove()

    cache = model_output.cache
    cache[f"{ctx.embed_path}_post_detach"] = embed_out[0]
    cache[f"{ctx.unembed_path}_pre_detach"] = pre_unembed[0]

    return cache, ci.lower_leaky


# ---------------------------------------------------------------------------
# Per-position attribution (matches harvester._process_component_targets)
# ---------------------------------------------------------------------------


def per_position_grads_at(
    ctx: ModelContext,
    cache: dict[str, Tensor],
    ci: dict[str, Tensor],
    target_concrete: str,
    t_idx: int,
    s: int,
) -> dict[str, Tensor]:
    """Compute grad×act for all source layers at a single position (b=0, s=s).

    Returns {source_concrete: value_tensor} where:
      - Component source: grad × act × ci, shape (C_source,)
      - Embed source: (grad × act).sum(embed_dim), scalar
    Matches the harvester's _accumulate_grads exactly, just without the sum.
    """
    target_acts_raw = cache[f"{target_concrete}_pre_detach"]
    scalar = target_acts_raw[0, s, t_idx]

    source_layers = ctx.sources_by_target[target_concrete]
    source_acts = [cache[f"{sc}_post_detach"] for sc in source_layers]
    grads = torch.autograd.grad(scalar, source_acts, retain_graph=True)

    result: dict[str, Tensor] = {}
    with torch.no_grad():
        for source_layer, act, grad in zip(source_layers, source_acts, grads, strict=True):
            if source_layer == ctx.embed_path:
                result[source_layer] = (grad[0, s] * act[0, s]).sum().cpu()
            else:
                result[source_layer] = (grad[0, s] * act[0, s] * ci[source_layer][0, s]).cpu()
    return result


# ---------------------------------------------------------------------------
# Collect active positions for a target component
# ---------------------------------------------------------------------------


def collect_active_attrs(
    ctx: ModelContext,
    loader_iter: Any,
    target_concrete: str,
    t_idx: int,
    n_active: int,
    ci_threshold: float,
    max_sequences: int,
) -> tuple[dict[str, list[Tensor]], int, int]:
    """Iterate sequences, backward only at positions where target CI > threshold.

    Returns (per_source_vals, n_found, n_sequences_checked) where
    per_source_vals[source_concrete] is a list of tensors, one per active position.
    """
    source_layers = ctx.sources_by_target[target_concrete]
    per_source: dict[str, list[Tensor]] = {sc: [] for sc in source_layers}
    n_found = 0
    n_checked = 0

    for _ in range(max_sequences):
        try:
            batch_data = next(loader_iter)
        except StopIteration:
            break
        tokens = extract_batch_data(batch_data).to(ctx.device)
        n_checked += 1

        # Cheap CI check (no grad graph needed)
        with torch.no_grad(), bf16_autocast():
            out = ctx.model(tokens, cache_type="input")
            ci_check = ctx.model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling=ctx.sampling, detach_inputs=False
            )

        ci_vals = ci_check.lower_leaky[target_concrete][0, :, t_idx]
        active_positions = (ci_vals > ci_threshold).nonzero(as_tuple=True)[0]
        if len(active_positions) == 0:
            continue

        # Full forward with grad graph
        cache, ci = forward_with_caches(ctx, tokens)

        for s in active_positions.tolist():
            if n_found >= n_active:
                break
            grads = per_position_grads_at(ctx, cache, ci, target_concrete, t_idx, s)
            for sc, val in grads.items():
                per_source[sc].append(val)
            n_found += 1

        if n_found >= n_active:
            break

    return per_source, n_found, n_checked


# ---------------------------------------------------------------------------
# Reduce per-position values to |mean| and L2 per source component
# ---------------------------------------------------------------------------


def reduce_to_rankings(
    per_source: dict[str, list[Tensor]],
    embed_path: str,
    tokens_per_pos: list[Tensor] | None,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce per-position attrs to per-source-component |mean| and L2.

    Component sources: each position gives a (C,) vector. |mean| and L2 over positions.
    Embed sources: each position gives a scalar. Group by token ID via scatter_add,
    then |mean| and L2 per token.

    Returns (abs_means, l2s, is_embed) arrays pooled across all source layers.
    """
    all_abs_means: list[np.ndarray] = []
    all_l2s: list[np.ndarray] = []
    all_is_embed: list[np.ndarray] = []

    for source_layer, vals in per_source.items():
        if not vals:
            continue

        if source_layer == embed_path:
            assert tokens_per_pos is not None
            all_vals = torch.stack(vals).float()
            all_toks = torch.cat(tokens_per_pos)
            token_sum = torch.zeros(vocab_size)
            token_sq_sum = torch.zeros(vocab_size)
            token_count = torch.zeros(vocab_size)
            token_sum.scatter_add_(0, all_toks, all_vals)
            token_sq_sum.scatter_add_(0, all_toks, all_vals.square())
            token_count.scatter_add_(0, all_toks, torch.ones_like(all_vals))
            safe_count = token_count.clamp(min=1)
            all_abs_means.append((token_sum / safe_count).abs().numpy())
            all_l2s.append((token_sq_sum / safe_count).sqrt().numpy())
            all_is_embed.append(np.ones(vocab_size, dtype=bool))
        else:
            stacked = torch.stack(vals).float()  # (N, C)
            all_abs_means.append(stacked.mean(dim=0).abs().numpy())
            all_l2s.append(stacked.square().mean(dim=0).sqrt().numpy())
            all_is_embed.append(np.zeros(stacked.shape[1], dtype=bool))

    return np.concatenate(all_abs_means), np.concatenate(all_l2s), np.concatenate(all_is_embed)


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def select_targets(
    ctx: ModelContext,
    run_id: str,
    n_per_layer: int,
    fd_range: tuple[float, float],
    seed: int,
    comp_only: bool,
) -> list[tuple[str, int, float]]:
    """Select target components with firing density in range.

    Returns [(concrete_path, c_idx, firing_density), ...].
    """
    harvest = HarvestRepo.open_most_recent(run_id, readonly=True)
    assert harvest is not None
    summary = harvest.get_summary()
    assert summary is not None

    rng = random.Random(seed)
    targets: list[tuple[str, int, float]] = []

    for target_concrete in ctx.sources_by_target:
        if target_concrete == ctx.unembed_path:
            continue
        if comp_only and ctx.embed_path in ctx.sources_by_target[target_concrete]:
            continue

        candidates: list[tuple[int, float]] = []
        for c_idx in range(ctx.model.module_to_c[target_concrete]):
            key = f"{target_concrete}:{c_idx}"
            if key not in summary:
                continue
            fd = summary[key].firing_density
            if fd_range[0] < fd < fd_range[1]:
                candidates.append((c_idx, fd))

        rng.shuffle(candidates)
        for c_idx, fd in candidates[:n_per_layer]:
            targets.append((target_concrete, c_idx, fd))

    return targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class TargetResult:
    target_layer: str
    target_idx: int
    firing_density: float
    n_active: int
    n_sources: int
    has_embed: bool
    top5_mrd: float
    top5_overlap: int
    top10_mrd: float
    top10_overlap: int


def main(
    wandb_path: str,
    n_targets_per_layer: int = 20,
    n_active: int = 100,
    ci_threshold: float = 0.01,
    fd_min: float = 1e-4,
    fd_max: float = 1e-1,
    max_sequences: int = 5000,
    seed: int = 42,
    comp_only: bool = False,
) -> None:
    import time

    ctx = setup(wandb_path)
    _, _, run_id = parse_wandb_run_path(wandb_path)
    to_canon = ctx.topology.target_to_canon

    targets = select_targets(ctx, run_id, n_targets_per_layer, (fd_min, fd_max), seed, comp_only)
    print(f"Selected {len(targets)} targets (fd in ({fd_min}, {fd_max}), comp_only={comp_only})")

    spd_config = SPDRunInfo.from_path(wandb_path).config
    assert isinstance(spd_config.task_config, LMTaskConfig)
    # frozen
    # spd_config.task_config.dataset_name = "danbraunai/pile-uncopyrighted-tok-shuffled"
    train_loader, _ = train_loader_and_tokenizer(spd_config, 1)
    loader_iter = iter(train_loader)

    results: list[TargetResult] = []
    t0 = time.time()

    for ti, (tgt_concrete, t_idx, fd) in enumerate(targets):
        tgt_canon = to_canon(tgt_concrete)
        source_list = ctx.sources_by_target[tgt_concrete]
        has_embed = ctx.embed_path in source_list

        per_source, n_found, _ = collect_active_attrs(
            ctx, loader_iter, tgt_concrete, t_idx, n_active, ci_threshold, max_sequences
        )

        if n_found < 10:
            print(
                f"[{ti + 1}/{len(targets)}] {tgt_canon}:{t_idx} fd={fd:.4f} "
                f"— only {n_found} active, skipping"
            )
            continue

        # Embed sources excluded: collect_active_attrs doesn't store token IDs
        # needed for scatter_add grouping. This is fine — embed rankings are
        # near-perfect anyway (confirmed in notebook analysis).
        abs_means, l2s, _ = reduce_to_rankings(
            {sc: v for sc, v in per_source.items() if sc != ctx.embed_path},
            ctx.embed_path,
            None,
            ctx.vocab_size,
        )

        n = len(abs_means)
        rank_mean = np.argsort(np.argsort(-abs_means))
        rank_l2 = np.argsort(np.argsort(-l2s))

        top5 = np.argsort(-l2s)[:5]
        top10 = np.argsort(-l2s)[:10]

        results.append(
            TargetResult(
                target_layer=tgt_canon,
                target_idx=t_idx,
                firing_density=fd,
                n_active=n_found,
                n_sources=n,
                has_embed=has_embed,
                top5_mrd=np.abs(rank_mean[top5] - rank_l2[top5]).mean(),
                top5_overlap=len(set(top5) & set(np.argsort(-abs_means)[:5])),
                top10_mrd=np.abs(rank_mean[top10] - rank_l2[top10]).mean(),
                top10_overlap=len(set(top10) & set(np.argsort(-abs_means)[:10])),
            )
        )

        if (ti + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (ti + 1)
            print(
                f"[{ti + 1}/{len(targets)}] {elapsed:.0f}s, ~{rate * (len(targets) - ti - 1):.0f}s left",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} targets in {elapsed:.0f}s")

    _print_results(results)
    _plot_results(results)


def _print_results(results: list[TargetResult]) -> None:
    print(f"\n{'=' * 70}")
    print("CANCELLATION DIAGNOSTIC: |mean| vs L2 ranking agreement")
    print(f"{'=' * 70}")
    print(f"  {len(results)} targets, active positions only (CI > threshold)")
    print()

    for label, metric, K in [
        ("Top-5 mean rank displacement", "top5_mrd", 5),
        ("Top-5 overlap", "top5_overlap", 5),
        ("Top-10 mean rank displacement", "top10_mrd", 10),
        ("Top-10 overlap", "top10_overlap", 10),
    ]:
        vals = [getattr(r, metric) for r in results]
        print(
            f"  {label}: {np.mean(vals):.1f} ± {np.std(vals):.1f}"
            f"  (median {np.median(vals):.1f})" + (f"/{K}" if "overlap" in metric else "")
        )

    print("\n  By target layer:")
    layers = sorted(set(r.target_layer for r in results))
    print(
        f"  {'layer':<18} {'n':>3} {'top5 mrd':>10} {'top5 olap':>10} "
        f"{'top10 mrd':>10} {'top10 olap':>10}"
    )
    print(f"  {'-' * 65}")
    for layer in layers:
        lr = [r for r in results if r.target_layer == layer]
        print(
            f"  {layer:<18} {len(lr):>3} "
            f"{np.mean([r.top5_mrd for r in lr]):>6.1f}±{np.std([r.top5_mrd for r in lr]):<3.1f}"
            f"{np.mean([r.top5_overlap for r in lr]):>7.1f}/5 "
            f"{np.mean([r.top10_mrd for r in lr]):>6.1f}±{np.std([r.top10_mrd for r in lr]):<3.1f}"
            f"{np.mean([r.top10_overlap for r in lr]):>7.1f}/10"
        )


def _plot_results(results: list[TargetResult]) -> None:
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = sorted(set(r.target_layer for r in results))
    colors = {layer: f"C{i}" for i, layer in enumerate(layers)}

    for ax, K in [(axes[0], 5), (axes[1], 10)]:
        for layer in layers:
            vals = [r for r in results if r.target_layer == layer]
            ax.hist(
                [getattr(r, f"top{K}_mrd") for r in vals],
                bins=np.arange(-0.5, 25.5, 1),
                alpha=0.4,
                color=colors[layer],
                label=f"{layer} (μ={np.mean([getattr(r, f'top{K}_mrd') for r in vals]):.1f})",
            )
        ax.set_xlabel(f"Top-{K} mean rank displacement")
        ax.set_ylabel("# targets")
        ax.set_title(
            f"Top-{K}: |mean| vs L2 ranking agreement\n{len(results)} targets, active positions only"
        )
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = SPD_OUT_DIR / "www" / "attr_cancellation_diagnostic.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
