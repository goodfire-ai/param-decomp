"""Benchmark the AutointerpLabels slow eval metric against a real decomposition.

Loads a saved PD run, drives the metric over N eval batches (mirroring the async
slow-eval pass), and reports:
  * micro-harvest throughput (tokens/s), split into shared ctx-build vs the metric's
    own update() overhead
  * compute() wall time (the concurrent LLM interpretation) and per-component cost
  * a sample of the produced labels

Run on a single GPU, e.g. on an existing dev allocation:
    srun --jobid=<dev_job> --overlap \
      python scripts/standalone_repros/bench_autointerp_labels.py --run_id p-73cf27e4 \
      --k 16 --n_batches 8 --batch_size 16
"""

import time

import fire
import torch

from param_decomp.optimize import _build_metric_context
from param_decomp.torch_helpers import bf16_autocast, loop_dataloader
from param_decomp_lab.adapters.pd import PDAdapter
from param_decomp_lab.autointerp.config import CompactSkepticalConfig, DualViewConfig
from param_decomp_lab.autointerp.providers import (
    AnthropicHaiku45LLMConfig,
    GoogleAILLMConfig,
    OpenRouterLLMConfig,
)
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.eval_metrics.autointerp_labels import (
    AutointerpLabels,
    AutointerpLabelsConfig,
)

_PROVIDERS = {
    "openrouter": OpenRouterLLMConfig(reasoning_effort="none"),
    "haiku": AnthropicHaiku45LLMConfig(),
    "gemini": GoogleAILLMConfig(),
}
_STRATEGIES = {
    "compact_skeptical": CompactSkepticalConfig(),
    "dual_view": DualViewConfig(),
}


def main(
    run_id: str = "p-73cf27e4",
    *,
    k: int = 16,
    n_batches: int = 8,
    batch_size: int = 16,
    max_examples: int = 20,
    context_tokens_per_side: int = 10,
    activation_threshold: float = 0.1,
    seed: int = 0,
    provider: str = "openrouter",
    strategy: str = "compact_skeptical",
    skip_llm: bool = False,
) -> None:
    device = "cuda"
    adapter = PDAdapter(run_id)
    cfg = adapter.pd_run.cfg
    seq_len = cfg.data.max_seq_len

    print(f"loading {run_id} ({cfg.target.spec.model_name}) ...", flush=True)
    t_load = time.perf_counter()
    component_model = adapter.component_model.to(device).eval()
    print(f"  loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    metric = AutointerpLabels(
        AutointerpLabelsConfig(
            k=k,
            seed=seed,
            activation_threshold=activation_threshold,
            max_examples=max_examples,
            context_tokens_per_side=context_tokens_per_side,
            llm=_PROVIDERS[provider],
            template_strategy=_STRATEGIES[strategy],
            dataset_name=cfg.data.dataset_name,
            seq_len=seq_len,
            tokenizer_name=adapter.tokenizer_name,
        )
    )
    metric.bind(model=component_model, device=device)

    loader = adapter.dataloader(batch_size)
    it = loop_dataloader(loader)
    weight_deltas = component_model.calc_weight_deltas()
    metric.reset()

    ctx_times: list[float] = []
    update_times: list[float] = []
    with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
        for _ in range(n_batches):
            batch = next(it)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ctx = _build_metric_context(
                batch,
                step=0,
                is_eval=True,
                device=device,
                wrapped_model=component_model,
                component_model=component_model,
                config=cfg.pd,
                reconstruction_loss=recon_loss_kl,
                weight_deltas=weight_deltas,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            metric.update(ctx)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            ctx_times.append(t1 - t0)
            update_times.append(t2 - t1)

    tokens = n_batches * batch_size * seq_len
    harvest_total = sum(ctx_times) + sum(update_times)
    print("\n=== micro-harvest ===", flush=True)
    print(
        f"  config: k={k} n_batches={n_batches} batch={batch_size} seq={seq_len} "
        f"max_examples={max_examples} ctx_per_side={context_tokens_per_side}"
    )
    print(f"  tokens processed:     {tokens:,}")
    print(f"  ctx-build (shared):   {sum(ctx_times) * 1e3 / n_batches:7.1f} ms/batch")
    print(f"  update (metric only): {sum(update_times) * 1e3 / n_batches:7.1f} ms/batch")
    print(f"  harvest wall:         {harvest_total:6.2f}s  ->  {tokens / harvest_total:,.0f} tok/s")
    print(
        f"  update-only tput:     {tokens / sum(update_times):,.0f} tok/s "
        f"(marginal cost of adding this metric)"
    )

    if skip_llm:
        print("\n(skip_llm) done.")
        return

    print("\n=== compute() / LLM interpretation ===", flush=True)
    t0 = time.perf_counter()
    result = metric.compute()
    compute_t = time.perf_counter() - t0
    table = result["labels"]
    rows = table.data
    print(f"  provider={provider} strategy={strategy}")
    print(f"  components labelled:  {len(rows)} / {k} requested")
    print(
        f"  compute wall:         {compute_t:6.2f}s  ({compute_t / max(len(rows), 1):.2f}s/component, concurrent)"
    )
    print("\n  sample labels:")
    for comp, label, _reasoning in rows[:12]:
        print(f"    {comp:24s}  {label}")


if __name__ == "__main__":
    fire.Fire(main)
