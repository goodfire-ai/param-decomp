"""Benchmark the base-DDP pd-lm training step (Track-2 §3.2 primary metric).

Reports wall-clock/step, tokens/sec, peak GPU memory, and a torch.profiler op
breakdown. Faithfulness warmup runs once up front and is NOT counted; eval is
excluded entirely. Single-process / one GPU by design — the primary metric is
measured on a pinned GPU at the config's fixed batch/seq.

    python -m param_decomp_lab.speedup.benchmark <config.yaml> \
        [--warmup_steps 10] [--measure_steps 30] [--profile_steps 5] [--out report.md]

Caveat: each `Trainer.run` call replays `self.step` data batches before resuming, so
the measure phase includes a small (`warmup_steps`-batch) dataloader replay before the
timed steps. Keep `warmup_steps` modest; the replay is data-only (no fwd/bwd) and small
relative to `measure_steps` of compute.
"""

import time
from pathlib import Path
from typing import Any

import fire
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from param_decomp.configs import Cadence
from param_decomp.optimize import Trainer
from param_decomp_lab.batch_and_loss_fns import recon_loss_kl
from param_decomp_lab.experiments.lm.run import (
    LMExperimentConfig,
    build_lm_loader,
    build_target,
    make_run_batch,
)
from param_decomp_lab.run_sink import RunSink

_NO_LOGGING = Cadence(train_log_every=10**9)


def _run_phase(
    trainer: Trainer, loader: DataLoader[Any], sink: RunSink, *, until_step: int
) -> None:
    """Advance the trainer to `until_step` with no eval and no logging."""
    trainer.pd_config = trainer.pd_config.model_copy(update={"steps": until_step})
    trainer.run(loader, sink, _NO_LOGGING, eval_loop=None)


def benchmark(
    config_path: str,
    *,
    warmup_steps: int = 10,
    measure_steps: int = 30,
    profile_steps: int = 5,
    out: str | None = None,
) -> None:
    assert torch.cuda.is_available(), "benchmark requires a CUDA device"
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)

    cfg = LMExperimentConfig.from_file(Path(config_path))
    cfg = cfg.model_copy(
        update={"runtime": cfg.runtime.model_copy(update={"device": device, "dp": None})}
    )

    target_model = build_target(cfg.target)
    loader = build_lm_loader(
        cfg.target,
        cfg.data,
        split="train",
        device=device,
        batch_size=cfg.pd.batch_size,
        dist_state=None,
        seed=cfg.pd.seed,
    )
    sink = RunSink.silent()
    trainer = Trainer(
        target_model=target_model,
        run_batch=make_run_batch(cfg.target),
        reconstruction_loss=recon_loss_kl,
        pd_config=cfg.pd,
        runtime_config=cfg.runtime,
    )

    # Phase 1: faithfulness warmup (once) + warmup_steps to reach steady state. Untimed.
    _run_phase(trainer, loader, sink, until_step=warmup_steps)

    # Phase 2: timed steady-state steps.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    _run_phase(trainer, loader, sink, until_step=warmup_steps + measure_steps)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    per_step_ms = 1000.0 * elapsed / measure_steps
    tokens_per_step = cfg.pd.batch_size * cfg.data.max_seq_len
    tokens_per_sec = tokens_per_step / (elapsed / measure_steps)

    # Phase 3: profiler op breakdown over a few steps.
    op_table = "(profiling skipped: profile_steps=0)"
    trace_path: Path | None = None
    if profile_steps > 0:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            _run_phase(
                trainer,
                loader,
                sink,
                until_step=warmup_steps + measure_steps + profile_steps,
            )
        op_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
        trace_dir = Path(out).parent if out is not None else Path.cwd()
        trace_path = trace_dir / "bench_trace.json"
        prof.export_chrome_trace(str(trace_path))

    report = _format_report(
        config_path=config_path,
        gpu_name=gpu_name,
        batch_size=cfg.pd.batch_size,
        max_seq_len=cfg.data.max_seq_len,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        profile_steps=profile_steps,
        per_step_ms=per_step_ms,
        tokens_per_sec=tokens_per_sec,
        peak_gb=peak_gb,
        op_table=op_table,
        trace_path=trace_path,
    )
    print(report)
    if out is not None:
        Path(out).write_text(report)
        print(f"\nWrote report to {out}")


def _format_report(
    *,
    config_path: str,
    gpu_name: str,
    batch_size: int,
    max_seq_len: int,
    warmup_steps: int,
    measure_steps: int,
    profile_steps: int,
    per_step_ms: float,
    tokens_per_sec: float,
    peak_gb: float,
    op_table: str,
    trace_path: Path | None,
) -> str:
    trace_line = f"- profiler trace: `{trace_path}`\n" if trace_path is not None else ""
    return (
        f"## pd-lm step benchmark\n\n"
        f"- config: `{config_path}`\n"
        f"- GPU (pinned): {gpu_name}\n"
        f"- batch_size: {batch_size}, max_seq_len: {max_seq_len}, "
        f"tokens/step: {batch_size * max_seq_len}\n"
        f"- warmup/measure/profile steps: {warmup_steps}/{measure_steps}/{profile_steps}\n\n"
        f"- **wall-clock / step: {per_step_ms:.2f} ms**\n"
        f"- **tokens / sec: {tokens_per_sec:,.0f}**\n"
        f"- **peak GPU mem: {peak_gb:.2f} GB**\n"
        f"{trace_line}\n"
        f"### Top ops by self-CUDA time\n\n```\n{op_table}\n```\n"
    )


def cli() -> None:
    fire.Fire(benchmark)


if __name__ == "__main__":
    cli()
