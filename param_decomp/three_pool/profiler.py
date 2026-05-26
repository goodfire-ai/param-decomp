"""``PhaseProfiler`` for 3-pool training.

Same machinery as ``two_pool.profiler.PhaseProfiler`` (CPU + CUDA events into
a Chrome trace JSON readable by Perfetto), parameterized on the 3-pool literal
``"ci" | "layerwise" | "ppgd"`` instead of ``"a" | "b"``. Duplicated rather
than imported to keep each subsystem owning its own surface.
"""

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.profiler

from param_decomp._trace import phase_trace_enabled, trace


@dataclass
class PhaseProfiler:
    """Thin wrapper around ``torch.profiler.profile`` for 3-pool training.

    Records CPU + CUDA events into a Chrome trace JSON that loads directly into
    Perfetto. Captures kernel timings, NCCL ops, and GPU memory allocations.

    Use as a context manager around the training loop, with ``phase(name)``
    annotations marking logical step phases ("ci/1_ci_fn_fwd", "lw/3_layerwise",
    "pgd/4_ppgd_warmup", etc.) inside. Call ``step()`` once per training
    iteration so the schedule progresses.

    Trace schedule by default skips the first 20 iters (CUDA / JIT warmup) then
    records 3 active steps and dumps the trace.
    """

    enabled: bool = False
    out_dir: Path | None = None
    rank: int = 0
    pool: Literal["ci", "layerwise", "ppgd"] = "ci"
    skip_first: int = 20
    active: int = 3
    _prof: torch.profiler.profile | None = None

    def __enter__(self) -> "PhaseProfiler":
        if not self.enabled:
            return self
        assert self.out_dir is not None, "PhaseProfiler enabled but no out_dir given"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.out_dir / f"trace_{self.pool}_rank{self.rank}.json"

        summary_path = self.out_dir / f"key_avgs_{self.pool}_rank{self.rank}.txt"

        def on_trace_ready(prof: "torch.profiler.profile") -> None:
            prof.export_chrome_trace(str(trace_path))
            table = prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=40,
                max_name_column_width=80,
            )
            summary_path.write_text(table)
            print(f"[profiler rank{self.rank}] wrote {trace_path}", flush=True)
            print(f"[profiler rank{self.rank}] wrote {summary_path}", flush=True)
            print(
                f"[profiler rank{self.rank}] view trace: drop the json into "
                "https://www.speedscope.app/ (or chrome://tracing/, or Perfetto)",
                flush=True,
            )

        self._prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=self.skip_first, wait=0, warmup=1, active=self.active, repeat=1
            ),
            on_trace_ready=on_trace_ready,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        )
        self._prof.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._prof is not None:
            self._prof.__exit__(exc_type, exc, tb)
            self._prof = None

    @contextmanager
    def phase(self, name: str) -> "Iterator[None]":
        """Annotate a logical step phase.

        When ``PD_PHASE_TRACE=1``, emits entry + exit traces tagged with
        memory deltas. Entry: ``cur=X.XXgb``. Exit: ``peak=X.XXgb end=X.XXgb
        delta=±X.XXgb``. ``peak`` is the within-phase peak (we
        ``reset_peak_memory_stats`` at entry); ``delta`` is end-current minus
        entry-current, which captures whether the phase persistently allocates.

        Note: resetting peak per phase makes any concurrent reader of
        ``max_memory_allocated`` see only this phase's peak, not the
        whole-step peak. That's fine in debug mode (where this is enabled);
        in production ``PD_PHASE_TRACE`` is off and this is a no-op.

        The ``record_function`` wrapper is only active when the torch
        profiler is enabled.
        """
        do_trace = phase_trace_enabled()
        device = None
        before_gb = 0.0
        if do_trace:
            device = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(device)
            before_gb = torch.cuda.memory_allocated(device) / 1e9
            trace(f"phase: {name} cur={before_gb:.2f}gb")

        inner = torch.profiler.record_function(name) if self.enabled else nullcontext()
        try:
            with inner:
                yield
        finally:
            if do_trace:
                peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                end_gb = torch.cuda.memory_allocated(device) / 1e9
                trace(
                    f"phase: {name} end peak={peak_gb:.2f}gb "
                    f"end={end_gb:.2f}gb delta={end_gb - before_gb:+.2f}gb"
                )

    def step(self) -> None:
        """Advance the profiler schedule (call once per training iteration)."""
        if self._prof is not None:
            self._prof.step()
