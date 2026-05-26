"""``PhaseProfiler`` for 3-pool training.

Same machinery as ``two_pool.profiler.PhaseProfiler`` (CPU + CUDA events into
a Chrome trace JSON readable by Perfetto), parameterized on the 3-pool literal
``"ci" | "layerwise" | "ppgd"`` instead of ``"a" | "b"``. Duplicated rather
than imported to keep each subsystem owning its own surface.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.profiler

from param_decomp._trace import phase_trace_enabled, trace


@dataclass
class _PendingPhase:
    """Buffered phase-exit info waiting on a CUDA event sync.

    ``entry_event``/``exit_event`` are recorded on the current stream at phase
    enter/exit. ``elapsed_time`` on these blocks until both events are done, so
    we defer the call to ``flush_pending_gpu_events`` (after a stream sync).
    """

    name: str
    cpu_ms: float
    peak_gb: float
    end_gb: float
    delta_gb: float
    entry_event: torch.cuda.Event
    exit_event: torch.cuda.Event


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
    _pending_gpu_events: list[_PendingPhase] = field(default_factory=list)

    def __enter__(self) -> "PhaseProfiler":
        if not self.enabled:
            return self
        assert self.out_dir is not None, "PhaseProfiler enabled but no out_dir given"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.out_dir / f"trace_{self.pool}_rank{self.rank}.json"

        def on_trace_ready(prof: "torch.profiler.profile") -> None:
            prof.export_chrome_trace(str(trace_path))
            print(f"[profiler rank{self.rank}] wrote {trace_path}", flush=True)

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
        cpu_start = 0.0
        entry_event: torch.cuda.Event | None = None
        if do_trace:
            device = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(device)
            before_gb = torch.cuda.memory_allocated(device) / 1e9
            trace(f"phase: {name} cur={before_gb:.2f}gb")
            entry_event = torch.cuda.Event(enable_timing=True)
            entry_event.record()
            cpu_start = time.perf_counter()

        inner = torch.profiler.record_function(name) if self.enabled else nullcontext()
        try:
            with inner:
                yield
        finally:
            if do_trace:
                assert entry_event is not None
                exit_event = torch.cuda.Event(enable_timing=True)
                exit_event.record()
                cpu_ms = (time.perf_counter() - cpu_start) * 1000.0
                peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                end_gb = torch.cuda.memory_allocated(device) / 1e9
                self._pending_gpu_events.append(
                    _PendingPhase(
                        name=name,
                        cpu_ms=cpu_ms,
                        peak_gb=peak_gb,
                        end_gb=end_gb,
                        delta_gb=end_gb - before_gb,
                        entry_event=entry_event,
                        exit_event=exit_event,
                    )
                )

    def flush_pending_gpu_events(self) -> None:
        """Drain buffered phase exits, emit one trace line per phase.

        ``cuda.Event.elapsed_time`` blocks until both events are recorded on the
        device, so callers should invoke this after a stream sync (e.g. after
        ``torch.cuda.current_stream().synchronize()`` at end-of-step) to avoid
        adding a hidden per-phase sync on the critical path.
        """
        if not self._pending_gpu_events:
            return
        for ph in self._pending_gpu_events:
            gpu_ms = ph.entry_event.elapsed_time(ph.exit_event)
            wait_ms = ph.cpu_ms - gpu_ms
            trace(
                f"phase: {ph.name} end peak={ph.peak_gb:.2f}gb "
                f"end={ph.end_gb:.2f}gb delta={ph.delta_gb:+.2f}gb "
                f"cpu={ph.cpu_ms:.1f}ms gpu={gpu_ms:.1f}ms wait={wait_ms:+.1f}ms"
            )
        self._pending_gpu_events.clear()

    def step(self) -> None:
        """Advance the profiler schedule (call once per training iteration)."""
        if self._prof is not None:
            self._prof.step()
