"""``PhaseProfiler``: ``torch.profiler.profile`` wrapper for 2-pool training.

Captures CPU + CUDA events into a Chrome trace JSON that loads directly into
Perfetto (https://ui.perfetto.dev). Use as a context manager around the
training loop with ``phase(name)`` annotations marking logical step phases.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.profiler


@dataclass
class PhaseProfiler:
    """Thin wrapper around ``torch.profiler.profile`` for 2-pool training.

    Records CPU + CUDA events into a Chrome trace JSON that loads directly into
    Perfetto (https://ui.perfetto.dev). Captures kernel timings, NCCL ops, and
    GPU memory allocations.

    Use as a context manager around the training loop, with ``phase(name)``
    annotations marking logical step phases ("a/5_layerwise" etc.) inside.
    Call ``step()`` once per training iteration so the schedule progresses.

    Trace schedule by default skips the first 20 iters (CUDA / JIT warmup) then
    records 3 active steps and dumps the trace.
    """

    enabled: bool = False
    out_dir: Path | None = None
    rank: int = 0
    pool: Literal["a", "b"] = "a"
    skip_first: int = 20
    active: int = 3
    _prof: torch.profiler.profile | None = None

    def __enter__(self) -> "PhaseProfiler":
        if not self.enabled:
            return self
        assert self.out_dir is not None, "PhaseProfiler enabled but no out_dir given"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.out_dir / f"trace_pool{self.pool}_rank{self.rank}.json"

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
        """Annotate a logical step phase. No-op when disabled."""
        if not self.enabled:
            yield
            return
        with torch.profiler.record_function(name):
            yield

    def step(self) -> None:
        """Advance the profiler schedule (call once per training iteration)."""
        if self._prof is not None:
            self._prof.step()
