"""Sweep specification and generator framework.

A sweep generator is any callable that maps a base experiment config to a list of
concrete runs (a ``SweepSpec``). The runner takes a generator, calls it, validates
every generated config against the driver's pydantic type, snapshots the
materialized spec to disk, and submits a SLURM array (one task per run).

Built-in generators live in this package and are auto-discovered by ``name``.
Custom generators can be referenced by ``module:Class`` import path.
"""

from param_decomp.sweeps.cartesian import CartesianGridSweep
from param_decomp.sweeps.discovery import discover_sweeps, resolve_sweep
from param_decomp.sweeps.spec import SweepGenerator, SweepRun, SweepSpec

__all__ = [
    "CartesianGridSweep",
    "SweepGenerator",
    "SweepRun",
    "SweepSpec",
    "discover_sweeps",
    "resolve_sweep",
]
