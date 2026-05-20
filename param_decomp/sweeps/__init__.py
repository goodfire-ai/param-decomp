"""Sweep specification and generator framework.

A single PD launch is just a ``RunConfig``. A sweep is a ``SweepSpec``
containing many ``RunConfig``\\ s that share a driver and runtime substrate.

A sweep generator is any zero-arg callable returning a ``SweepSpec``. The
runner imports a user-specified ``.py`` file, calls the named function,
validates every generated config against the driver, snapshots the
materialized spec to disk, and submits a SLURM array (one task per run).
"""

import importlib.util
from pathlib import Path
from typing import cast

from param_decomp.sweeps.spec import SweepGenerator, SweepSpec

__all__ = [
    "SweepGenerator",
    "SweepSpec",
    "load_sweep_generator",
]


def load_sweep_generator(spec: str) -> SweepGenerator:
    """Load a sweep generator from ``"/abs/path/to/file.py:func_name"``.

    The path must be absolute and end in ``.py``; the function must exist in
    the module and be callable. Static conformance to the ``SweepGenerator``
    protocol (zero-arg, returns ``SweepSpec``) is enforced by type checking;
    the runner additionally validates the return value at runtime.
    """
    path_str, sep, func_name = spec.rpartition(":")
    assert sep and path_str and func_name, (
        f"--sweep_generator_path {spec!r}: expected '/abs/path/file.py:func_name'"
    )
    path = Path(path_str)
    assert path.is_absolute(), f"--sweep_generator_path must be absolute, got {path_str!r}"
    assert path.suffix == ".py", f"--sweep_generator_path must end in .py, got {path_str!r}"
    assert path.is_file(), f"--sweep_generator_path file not found: {path}"

    module_spec = importlib.util.spec_from_file_location("_pd_user_sweep", path)
    assert module_spec is not None and module_spec.loader is not None, (
        f"failed to load sweep module from {path}"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    assert hasattr(module, func_name), (
        f"{path}: no function named {func_name!r} (have: {sorted(n for n in dir(module) if not n.startswith('_'))})"
    )
    func = getattr(module, func_name)
    assert callable(func), f"{path}:{func_name} is not callable"
    return cast(SweepGenerator, func)
