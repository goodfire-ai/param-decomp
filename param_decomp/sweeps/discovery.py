"""Auto-discovery + CLI-string resolution for sweep generators.

Discovery scans ``param_decomp/sweeps/*.py`` for ``SweepGenerator`` subclasses
with a ``name`` class var. Resolution turns a single ``--sweep <spec>`` string
into a constructed generator instance:

    - ``<name>:<arg>``     → look up ``<name>`` in the registry, pass ``<arg>``
    - ``<name>``           → look up ``<name>``, no arg
    - ``<module>:<Class>`` → import; only used if ``<module>`` contains a dot
    - ``<path>.yaml``      → shorthand for ``cartesian:<path>``

Tiebreaker for ``name:arg`` vs ``module:Class``: if the part before ``:`` is a
registered short name, use the registry. Registered names are simple identifiers
(no dots), so ``my_pkg.sweeps:MyClass`` is unambiguous.
"""

import importlib
import inspect
import pkgutil
from functools import cache
from pathlib import Path

from param_decomp.sweeps.spec import SweepGenerator

_PKG = "param_decomp.sweeps"
_PKG_PATH = Path(__file__).parent


@cache
def discover_sweeps() -> dict[str, type[SweepGenerator]]:
    """Find all SweepGenerator subclasses in param_decomp.sweeps.* and register by name."""
    registry: dict[str, type[SweepGenerator]] = {}
    for module_info in pkgutil.iter_modules([str(_PKG_PATH)]):
        if module_info.name.startswith("_") or module_info.name in ("discovery", "spec"):
            continue
        module = importlib.import_module(f"{_PKG}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, SweepGenerator)
                and obj is not SweepGenerator
                and "name" in obj.__dict__
            ):
                if obj.name in registry:
                    assert registry[obj.name] is obj, (
                        f"Duplicate sweep generator name {obj.name!r}: "
                        f"{registry[obj.name]} vs {obj}"
                    )
                registry[obj.name] = obj
    return registry


def resolve_sweep(spec: str) -> SweepGenerator:
    """Parse ``--sweep <spec>`` and return a constructed generator."""
    head, sep, tail = spec.partition(":")
    # Treat ``name:`` (no arg after the colon) as a no-arg invocation, not as
    # ``arg=""`` — empty strings reach Path()/yaml.safe_load() with confusing errors.
    arg: str | None = tail if sep and tail else None
    registry = discover_sweeps()

    if head in registry:
        return _construct(registry[head], arg)

    if not sep and (spec.endswith(".yaml") or spec.endswith(".yml")):
        return _construct(registry["cartesian"], spec)

    assert sep and "." in head, (
        f"--sweep {spec!r}: not a known sweep name (have {sorted(registry)}), "
        f"not a yaml path, and not a 'module.path:Class' import."
    )
    module_path, _, class_name = head.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    assert issubclass(cls, SweepGenerator), f"{cls!r} is not a SweepGenerator subclass"
    return _construct(cls, arg)


def _construct(cls: type[SweepGenerator], arg: str | None) -> SweepGenerator:
    # The base class declares no constructor signature, but concrete subclasses may
    # accept an arg (e.g. CartesianGridSweep's yaml path). Trust the discovery contract.
    if arg is None:
        return cls()
    return cls(arg)  # pyright: ignore[reportCallIssue]
