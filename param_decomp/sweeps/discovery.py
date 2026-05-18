"""Auto-discovery + CLI-string resolution for sweep generators.

Discovery scans ``param_decomp/sweeps/*.py`` for ``SweepGenerator`` subclasses
with a ``name`` class var. Resolution turns a single ``--sweep <spec>`` string
into a constructed generator instance:

    - ``<name>:<arg>``    → look up ``<name>`` in the registry, pass ``<arg>``
    - ``<name>``          → look up ``<name>``, no arg
    - ``<module>:<Class>`` → import; only used if ``<module>`` isn't a registered name
    - ``<path>.yaml``     → shorthand for ``cartesian:<path>``

Tiebreaker for ``name:arg`` vs ``module:Class``: if the part before ``:`` is a
registered short name, use the registry. Registered names are simple identifiers
(no dots), so ``my_pkg.sweeps:MyClass`` is unambiguous.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import cast

from param_decomp.sweeps.spec import SweepGenerator

_PKG = "param_decomp.sweeps"
_PKG_PATH = Path(__file__).parent


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
    arg = tail if sep else None
    registry = discover_sweeps()

    # `<name>` or `<name>:<arg>` where name is a registered short identifier.
    if head in registry:
        return _instantiate_class(registry[head], arg)

    # No colon and looks like a yaml file → shorthand for `cartesian:<spec>`.
    if not sep and (spec.endswith(".yaml") or spec.endswith(".yml")):
        return _instantiate("cartesian", spec)

    # `<module.path>:<Class>` import (head must contain a dot to disambiguate).
    assert sep and "." in head, (
        f"--sweep {spec!r}: not a known sweep name (have {sorted(registry)}), "
        f"not a yaml path, and not a 'module.path:Class' import."
    )
    module_path, _, class_name = head.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return _instantiate_class(cls, arg)


def _instantiate(name: str, arg: str | None) -> SweepGenerator:
    registry = discover_sweeps()
    assert name in registry, f"unknown sweep generator: {name!r} (have {sorted(registry)})"
    return _instantiate_class(registry[name], arg)


def _instantiate_class(cls: type, arg: str | None) -> SweepGenerator:
    instance = cls(arg) if arg is not None else cls()
    assert callable(instance), f"{cls!r} is not callable; SweepGenerator must define __call__"
    return cast(SweepGenerator, instance)
