"""Resolve ``"module:attr"`` driver paths to driver objects.

Lives outside ``experiments/`` so both ``param_decomp.run`` (for ``Run`` subclass
dispatch) and ``experiments.driver`` (for the public re-export) can import it
without forming an import cycle through the ``ExperimentDriver`` Protocol.
"""

from importlib import import_module
from typing import Any


def load_driver(driver_path: str) -> Any:
    """Load a driver object or no-arg driver class from a ``module:attr`` import path."""
    module_path, sep, attr = driver_path.partition(":")
    if sep == "":
        raise ValueError(f"Driver path must be of the form 'module:attr', got {driver_path!r}")
    driver = getattr(import_module(module_path), attr)
    if isinstance(driver, type):
        driver = driver()
    return driver
