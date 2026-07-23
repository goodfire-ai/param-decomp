"""The layering rule, as a test: `lab → targets → engine`, downward imports only.

The ENGINE (`param_decomp/`) is generic: it sees targets only through the
`DecomposedModel` protocol and the `ArchFamily` grammar contract, so it must never
import the targets distribution (`param_decomp_targets`) — nor the lab
(`param_decomp_lab`), which owns every composition root (YAML→ExperimentConfig, the
`main()` entrypoints, run-loading) — nor `torch` (the engine is the GPU runtime; the
JAX stack + its own siblings `pretrain`/`vendored_jax` only).

The TARGETS layer (`param_decomp_targets/`) implements the engine's protocol per
architecture; it depends downward on the engine (and `vendored_jax`) but never on the
lab, and never on `torch`.

The runtime is every `.py` that ships in a wheel — i.e. not the test suites and not
`tools/` (torch-env scripts that run in the torch venv by design). Test suites are
exempt on purpose: engine tests may use a concrete target as a fixture. This static AST
scan is the CI form of the `grep -rniE` acceptance check, scoped to those runtime files.
"""

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_NON_RUNTIME_DIRS = {"tests", "tools"}
_LAYERS = {
    "param_decomp": ("param_decomp_targets", "param_decomp_lab", "torch"),
    "param_decomp_targets": ("param_decomp_lab", "torch"),
}


def _forbidden_imports(path: Path, forbidden_roots: tuple[str, ...]) -> list[str]:
    is_forbidden = lambda module: module.split(".", 1)[0] in forbidden_roots  # noqa: E731
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                found += [alias.name for alias in names if is_forbidden(alias.name)]
            case ast.ImportFrom(module=module) if module is not None:
                if is_forbidden(module):
                    found.append(module)
            case _:
                pass
    return found


def _runtime_python_files() -> list[tuple[Path, tuple[str, ...]]]:
    cases: list[tuple[Path, tuple[str, ...]]] = []
    for layer, forbidden in _LAYERS.items():
        layer_root = _REPO_ROOT / layer
        assert layer_root.is_dir(), layer_root
        for path in sorted(layer_root.rglob("*.py")):
            rel = path.relative_to(layer_root)
            if rel.parts[0] in _NON_RUNTIME_DIRS:
                continue
            cases.append((path, forbidden))
    return cases


@pytest.mark.parametrize(
    ("path", "forbidden_roots"),
    _runtime_python_files(),
    ids=lambda v: str(v.relative_to(_REPO_ROOT)) if isinstance(v, Path) else None,
)
def test_runtime_imports_only_downward(path: Path, forbidden_roots: tuple[str, ...]):
    forbidden = _forbidden_imports(path, forbidden_roots)
    assert not forbidden, (
        f"{path.relative_to(_REPO_ROOT)} imports {forbidden}; the layering is "
        "`lab -> targets -> engine` and runtime imports only point downward"
    )
