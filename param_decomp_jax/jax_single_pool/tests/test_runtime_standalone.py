"""The GPU runtime is self-contained: it imports the JAX stack + `param_decomp_config`
ONLY — never the adjacent torch-stack distributions (`param_decomp`, `param_decomp_lab`)
and never `torch`. This pins the dependency boundary the standalone `pyproject` declares
(issue #809 / TRANSITION.md §5.4): `lab → param_decomp_jax` is allowed; the reverse edge
`param_decomp_jax → adjacent` is forbidden for the runtime.

The runtime is every `.py` under `jax_single_pool/` that ships in the wheel — i.e. not
the test suite, not the torch-env `tools/` scripts (export verifier / checkpoint
converters that run in the torch venv by design), and not the torch-side config YAMLs.
This static AST scan is the CI form of the `grep -rniE "param_decomp\\.|param_decomp_lab\\.|import torch"`
acceptance check, scoped to those runtime files.
"""

import ast
from pathlib import Path

import pytest

_RUNTIME_ROOT = Path(__file__).resolve().parent.parent
_NON_RUNTIME_DIRS = {"tests", "tools"}
_FORBIDDEN_ROOTS = ("param_decomp", "param_decomp_lab", "torch")


def _is_forbidden(module: str) -> bool:
    head = module.split(".", 1)[0]
    if head == "param_decomp_config":
        return False
    return head in _FORBIDDEN_ROOTS


def _runtime_python_files() -> list[Path]:
    files: list[Path] = []
    for path in _RUNTIME_ROOT.rglob("*.py"):
        rel = path.relative_to(_RUNTIME_ROOT)
        if rel.parts[0] in _NON_RUNTIME_DIRS:
            continue
        files.append(path)
    return files


def _forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                found += [alias.name for alias in names if _is_forbidden(alias.name)]
            case ast.ImportFrom(module=module) if module is not None:
                if _is_forbidden(module):
                    found.append(module)
    return found


@pytest.mark.parametrize(
    "path", _runtime_python_files(), ids=lambda p: str(p.relative_to(_RUNTIME_ROOT))
)
def test_runtime_imports_nothing_adjacent(path: Path):
    forbidden = _forbidden_imports(path)
    assert not forbidden, (
        f"{path.relative_to(_RUNTIME_ROOT)} imports adjacent/torch modules {forbidden}; "
        "the GPU runtime must depend on the JAX stack + param_decomp_config only"
    )
