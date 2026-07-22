"""The GPU runtime core (`param_decomp/`) is self-contained: it imports the JAX stack
(plus the torch-free pydantic config schema it now carries, `param_decomp.configs` /
`base_config` / `schedule`) + its own siblings (`pretrain`, `vendored_jax`) ONLY — never
the lab distribution (`param_decomp_lab`) and never `torch`. This pins the dependency
boundary: `lab → param_decomp` is allowed; the reverse edge `param_decomp → lab` is
forbidden for the runtime. Notably the composition root (the YAML→ExperimentConfig
conversion, the `main()` entrypoints, run-loading) lives lab-side, so this scan also
guards that none of it leaked back into core.

The runtime is every `.py` under `param_decomp/` that ships in the wheel — i.e. not
the test suite, not the torch-env `tools/` scripts (export verifier / checkpoint
converters that run in the torch venv by design), and not the torch-side config YAMLs.
This static AST scan is the CI form of the `grep -rniE "param_decomp_lab\\.|import torch"`
acceptance check, scoped to those runtime files.
"""

import ast
from pathlib import Path

import pytest

_RUNTIME_ROOT = Path(__file__).resolve().parent.parent
_NON_RUNTIME_DIRS = {"tests", "tools"}
_FORBIDDEN_ROOTS = ("param_decomp_lab", "torch")


def _is_forbidden(module: str) -> bool:
    head = module.split(".", 1)[0]
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
            case _:
                pass
    return found


@pytest.mark.parametrize(
    "path", _runtime_python_files(), ids=lambda p: str(p.relative_to(_RUNTIME_ROOT))
)
def test_runtime_imports_nothing_adjacent(path: Path):
    forbidden = _forbidden_imports(path)
    assert not forbidden, (
        f"{path.relative_to(_RUNTIME_ROOT)} imports adjacent/torch modules {forbidden}; "
        "the GPU runtime must not depend on the lab distribution or torch"
    )
