"""The layering rule, as a test: subpackages of `param_decomp` import only DOWNWARD.

`param_decomp` is JUST a library — mostly pure functions, mostly logic; everything
infra-ish (schedulers, submission, cluster paths, code-shipping) lives in the private
wrapper (`param_decomp_goodfire`), which imports the library and never vice versa
(pinned by the head check below: `param_decomp_goodfire` is a forbidden import root
everywhere in the library). The full principle is codified in the root CLAUDE.md,
"The library rule". Within it, the library is enumerated layers. Each subpackage declares the
`param_decomp.*` prefixes it may import (`_LAYER_ALLOWED`); anything outside that set —
including `torch`, banned everywhere (the runtime is JAX; the torch oracle lives at git
tag `torch-oracle`) — fails this test. A subpackage that is not enumerated at all fails
collection: a new layer is added here deliberately, never absorbed silently.

The load-bearing directions:
  * `core` (the engine) sees a target only through the `DecomposedModel` protocol and
    the `ArchFamily` grammar contract — it must never import `targets`, nor anything
    composition-shaped built on top.
  * `targets` implements the engine's protocol per architecture — engine + vendored
    numerics only.
  * `vendored_jax` is a leaf (verbatim numeric mirrors).

The runtime is every `.py` that ships in a wheel — i.e. not the per-layer `tests/` and
`tools/` dirs. Test suites are exempt on purpose: engine tests may use a concrete target
as a fixture.
"""

import ast
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
_NON_RUNTIME_DIRS = {"tests", "tools"}

_ANY = ("param_decomp",)
"""The composition layers (the merged lab): free to import anything in the library.
Tightening these to real per-layer sets is deliberate follow-up work."""

_LAYER_ALLOWED: dict[str, tuple[str, ...]] = {
    "vendored_jax": ("param_decomp.vendored_jax",),
    "core": ("param_decomp.core", "param_decomp.vendored_jax"),
    "pretrain": ("param_decomp.core", "param_decomp.pretrain", "param_decomp.vendored_jax"),
    "targets": ("param_decomp.core", "param_decomp.targets", "param_decomp.vendored_jax"),
    "adapters": _ANY,
    "autointerp": _ANY,
    "clustering": _ANY,
    "experiments": _ANY,
    "harvest": _ANY,
    "infra": _ANY,
    "migrations": _ANY,
    "topology": _ANY,
}


def _subpackages() -> list[str]:
    subs = sorted(
        p.name
        for p in _PACKAGE_ROOT.iterdir()
        if p.is_dir() and (p / "__init__.py").exists() and p.name not in _NON_RUNTIME_DIRS
    )
    unlisted = [s for s in subs if s not in _LAYER_ALLOWED]
    assert not unlisted, (
        f"subpackages {unlisted} are not enumerated in _LAYER_ALLOWED — declare each new "
        "layer's allowed imports deliberately"
    )
    return subs


def _bad_imports(path: Path, allowed: tuple[str, ...]) -> list[str]:
    def is_bad(module: str) -> bool:
        head = module.split(".", 1)[0]
        if head in ("torch", "param_decomp_goodfire"):
            return True
        if head != "param_decomp":
            return False
        return not any(module == p or module.startswith(p + ".") for p in allowed)

    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                found += [alias.name for alias in names if is_bad(alias.name)]
            case ast.ImportFrom(module=module) if module is not None:
                if is_bad(module):
                    found.append(module)
            case _:
                pass
    return found


def _runtime_python_files() -> list[tuple[Path, tuple[str, ...]]]:
    cases: list[tuple[Path, tuple[str, ...]]] = []
    for sub in _subpackages():
        for path in sorted((_PACKAGE_ROOT / sub).rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT / sub)
            if rel.parts[0] in _NON_RUNTIME_DIRS:
                continue
            cases.append((path, _LAYER_ALLOWED[sub]))
    return cases


@pytest.mark.parametrize(
    ("path", "allowed"),
    _runtime_python_files(),
    ids=lambda v: str(v.relative_to(_PACKAGE_ROOT)) if isinstance(v, Path) else None,
)
def test_runtime_imports_only_downward(path: Path, allowed: tuple[str, ...]):
    bad = _bad_imports(path, allowed)
    assert not bad, (
        f"{path.relative_to(_PACKAGE_ROOT)} imports {bad}, outside its layer's allowed set "
        f"{allowed} — runtime imports only point downward"
    )
