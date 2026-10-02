"""The CI shards are an exact partition of the test tree.

The workflow matrix names the shards; each `test-ci-<shard>` Makefile target names its
paths. This pins that every test file runs in exactly one xdist shard and every
multidevice-marked file runs in exactly one multidevice shard — a directory added
to the tree, or a marked file added to a subsystem, must be placed deliberately rather
than silently never run.
"""

import re
import shlex
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
TEST_ROOTS = (REPO / "param_decomp/tests",)
# The mark as code — a decorator, a `pytestmark` assignment, or a marks-list entry — not
# the phrase quoted in prose.
MULTIDEVICE_MARK = re.compile(r"(?:^|[@=\[(,\s])pytest\.mark\.multidevice\b(?![^\n]*`)", re.M)


def _shards() -> list[str]:
    workflow = yaml.safe_load((REPO / ".github/workflows/checks.yaml").read_text())
    return list(workflow["jobs"]["build"]["strategy"]["matrix"]["shard"])


def _pytest_argv(shard: str) -> list[str]:
    """The pytest command `make -n` prints for a shard, tokenised."""
    printed = subprocess.run(
        ["make", "-n", f"test-ci-{shard}"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout
    (line,) = [ln for ln in printed.splitlines() if "pytest" in ln]
    tokens = shlex.split(line)
    return tokens[tokens.index("pytest") + 1 :]


def _test_files(path: Path) -> set[Path]:
    if path.is_file():
        return {path}
    assert path.is_dir(), path
    return set(path.rglob("test_*.py"))


def _selected_files(argv: list[str]) -> set[Path]:
    """Files a pytest invocation collects: its positional paths minus its `--ignore`s."""
    included: set[Path] = set()
    ignored: set[Path] = set()
    for token in argv:
        if token.startswith("--ignore="):
            ignored.update(_test_files(REPO / token.removeprefix("--ignore=")))
        elif not token.startswith("-") and (token.endswith(".py") or token.endswith("/")):
            included.update(_test_files(REPO / token))
    return included - ignored


def _all_test_files() -> set[Path]:
    return {f for root in TEST_ROOTS for f in root.rglob("test_*.py")}


def _multidevice_files() -> set[Path]:
    return {f for f in _all_test_files() if MULTIDEVICE_MARK.search(f.read_text())}


def test_workflow_shards_are_make_targets() -> None:
    targets = set(re.findall(r"^test-ci-([a-z0-9-]+):", (REPO / "Makefile").read_text(), re.M))
    assert set(_shards()) == targets, (sorted(_shards()), sorted(targets))


def test_every_shard_runs_slow_tests() -> None:
    """CI runs the whole suite: a shard command without `--runslow` would silently drop
    every `slow`-marked test in its files (the multidevice pass once did exactly that)."""
    for shard in _shards():
        assert "--runslow" in _pytest_argv(shard), shard


def test_xdist_shards_partition_every_test_file() -> None:
    argvs = {shard: _pytest_argv(shard) for shard in _shards()}
    xdist = {shard: argv for shard, argv in argvs.items() if "multidevice" not in argv}
    owners: dict[Path, list[str]] = {f: [] for f in _all_test_files()}
    for shard, argv in xdist.items():
        for f in _selected_files(argv):
            owners[f].append(shard)
    unowned = sorted(str(f.relative_to(REPO)) for f, s in owners.items() if not s)
    shared = {str(f.relative_to(REPO)): s for f, s in owners.items() if len(s) > 1}
    assert not unowned, f"test files in no xdist shard: {unowned}"
    assert not shared, f"test files in several xdist shards: {shared}"


def test_multidevice_shards_partition_every_marked_file() -> None:
    argvs = {shard: _pytest_argv(shard) for shard in _shards()}
    multidevice = {shard: argv for shard, argv in argvs.items() if "multidevice" in argv}
    marked = _multidevice_files()
    owners: dict[Path, list[str]] = {f: [] for f in marked}
    for shard, argv in multidevice.items():
        for f in _selected_files(argv):
            assert f in marked, (
                f"{shard} lists {f.relative_to(REPO)}, which has no multidevice test"
            )
            owners[f].append(shard)
    unowned = sorted(str(f.relative_to(REPO)) for f, s in owners.items() if not s)
    shared = {str(f.relative_to(REPO)): s for f, s in owners.items() if len(s) > 1}
    assert not unowned, f"multidevice-marked files in no multidevice shard: {unowned}"
    assert not shared, f"multidevice-marked files in several shards: {shared}"
