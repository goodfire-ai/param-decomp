"""The core engine reads NO ambient process environment (the library rule): every
behavior — profiling included — arrives as typed data threaded from a composition root.
Source-level and fail-closed over ALL of core (tools and tests included), so a new
`os.environ` read anywhere in core dies here. The only exemption is this scanner, whose
needle strings would match themselves."""

from pathlib import Path

import param_decomp.core

CORE_DIR = Path(param_decomp.core.__file__).parent
ENV_READS = ("os.environ", "os.getenv", "getenv(")


def test_core_reads_no_process_environment():
    offenders = sorted(
        f"{module.relative_to(CORE_DIR)}: {line.strip()}"
        for module in CORE_DIR.rglob("*.py")
        if module != Path(__file__) and "__pycache__" not in module.parts
        for line in module.read_text().splitlines()
        if any(read in line for read in ENV_READS)
    )
    assert not offenders, offenders
