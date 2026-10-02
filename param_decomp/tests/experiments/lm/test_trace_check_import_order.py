"""The trace root loads tokamax before Arrow-backed data can register protobuf.

The probe starts a fresh interpreter because this test process already imported both
libraries. Data loading is a separate import, independent of trace_check's dependencies.
"""

import os
import subprocess
import sys

_PROBE = """
import sys

import param_decomp.experiments.lm.trace_check

assert "tokamax" in sys.modules, "the trace gate must import tokamax"

import param_decomp.lm.batch_data

loaded = list(sys.modules)
assert "pyarrow" in loaded, "the batch loader must import pyarrow"
assert loaded.index("tokamax") < loaded.index("pyarrow"), (
    loaded.index("tokamax"),
    loaded.index("pyarrow"),
)
"""


def test_trace_check_imports_tokamax_before_pyarrow() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=os.environ | {"JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-3000:]
