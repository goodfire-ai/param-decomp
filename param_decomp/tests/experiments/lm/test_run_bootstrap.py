import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from param_decomp.experiments.lm import run


def test_bootstrap_applies_launch_env_before_loading_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "mesh": {"replicate": 1, "fsdp": 1, "tp": 1},
                    "sharding": "ddp",
                    "compilation_cache_dir": "~/.cache/param-decomp/xla",
                    "compiler_options": "tuned-v2",
                    "launch_env": {
                        "xla_python_client_mem_fraction": 0.5,
                        "env": {"PD_BOOTSTRAP_SENTINEL": "present"},
                    },
                }
            }
        )
    )
    # a wrapper's export must survive the bootstrap, composed additively with the
    # config's flags — the harness spelling that realizes simulated devices.
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    observed: list[tuple[str, str, str]] = []
    training = ModuleType("param_decomp.experiments.lm.training")

    def train_main(
        config: object,
        data_root: object,
        local_device_count: int,
        run_id: object = None,
    ) -> None:
        del config, data_root, local_device_count, run_id
        observed.append(
            (
                os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"],
                os.environ["PD_BOOTSTRAP_SENTINEL"],
                os.environ["XLA_FLAGS"],
            )
        )

    training.__dict__["main"] = train_main
    monkeypatch.setitem(sys.modules, training.__name__, training)

    run.main(config, Path("/tmp/unused-data-root"), 1, "p-00000000")

    assert observed == [
        (
            "0.5",
            "present",
            "--xla_gpu_nccl_termination_timeout_seconds=600"
            " --xla_force_host_platform_device_count=8",
        )
    ]


def test_importing_the_bootstrap_does_not_import_jax() -> None:
    """The knobs `launch_env` carries are read at backend init, so exporting them after JAX
    is imported is a silent no-op. That makes the bootstrap's import closure load-bearing:
    it may reach the `runtime:` schema (`lm/runtime.py`) but nothing that pulls JAX in — the
    reason that schema is its own module rather than part of `lm/config.py`. A subprocess
    because the suite has JAX resident by the time this runs."""
    probe = "import sys, param_decomp.experiments.lm.run as _; print('jax' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout


_REAL_DISTRIBUTED_PROBE = r"""
import socket
import sys
from pathlib import Path

import jax
import yaml

from param_decomp.experiments.lm import training, training_targeted
from param_decomp.tests.experiments.lm.test_lm_targeted import _targeted_raw


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TopologyInitialized(Exception):
    pass


def single_process_topology(*_: object) -> None:
    # The production MultiNode arm, with this process as the whole world: JAX refuses
    # this call once any backend is up, which is exactly the regression under test.
    # Everything after topology bring-up needs a GPU backend, so the probe ends here.
    jax.distributed.initialize(
        coordinator_address=f"127.0.0.1:{free_port()}", num_processes=1, process_id=0
    )
    assert jax.process_count() == 1
    raise TopologyInitialized


entry, data_root = sys.argv[1], Path(sys.argv[2])
match entry:
    case "plain":
        module = training
        config = Path("param_decomp/experiments/lm/configs/qwen3_0_6b.yaml")
    case "targeted":
        module = training_targeted
        config = data_root / "targeted.yaml"
        config.write_text(yaml.safe_dump(_targeted_raw()))
    case other:
        raise AssertionError(other)
module.initialize_topology = single_process_topology
try:
    module.main(config, data_root, 8, "p-00000000")
except TopologyInitialized:
    print(f"{entry}: distributed init accepted after config resolution")
else:
    raise AssertionError("startup did not reach topology bring-up")
"""


@pytest.mark.parametrize("entry", ["plain", "targeted"])
def test_lm_entries_resolve_configs_before_distributed_init(entry: str, tmp_path: Path) -> None:
    """Both LM entries build their objective while resolving the config, before
    `initialize_topology`; that build must not initialize a backend, or the multi-node
    arm's `jax.distributed.initialize` is refused (PR586 regression). A subprocess
    because the suite has a backend resident by the time this runs."""
    result = subprocess.run(
        [sys.executable, "-c", _REAL_DISTRIBUTED_PROBE, entry, str(tmp_path)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        },
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{entry}: distributed init accepted after config resolution" in result.stdout
