import os
import subprocess
import sys
from pathlib import Path


def test_settings_import_does_not_create_output_dirs(tmp_path: Path):
    data_mount = tmp_path / "data"
    data_mount.mkdir()

    env = os.environ.copy()
    env["DATA_MOUNT"] = str(data_mount)
    env.pop("PARAM_DECOMP_OUT_DIR", None)

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import param_decomp.infra.settings",
        ],
        env=env,
        check=True,
    )

    assert not (data_mount / "artifacts").exists()


def test_environment_from_env_is_explicit(tmp_path: Path):
    from param_decomp.infra.settings import Environment

    mount = tmp_path / "mnt"
    mount.mkdir()
    env = Environment.from_env({"DATA_MOUNT": str(mount)})
    assert env.data_mount == mount
    assert env.output_root == mount / "artifacts/mechanisms/param-decomp"

    dead = Environment.from_env({"DATA_MOUNT": str(tmp_path / "missing")})
    assert dead.data_mount is None
    assert dead.output_root == Path("out")

    pinned = Environment.from_env({"PARAM_DECOMP_OUT_DIR": "/x/y", "PARTITION_RESERVED": "h100"})
    assert pinned.output_root == Path("/x/y")
    assert pinned.default_partition == "h100"
