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
            "import param_decomp_lab.infra.settings",
        ],
        env=env,
        check=True,
    )

    assert not (data_mount / "artifacts").exists()
