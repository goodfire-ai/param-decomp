from pathlib import Path

from param_decomp_goodfire.env import GoodfireEnvironment


def test_goodfire_env_resolves_cluster_defaults(tmp_path: Path):
    mount = tmp_path / "mnt"
    mount.mkdir()
    env = GoodfireEnvironment.from_env({"DATA_MOUNT": str(mount)})
    assert env.data_mount == mount
    assert env.output_root == mount / "artifacts/mechanisms/param-decomp"

    dead = GoodfireEnvironment.from_env({"DATA_MOUNT": str(tmp_path / "missing")})
    assert dead.data_mount is None
    assert dead.output_root == Path("out")

    pinned = GoodfireEnvironment.from_env(
        {"PARAM_DECOMP_OUT_DIR": "/x/y", "PARTITION_RESERVED": "h100"}
    )
    assert pinned.output_root == Path("/x/y")
    assert pinned.default_partition == "h100"
