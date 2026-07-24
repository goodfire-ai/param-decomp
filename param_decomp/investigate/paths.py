from pathlib import Path

from param_decomp.infra.settings import ENV


def investigation_output_dir(inv_id: str) -> Path:
    return ENV.output_root / "investigations" / inv_id
