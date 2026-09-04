"""Regenerate `placement_goldens.json` — only when placement semantics deliberately
change. The grid and serialization live in `test_placement_goldens`."""

import json

from param_decomp.tests.core.test_placement_goldens import GOLDENS_PATH, GRID_KEYS, build_cell


def main() -> None:
    goldens = {key: build_cell(key) for key in GRID_KEYS}
    GOLDENS_PATH.write_text(json.dumps(goldens, indent=1, sort_keys=True) + "\n")
    print(f"wrote {len(goldens)} cells to {GOLDENS_PATH}")


if __name__ == "__main__":
    main()
