"""CLI entry point: ``python -m param_decomp.scripts.two_pool_benchmark.submit_sweep``.

Reads a sweep YAML, expands the cartesian grid into :class:`SweepPoint`s, and
submits each via ``submit_point``.

Usage::

    python -m param_decomp.scripts.two_pool_benchmark.submit_sweep \\
        --config path/to/sweep.yaml [--dry-run]
"""

import argparse
from pathlib import Path

import yaml

from param_decomp.scripts.two_pool_benchmark.submit_sweep.schema import SweepConfig
from param_decomp.scripts.two_pool_benchmark.submit_sweep.submit import submit_point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to sweep YAML.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved sweep points without writing files or submitting.",
    )
    parser.add_argument(
        "--master-port-base",
        type=int,
        default=30100,
        help="First MASTER_PORT; incremented per submitted point.",
    )
    args = parser.parse_args()

    cfg = SweepConfig.model_validate(yaml.safe_load(args.config.read_text()))
    points = cfg.expand()
    print(f"[sweep] {len(points)} points from {args.config}")
    for i, point in enumerate(points):
        submit_point(
            point,
            cfg,
            master_port=args.master_port_base + i,
            do_submit=not args.dry_run,
        )


if __name__ == "__main__":
    main()
