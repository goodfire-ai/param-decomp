# Performance notes

Performance claims are cell-indexed: compare only runs with the same hardware, mesh,
batch, sequence length, objective, resources, and measurement window, with code as the
sole delta. Code comments and docstrings should explain mechanisms rather than carrying
figures that cannot be reproduced from the repository.

The current implementation's main performance mechanisms are:

- chained-reduced component residents, which move cross-replica communication to the
  boundaries of the target scan;
- ordinary autodiff over declared forward placements rather than a hand-written
  transpose;
- per-kind Newton–Schulz staging at the stack-sharding waypoint;
- the `tuned-v1` compiler preset, with `bare` as the true XLA-default baseline; and
- configurable reconstruction rematerialization, trading compute time for peak memory.

Treat these as hypotheses to measure in a complete cell, not as universal rankings.
Record the exact commit, authored configuration, mesh, accelerator, software versions,
warmup, measurement window, throughput, and peak memory for every comparison.
