# param_decomp.routed — architecture notes

JAX expert-parallel routed compute for MoE layers: static jobs schedules (`RoutedJobs`
for a single shard, `ExpertShardedJobs` for the explicit `(data, tp)` mesh), the
sort/gather/unsort primitives with custom VJPs (transposes stay gathers, never XLA
scatter-adds — except the honestly-partial block gather), and the grouped-matmul backend
arms (`GroupedMatmulBackend`: `ragged_dot` | `tokamax` | `tokamax_split_vjp`). One
routed kernel module, `experts.py`; its docstring describes the job layout.
`dense.py` provides the dense mathematical reference on token/expert axes.
`dense_select_experts` reads expert-sharded dense values into token-ordered,
TP-replicated selected values.
`ExpertImplementation` chooses dense masked execution or a grouped-matmul backend;
this choice does not determine the CI emission interface.

`dense_project_and_combine_experts` checkpoints 32-position sequence chunks, so
expert down-projection outputs stay inside the chunk. It preserves each expert's
compute-dtype output rounding before the fp32 routing reduction; the projection
gradient accumulates over chunks in the compute dtype.

## Public surface

The names consumers import today:

- Schedules: `RoutedJobs` / `routed_jobs`, `ExpertShardedJobs` / `expert_sharded_jobs`.
- Config vocabulary: `GroupedMatmulBackend`.
- Local primitives: `gather_tokens`, `sort_jobs`, `unsort_jobs`, `scatter_jobs`,
  `gather_job_blocks`, `sort_job_values`, `sum_jobs`, `combine_jobs`, `grouped_matmul`,
  `transposed_grouped_matmul`.
- Expert-parallel siblings: `ep_gather_tokens`, `ep_sort_jobs`, `ep_unsort_jobs`,
  `ep_gather_job_blocks`, `ep_sort_job_values`, `ep_sum_jobs`, `ep_combine_jobs`,
  `ep_grouped_matmul`.

`ep_gather_job_blocks` reads full-token or broadcast blocked sources directly into
expert-owned jobs. Each source leading extent is either one or its token extent;
position broadcasts stay in the indices, and dead jobs contribute no source gradient.
Source storage ownership is independent: the read explicitly redistributes to the
consumer's expert layout, and autodiff returns cotangents to the storage layout.
Per-batch minipool sampling preserves its input component layout before this boundary.

## Layering

A leaf of the `param_decomp` layer graph, pinned by
`tests/core/test_runtime_standalone.py`: it imports jax (pallas included), jaxtyping,
and tokamax only — no `param_decomp` modules. `_`-prefixed names are module-private;
consumers import only the public surface above. Its tests live at `tests/routed/`.
