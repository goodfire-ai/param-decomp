# Lore-referenced artifacts

Frozen repro / probe scripts cited by lore docs and CLAUDE.md, kept off the main
branch so they don't clutter the codebase while still resolving to a GitHub permalink.

- `nccl_*_repro.py` — standalone NCCL repros from the 3-pool asym-seq2048 cross-pool
  deadlock investigation (lore: three-pool-asym-deadlock-*).
- `ppgd_*_probe.py` — 1-GPU correctness/perf probes for torch.compile + autograd.grad
  on the PPGD pool (param_decomp_lab/three_pool/CLAUDE.md "PPGD torch.compile").

These are point-in-time snapshots; they are not maintained against the evolving codebase.
