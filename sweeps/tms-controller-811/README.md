# Task 811 controller acceptance

Pinned TMS 40→10 acceptance suite for the reconstruction-budget controller + logical capacity lifecycle.

- **A:** physical Cmax=20, logical prefix 5/site. Complexity-OFF must trigger protected GradMax feasibility births and grow enough capacity to meet the fixed raw-MSE fresh-PGD budget `tau=5e-4`.
- **B:** Cmax=40, logical rank-10 SVD prefix. OFF is feasible but the basis is the known identity-error ≈117 plateau; protected column probes must improve declared complexity at matched recon or roll back transactionally.
- **C:** repeat the combined recipe at physical Cmax ∈ {40,100,200,400}, fixed logical prefix 10 and zero per-C tuning. Read settled recon/complexity, identity-CI error, active count, birth/probe events, and trial verdicts.

All arms hold SmoothL0 gamma at 0.1 and primal/CI LR constant; the controller is the only slow outer loop. Three seeds each, 40k accepted training steps, 20-step fresh-PGD referee (two eval batches) every 500 steps.
