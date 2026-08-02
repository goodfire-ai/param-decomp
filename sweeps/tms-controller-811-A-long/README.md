# Task 811 controller case-A convergence extension

The 40k acceptance arms ended after 8–9 accepted births, still above the authored
fresh-PGD MSE budget, and inside the next birth transaction. These three 80k arms test
whether the serial feasibility-birth lifecycle reaches the budget and a terminal
operating point or merely shifts the incomplete endpoint.

The persistent-adversary LR reaches its maximum after the same 1,000 physical training
steps as the 40k arms (`at: 0.0125` at 80k rather than normalized-progress `0.025`). All
scientific settings remain fixed: tau=5e-4, gamma=.1, primal/CI LR=1e-3, Cmax=20 per site,
logical prefix=5 per site. The trainer rolls back any live transaction before final eval
and checkpoint; `controller/terminal_rollback` records that cleanup.
