# ResidMLP1 overcomplete-C interaction grid (#811)

A fixed-recipe, converged `resid_mlp_1l` grid with every capacity safely overcomplete:
`C={200,400,800}` (2–8x the 100 input mechanisms; 4–16x the 50 hidden mechanisms),
combined importance+frequency force scale `{0.25,0.5,1,2,4}`, and three decomposition seeds.
The frozen target/data/config are otherwise identical to the canonical 50k-step ResidMLP1
recipe. Scaling both complexity terms preserves their marginal ratio; the grid tests both the
fixed-config C effect and whether the sweep-optimal force moves with C.

Acceptance/diagnostics: settled-window `identity_ci_error` and `dense_ci_error`, fresh PGD
reconstruction, faithfulness, alive/active count, convergence, and C x force interactions.
Ground-truth recovery outranks reconstruction whenever they disagree.
