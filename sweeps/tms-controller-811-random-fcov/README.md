# Task 811 random-overcomplete controller preservation

Decisive C-upper-bound test: random fully-active dictionaries at C=100,200,400,800,
function-covariant balanced Adam, and one fixed reconstruction-budget controller. The
first 20k steps reproduce the pinned fcov schedule shape (gamma 1→.01 and cosine LR
decay); the second 20k holds those fast schedules fixed so the settled outer controller
can finish. All controller settings and τ=5e-4 are identical across C; three seeds.

Acceptance: transaction-safe endpoints; identity recovery and fresh-PGD constraint flat
in C; controller-selected complexity scale/complexity do not drift systematically with C.
This tests whether coefficient control preserves the banked random-start C=100–800
flatness rather than reintroducing a width-dependent trajectory.
