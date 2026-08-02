# Task 811 initial complexity-scale robustness

Endpoint-width robustness extension to the random-overcomplete fcov controller grid:
C={100,800}, initial complexity scale={0.25,4}, three seeds, with the main grid scale=1
arms as the center. Every other setting is identical.

This checks whether the controller actually removes coefficient initialization as a
hyperparameter or merely transfers C-insensitivity at one lucky initial force. Acceptance:
all three initial scales converge to the same feasible identity/PGD/selected-scale
operating point at both endpoints, without per-C tuning.
