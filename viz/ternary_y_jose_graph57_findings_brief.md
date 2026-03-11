# Graph 57: `z = y ? x :` → ` y`

Optimized graph (L0=210, 3057 edges). The model predicts ` y` at position 5.

## Mechanism

**Layer 0 — Token identity.** `0.mlp.up:2441`, a y-letter detector, fires only at position 2 (` y`) with no corresponding x-detector at position 4. This is the earliest source of asymmetry between the two variable positions.

**Layer 2 — Induction-like copying.** The dominant head `2.attn.o:5:210` (~5.3 total effect on output) attends roughly equally to both variable positions via K. The y-vs-x preference comes from the V-side: V components read different content from pos 2 and pos 4. The biggest differentiator is `V:627` ("completions for words starting with j or y"), whose activation flips sign between positions — negative at ` y`, positive at ` x` — creating opposite-sign contributions. Two additional V components (`V:406`, `V:363`) fire only at pos 2. The total V-side asymmetry is +0.98 favoring ` y`. Some secondary O heads also show K-routing preference toward pos 2.

**Layer 3 — Amplification.** 33 `3.mlp.down` components amplify the layer 2 signal into +8.60 net attribution to the output (the largest single-layer contribution). `3.attn.o:806` provides a suppressive -1.38 based on structural syntax tokens (` =`, ` :`), responding to the ternary pattern itself rather than the variable values.

## Key chain

`0.mlp.up:2441` (y-detector, pos2 only) → `0.mlp.down:3082` → `V:627` (sign flip) → `2.attn.o:210` → `3.mlp.down` → output ` y`.
