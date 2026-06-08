# Track-2 ledger — experiment index

The human's whole report surface (plus [`baselines.md`](baselines.md)). One row per
experiment; the detailed card lives in `experiments/<id>.md` (from
[`TEMPLATE.md`](TEMPLATE.md)). Update the row whenever the stage changes.

**Claim type:** `speedup` (faster, same quality) or `simplification` (fewer moving parts,
similar quality). **Stage:** `proposed` / `T0` / `T1` / `merged` / `killed` / `parked`.

| id | idea | claim | stage | headline result |
|---|---|---|---|---|
| [spd-ppgd-nwarmup0](experiments/spd-ppgd-nwarmup0.md) | PPGD inner warmup steps 2 → 0 | speedup | T0 | ~33% faster (156→105 ms/step); quality TBD (run `p-bad26c65`) |
| [spd-ci-blocks2](experiments/spd-ci-blocks2.md) | CI-fn transformer depth 4 → 2 blocks | speedup | killed | only ~4.8% faster — below 10% bench gate; CI fn ~5% of step |
