# Track-2 ledger — experiment index

The human's whole report surface (plus [`baselines.md`](baselines.md)). One row per
experiment; the detailed card lives in `experiments/<id>.md` (from
[`TEMPLATE.md`](TEMPLATE.md)). Update the row whenever the stage changes.

**Claim type:** `speedup` (faster, same quality) or `simplification` (fewer moving parts,
similar quality). **Stage:** `proposed` / `running` / `confirmed` / `merged` / `killed` / `parked`.

| id | idea | claim | stage | headline result |
|---|---|---|---|---|
| [spd-ppgd-nwarmup0](experiments/spd-ppgd-nwarmup0-t1.md) | PPGD n_warmup_steps 2→0 | speedup | running | bench **28% faster** (650→468 ms/step b16); quality screening (`p-3105a340`) |
| [spd-ppgd-nwarmup1](experiments/spd-ppgd-nwarmup1.md) | PPGD n_warmup_steps 2→1 | speedup | running | bench **14% faster** (650→558 ms/step b16); quality screening (`p-ebc2de5b`) |
| [spd-ppgd-sign](experiments/spd-ppgd-sign.md) | PPGD source optimizer adam→sign | speedup | killed | only **3.5% faster** (650→627 ms/step b16) — below 5% floor |
