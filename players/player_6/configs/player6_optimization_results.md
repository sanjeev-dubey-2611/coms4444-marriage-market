# Player6 parameter optimization — November 9, 2026

Changed `EARLY_EXCEPTION_RATIO` from **0.65 to 0.50**. Retained the other defaults
after evaluating them. Positive final happiness is the primary objective; mean
happiness breaks ties. This favors more positive outcomes at some cost to their size.

| Parameter | Original | Training choice | Adopted |
|---|---:|---:|---:|
| EXPLORE_FRACTION | 0.30 | 0.15 | 0.30 |
| EARLY_EXCEPTION_RATIO | 0.65 | 0.50 | **0.50** |
| PRIOR_STRENGTH | 3 | 1 | 3 |
| EARLY_CONFIDENCE | 0.72 | 0.72 | 0.72 |
| END_MARGIN | 0.01 | 0.01 | 0.01 |
| ONE_SIDED_LIMIT | 3 | 1 | 3 |
| REFERENCE_FLOOR_RATIO | 0.35 | 0.50 | 0.35 |
| DECAY_EXPONENT | 1.4 | 1.0 | 1.4 |
| AD_WEIGHT | 0.25 | 0.50 | 0.25 |
| DISCOVERY_BONUS_RATIO | 0.10 | 0.10 | 0.10 |

The four previously hardcoded constants are now named class attributes so the
runner actually varies them. Baseline behavior was preserved during experiments.

## Experiments

Seven arenas: self-play, greedy opposition, all groups, low/high marriage bonuses,
uniform traits, and the group 6 distribution. Identical seeds and roster seats
are used across variants. All Player6 seats use the tested constants.

1. `player6_optimization.json`: 22 variants × 7 arenas × 10 seeds (11–20),
   **1,540 games**. Each constant varied separately; training ties prefer baseline.
2. `player6_validation.json`: original, combined training choices, and best single
   training choice (linear decay), 20 seeds (101–120), **420 games**.
   Both alternatives lost to baseline on positive happiness.
3. `player6_parameter_validation.json`: original plus seven individual training
   winners, same 20 validation seeds, **1,120 games**. Early exception 0.50 won;
   proposal limit 1 slightly improved mean happiness with unchanged positive rate.
4. `player6_final_validation.json`: original, early exception 0.50, and early
   exception 0.50 plus proposal limit 1, 30 fresh seeds (201–230), **630 games**.

Total: **3,710 games, zero engine/player faults**.

## Final comparison

| Settings | Positive happiness | Mean happiness |
|---|---:|---:|
| Original defaults | 73.48% | 52.17 |
| Early exception 0.50 — adopted | **75.05%** | 50.82 |
| Early exception 0.50 + proposal limit 1 | 74.10% | 50.95 |

Adopted setting improves positive rate by **1.57 percentage points**, while mean
happiness falls by **1.36**. Linear decay's training advantage did not generalize,
so the exponent stays at 1.4. Individually selected constants did not combine well.
The final set selected between two finalists; this is empirical tuning over a
limited search space, not proof of optimality or a fully untouched final test.
The paired seed bootstrap 95% interval for the adopted positive-rate improvement
is approximately **−0.14 to +3.24 percentage points**, which includes zero.
The observed improvement therefore remains uncertain.

Raw data, ranked CSVs, and PNG/SVG graphs live under `results/` in directories
matching the config names, except `player6_validation.json` outputs to
`player6_optimization_validation/`. Final `selection.json` records original and
adopted constants and a paired bootstrap interval that resamples whole seeds,
keeping arenas within a seed together. All summaries weight games equally.

These configs were run before updating the production default. The original
baseline constants are preserved in each `experiment.json`; future reruns use
the then-current class defaults as their baseline.
