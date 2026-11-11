# Player6 tuning

Run from the repository root using the project's environment (some opponents
require NumPy from the dev dependencies):

```sh
uv run --group dev tune_player6.py --output results/player6_experiment_1
```

The default config tests 20 variants (including the unchanged baseline), seven
arenas, and five seeds: 700 games. It varies one constant at a time to make effects
interpretable. All Player6 seats use the tested variant; other players are unchanged.
Production constants are never updated automatically.

For a quick smoke run:

```sh
uv run --group dev tune_player6.py --max-variants 2 --seeds 1 --output results/player6_smoke
```

Output directories must be new to prevent accidental overwrites:

- `experiment.json`: configuration, resolved constants, seeds, and player source hashes.
- `runs.jsonl`: one flushed record per game, including constants, baseline deltas,
  full engine results for every player, profiles, outcomes, and faults.
- `summary.csv`: completed variants ranked by Player6 positive-happiness rate,
  then mean happiness. Includes marriage rate, across-game standard deviations,
  paired baseline deltas, and fault counts. Variant 0 is the baseline.
- `graphs/`: automatic PNG and SVG charts for ranking, parameter sensitivity,
  changes by arena, the happiness/positive-rate tradeoff, and a focused
  `early_exception` chart showing both positive rate and mean happiness.

To sweep all constants around the updated 0.50 early-exception strategy and
test early-exception ratios 0.25–0.65 in steps of 0.05:

```sh
uv run --group dev tune_player6.py --config configs/player6_updated_strategy.json \
  --output results/player6_updated_strategy_new
```

This config tests 28 variants across seven arenas and ten seeds (301–310),
for 1,960 games. It compares each constant separately against the current
production baseline; it does not change production defaults automatically.

Regenerate graphs from an existing experiment without rerunning games:

```sh
uv run --group dev plot_player6.py results/player6_tuning
```

Summary metrics average Player6 seats within each game, then give every
arena/seed game equal weight. They are actual final happiness, not predicted scores.
Standard deviations describe spread across games, not confidence intervals.
Faulty runs remain in the data and summaries; inspect fault counts before choosing.
Engine timeouts can introduce machine-dependent faults.

Edit `configs/player6_tuning.json` or supply `--config PATH`. Arena `settings`
override `defaults`. Distribution values support presets and dimension maps such
as `{"0": "6", "1": "6"}`. Keep rosters identical across variants: the runner
uses the same seeds and seats for paired comparisons. Player6's learned field
distribution remains private; selecting group 6's distribution changes the arena.

After identifying promising constants, narrow `parameters` to two or three
variables and set `mode` to `grid` to test interactions. A full grid multiplies
all value-list lengths, so don't use the entire default search space at once.
Use additional seeds (`--seeds 101 102 103 104 105`) as held-out validation before
adopting values; the tuning winner is not necessarily best on unseen markets.

The named constants in Player6 also include `END_MARGIN`, which can be added to
the search. The refactor preserves every production default, including exponent 1.4.
