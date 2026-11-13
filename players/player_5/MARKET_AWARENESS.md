# Group 5: market awareness

This branch adapts two ideas from Group 6: learning per-dimension trait and
preference distributions, and reacting to the declining number of singles.
It keeps Group 5's comparison between known matches and continued search,
selective interest lists, proposal rules, and rejection cooldowns.

## Behavior

- **Shrinking market:** multiply the existing search estimate by
  `sqrt((last_seen_singles - 1) / (initial_people - 1))`, bounded to `[0, 1]`.
  This lowers the reservation for searching as people marry, even when the
  existing five-attempt cap is reached. It uses `RoundContext.singles`, not
  the candidate-list length: reserved repeat pairs and incompatible people
  can be absent from that list without having married.
- **Distribution learning:** estimate unknown trait and preference costs
  from per-dimension histograms, starting with a uniform prior of total
  weight 11. A person's first full date contributes once. A previously seen
  advertisement is removed when that date replaces it; neither repeat ads
  nor repeat dates add further samples.
- **Cautious ad evidence:** an advertisement has weight 0.25, and total ad
  evidence is capped at weight 2 per dimension and type. This prevents many
  selected or misleading advertisements from overwhelming the prior.
- **Using the estimates:** learned costs affect ranking other people's ads
  and the fallback partner-penalty estimate for suspiciously copied ideals.
  Our own advertisement remains fixed at startup, as the interface requires.
  Exactly mirrored ideals (three or more dimensions) do not train that
  fallback estimate. The worst observed own mismatch is still retained.

Only positive **estimated** scores can lead to a proposal, and a proposal
still needs at least two dates. Other players can misrepresent themselves,
so this cannot guarantee positive final engine-truth happiness.

`DateContext` does not reveal the current singles count during reserved
repeats. The player therefore uses its last observed count rather than
inventing an update. The learned distributions reflect observed claims,
not access to the hidden population distribution. The first date sample is
kept even if the person later changes their claims; repeat dates do not get
extra statistical weight.

## Reproduce the comparison

From the repository root:

```sh
uv run --frozen pytest -q
uv run --frozen ruff format --check players/player_5
uv run --frozen ruff check players/player_5
uv run --frozen python -m players.player_5.compare --seed 7 --out /tmp/group5-seed7.json
```

The runner loads the original Group 5 source from commit
`9e4c5d8c646013b50554e6ad39e0c2025d2b3f4f`, without changing the working tree.
It compares the baseline, market awareness only, learning only, and both
changes. Every variant starts with identical profiles, ideals, seat order,
and seed; the runner verifies the initial-market hash. Only Group 5 seats
change strategy. Results include all faults as well as Group 5 faults.

The 47 configurations cover the board **one factor at a time**, not as a
full Cartesian product. The reference is N=110, t=50, B=100, d=5, m=2,
mixed distribution, with ten seats per group plus ten random players.
The interest-list multiple is 4 and the per-call timeout is one second.

- Bonuses: 10, 50, 100, 150, 500; rounds: 5, 50, 150, 500.
- Dimensions: 1, 2, 5, 8, 10, 24; for each, no monotonic traits,
  `floor(d/2)` monotonic traits, and all monotonic traits, deduplicating overlaps.
- Distributions: uniform, normal, mixed, Group 9's submitted U-shape,
  Group 10's 90%-at-0/10%-at-10 row, its reversal, and Group 2's submitted row.
  Each submitted row is repeated across dimensions for that case.
- Population sizes: 20 (two per group), 22 (two per group plus two random),
  110, 330, 1100 (equal counts across the ten groups and random player).
- Additional rosters at N=110: all Group 5; 100 Group 5 plus ten random;
  11 Group 5 plus 99 random; all groups without random (11 each);
  and a 55/55 split against each other group.

Use `--case reference` or another case name to rerun a subset. Summary rates
give every configuration equal weight; they are not pooled across seats.
One seed provides a regression comparison, not evidence of a general
performance improvement. The board's full parameter combinations and
additional seeds require a broader evaluation.

## Seed 7 results

All 188 games (47 configurations times four variants) completed with zero
player faults. The full per-configuration results are in
[`seed7_comparison.json`](seed7_comparison.json).

| Variant | Average positive rate | Change vs baseline | Mean happiness | Better / tied / worse configurations |
| --- | ---: | ---: | ---: | ---: |
| Original Group 5 | 86.14% | — | 64.29 | — |
| Market awareness only | 86.03% | -0.11 percentage points | 64.33 | 7 / 34 / 6 |
| Distribution learning only | 85.45% | -0.69 percentage points | 63.33 | 9 / 26 / 12 |
| Both changes | 85.24% | -0.90 percentage points | 63.75 | 10 / 22 / 15 |

Better/tied/worse refers to positive-happiness rate, not mean happiness.
These results do **not** establish an improvement. Keep this as an
experimental feature branch pending team review and additional seeds;
do not interpret passing correctness tests as a performance win.
The combined variant had negative true happiness in two configurations
despite positive estimated scores, illustrating the limits of relying on
other players' projected profiles.

## Checks

The full test suite passes (279 tests, including 20 new Group 5 cases).
Group 5 passes formatting and lint checks. The repository-wide checks find
pre-existing formatting issues in six other files and three lint errors in
Groups 7 and 8; this branch changes only `players/player_5/`.
