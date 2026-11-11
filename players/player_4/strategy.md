# Player 4 strategy (Group 4, final revised player)

## Idea in one line
A careful matchmaker: picky early, less picky late, and never accept a
marriage on a bad score.

## How it picks who to date
It reads every ad first. An ad shows one trait a person has and one trait
they want. The player ranks people by ad penalty (our full penalty plus
half of theirs, matching the payoff) and dates the best-looking ones first.

## How it decides to propose
- The bar is a falling quantile of first-date scores, from the 90th
  percentile early to the minimum late. It is flat for the first and last
  10 percent of rounds. It is not clamped at 0, so it tracks the actual
  score distribution even when scores are mostly negative.
- It only proposes above the bar plus a safety margin: 0.10 times the
  bonus early, decaying to 0.02 times the bonus late. The margin absorbs
  estimation error.
- It never proposes on an estimated score at or below the floor. The floor
  is the larger of 0.05 times the bonus and 1.0 point per dimension. The
  floor does not decay. Estimation error from a flattering projection does
  not shrink with the bonus; it scales with the number of dimensions, so a
  purely fractional floor is far too small at low bonus.

## How it handles copying and lying
- Mirror check: if a partner shows our ideal back, or states our own
  profile or our own projection as their ideal, that half of the estimate
  carries no information. It is replaced by that partner's own last
  informative value for that half if one exists, else by the running mean
  of that half across partners dated.
- Lie check: a partner who contradicts their ad or changes their shown
  traits between dates is flagged, and their penalty is multiplied by 1.2.
  The multiplier is skipped when both halves come from that partner's own
  past informative data, because that data is trustworthy (true traits do
  not change).

## Patience and second chances
- A bad first date gets a second look, not a burn. Only a second bad date
  marks a partner incompatible.
- If a partner proposed to us and was close to the bar, we accept their
  next proposal when the score clears the floor.
- After 2 failed proposals in a row, a partner is excluded from our list
  for 5 rounds. We still propose if we meet them by chance and they
  qualify.
- In the last 3 rounds, only known good partners are listed.

## What it shows about itself
A masked profile: regular dims clamped to [2, 8], monotonic dims truthful.
Its ad matches what it shows, and it states its true ideal honestly. No
inflated projection.

## Results (seed 1, 10-scenario sweep; seed 4444, 25-scenario comparison)
- Headline (20 players, 30 rounds, bonus 150, 5 dims): mean 87.0, married
  2 of 2, zero negatives.
- 25 scenarios vs Player 1 (seed 4444): mean wins 14 of 25, Player 1 8 of
  25, 3 ties. Total negatives 10 (same-group 3, low bonus 6, mid-bonus 1)
  vs Player 1's 0. Total married 1529 of 3054 vs Player 1's 1724 of 3054.
- Known limits: at 10 dims the bottleneck is pairing luck (missing the one
  good match in a short window), not the decision rule. At low and mid
  bonus the floor does not fully block small positive estimates that are
  truly negative.
