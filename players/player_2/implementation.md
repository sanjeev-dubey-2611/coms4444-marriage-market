# Player 2: implementation write-up

This replaces the old Player 2, which observed for the first 37% of rounds and then used a bar that fell in a straight line. The new player follows `plan.md`: it is honest, it detects liars, it learns the value distributions, and it observes before it chooses. All changes are in `players/player_2/player.py`.

## What it keeps track of

For each person it has dated, the player stores:

- the latest score estimate from the handout formula, using their projection and their stated ideal;
- every projection they showed us;
- every ideal they described to us;
- every ideal we described to them;
- whether they ever proposed to us;
- whether they have been flagged as a liar.

It also keeps one histogram per dimension, and the best estimate it has seen so far.

## Distributions

- Each dimension starts as a discretised normal around 5 with a standard deviation of 2. The prior is worth 10 observations (`PRIOR_WEIGHT`), so a few rounds of data cannot swing it too far.
- Each round, `choose_dates` adds the `own_dim`/`own_value` of every ad it has not seen before. `decide` adds the first projection of every new partner, one value per dimension. Each person counts once, so repeat dates do not bias the counts.
- **Liar adjustment.** In monotonic dimensions, a value above 5 counts for less. The weight falls linearly to 0.5 at a value of 10 (`MONO_LIAR_DISCOUNT`). Non-monotonic values count fully, since lying there hurts both sides.
- At the start of each round the model draws 200 synthetic people from the histograms, each with a profile and an ideal, and scores them with the formula. These scores, together with the real estimates, form the score distribution behind the threshold. The model also stores the expected cost of each dimension, which is used to rank people we have not dated yet by their ad.

## Honesty

- `advertise` shows the non-monotonic dimension where our true value is furthest from 5, so the highest or the lowest value. It asks for the non-monotonic dimension where our true ideal is furthest from 5. If there are no non-monotonic dimensions, it falls back to every dimension.
- `project` keeps the default, which is the true profile. `describe_ideal` always returns the true ideal and records it so liar detection can use it.

## Liar detection

A partner is flagged as a liar if any of these is true:

1. their projection differs from one they showed us on an earlier date;
2. their stated ideal differs from one they gave us on an earlier date;
3. their stated ideal matches our projection on the same date, either exactly or with a centred cosine similarity of at least 0.98;
4. their projection matches an ideal we described on an earlier date, by the same test.

The cosine similarity is taken after subtracting 5 from every value. Raw 0-10 vectors are all positive, so their plain cosine is high even for unrelated people. The cutoff is `SIMILARITY_CUTOFF`.

Once flagged, a liar is never proposed to, never asked for again, and dropped from the `choose_dates` list.

## Observe-then-choose

- **Observe phase.** The player observes until it has dated `floor(sqrt(N))` unique people. Repeat dates do not count. As a safety net, observing also ends after 37% of rounds. This cap is not in the plan. It stops a short run with a large N from spending every round observing.
- **Threshold.** `gamma = chances / (N - 1)` where `chances = min(rounds_left, singles - 1)`. The threshold is the score at quantile `gamma * 0.95` of all known scores, sampled and real, and never below 0. It drops as rounds run out and as singles marry.

## Decisions (`decide`)

The player keeps every estimate it has made for each partner. A score has "decreased" when today's estimate is below the best earlier estimate for that partner.

1. If the estimate is negative, mark them incompatible.
2. **Long repeat streaks.** Once we have had more than `REPEAT_PROPOSE_AFTER` dates with someone (3 by default):
   - if their score decreased, mark them incompatible;
   - if their score is positive and has not decreased, ask again and propose. This applies in both phases, unless they are a liar.
3. If they are a liar, return an empty decision.
4. **They proposed to us.** If they proposed on an earlier date and their score has not decreased, ask again and propose. This applies in both phases. Before this change, the decide phase ignored their proposal and only checked the threshold.
5. In the observe phase, ask again only if this is the best estimate seen so far.
6. In the decide phase:
   - if the estimate is positive and meets the threshold, ask again and propose;
   - otherwise ask again only if this is the best estimate seen so far.
7. A proposal is only sent from the second date on. On a first date, a qualifying person is asked for a repeat so a proposal can follow.

The engine only reports a proposal after both sides decide. So "they proposed" always means a proposal from an earlier date with that person.

Positive people who are not the best and not liars get an empty decision. They stay on the `choose_dates` list.

## Choosing dates (`choose_dates`)

Liars are always excluded. People we have not dated are ranked by their expected score given their ad.

- **Observe phase:** people we have not dated come first, so the unique count grows. Known people with a positive estimate follow, best first.
- **Decide phase:** known people at or above the threshold come first, then people we have not dated, then the other known positives.

The list is cut at `self.list_limit`.

## Results

Eight seeds, 10 Player 2s, 10 greedy and 10 random players, the default mixed distribution, with no faults:

| Setting    | New mean | New positive rate | Old mean | Old positive rate |
| ---------- | -------- | ----------------- | -------- | ----------------- |
| 100 rounds | 46.3     | 0.99              | 49.1     | 0.91              |
| 10 rounds  | 40.9     | 0.90              | 32.7     | 0.71              |

The new player does better on the primary goal, which is ending with positive happiness. On long runs its mean is a little lower. Lint passes and all 229 tests pass.

### After adding the repeat-date and proposal rules

When 10 copies of our player ran alone for 50 rounds, couples kept asking each other for repeat dates and only married near the end. Eight seeds, before and after the two new rules:

| Setting                         | Mean | Positive rate | Average marriage round |
| ------------------------------- | ---- | ------------- | ---------------------- |
| Before, 10 Player 2s, 50 rounds | 33.8 | 0.80          | 44.3                   |
| After, 10 Player 2s, 50 rounds  | 34.3 | 0.82          | 4.3                    |
| After, mixed roster, 100 rounds | 45.4 | 0.99          | not measured           |

About two people per seed still end unmarried. In every seed, none of the leftover pairs would have given both sides positive happiness, so these are true mismatches and not missed marriages. Limits of 2, 3 and 5 gave nearly the same results. Only the marriage round changed, so the default stays at 3.

## Tunable constants

These are at the top of `player.py`: `PRIOR_WEIGHT`, `PRIOR_SD`, `MONO_LIAR_DISCOUNT`, `SAMPLE_SIZE`, `TOP_QUANTILE`, `OBSERVE_ROUND_CAP`, `REPEAT_PROPOSE_AFTER` and `SIMILARITY_CUTOFF`.
