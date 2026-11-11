# Group 2 strategy modification plan

Status: planning only; no player code changed. Based on the repository's main branch inspected on November 6, 2026.

## 1. Scope and terminology

Revise `players/player_2/player.py` to learn separate distributions for partner traits and preferences, rank dates by risk-adjusted expected happiness, replace liar exclusion with a soft penalty, and experiment with consistent late-game deception.

Use N for participants, T for total dating rounds, d for dimensions, m for monotonic dimensions, and B for the marriage bonus. Our fixed true profile is x and our fixed ideal is y. A partner's unknown true profile is z and their unknown ideal is w. A “round” means one dating step; a “run” means the entire simulation.

All learned state must remain private to each player instance. Never read simulator truth, logs, or other instances' state.

## 2. Corrections to the current-strategy description

- The current normal prior is discretized over integers 0–10, with mean 5, standard deviation 2, and total pseudo-count weight 10. Thus N(5, 2) is ambiguous notation: its variance is 4.
- Current `_ad_score()` already treats the advertised own value and advertised desired value as known and fills the other dimensions using population expectations. Incorporating the partner's preferences is already present. The new work is changing the prior, separating the two population models, and changing ranking/decision policy.
- The current code learns one histogram per dimension from advertised own traits and projected traits, then uses that same histogram for both traits and preferences. It does not learn a separate preference histogram from stated ideals.
- True profiles and ideals are fixed and known to their owner from initialization. Advertisements are chosen once before round 1 and cannot be revised after dates. Do not assume `advertise()` receives other people's advertisements; candidate advertisements arrive through `choose_dates()`.
- Dates reveal projected traits and stated ideals, not verified truth. Name the new models “estimated trait distribution” and “estimated preference distribution.”
- The engine does not impose a lying penalty. Other strategies may react to inconsistencies; our new penalty is our own decision heuristic.
- Current observation duration already depends on both N and T: a unique-partner target of floor(sqrt(N)) and a round cap of ceil(0.37T). The requested change is a better joint schedule, not adding those dependencies for the first time.
- Current remembered-proposal reciprocation also permits an estimated score of zero, provided it has not decreased and no earlier rejection applies. Removing this exception should route the decision through the normal observation and acceptance rules.
- There is another early-proposal exception on the fourth or later date with a partner. “Repeated” counts total dates with that person, not necessarily consecutive dates. This must be addressed if observation is meant to prohibit all proposals.

## 3. Separate population models and updates

Maintain two d-by-11 weighted histograms:

1. Trait model Pz[i][v]: estimated frequency of true trait value v in dimension i, learned from imperfect reports.
2. Preference model Pw[i][v]: estimated frequency of desired value v in dimension i, learned from stated ideals.

Provisional initialization: seed each trait histogram at our own x_i and each preference histogram at our own y_i. This interprets “use our own value” as using the corresponding personal observation for each model, rather than using x_i to represent other people's desires. Treat these as weak seeds, not reliable population knowledge.

Update after each date, including automatic repeat dates. Each unique partner contributes at most one effective observation per dimension per model; a repeated report revises that partner's contribution rather than counting them as a new independent sample. Track effective sample counts separately for each model and dimension. Recompute model-dependent scores after updates, not only inside `choose_dates()`.

Use advertisements directly for that candidate's estimate. Provisionally exclude advertisements from population histogram updates: advertising the largest/extreme value creates selection bias. If ads are used globally, weight them separately and avoid counting an ad and a date report as independent people.

“Rolling” initially means cumulative weighted observations with revisions, not a fixed-size moving window. Compare forgetting/recency weighting as an experiment because the remaining singles pool changes over time.

**EXPERIMENT:** own-sample seed weight; smoothing versus point-mass initialization; using our ideal as the preference seed; minimum effective sample count for trusting the model; ad inclusion/weight; repeated-report revision policy; reliability weights; cumulative versus recency-weighted learning. A single personal sample is noisy, and selected dates are not a random population sample.

## 4. Candidate estimates and scoring

For a new partner, use a point mass at their advertised trait value for that dimension and their advertised desired value for its dimension. Use Pz and Pw for all other dimensions. After dates, use their reports and history, with uncertainty/reliability handling described below.

Define the per-dimension cost c_i(w, v):

- Monotonic dimension: max(w − v, 0)^2.
- Other dimension: (w − v)^2.

Compute:

    expected_happiness(j) = B
        − sum_i E[c_i(y_i, Z_ji)]
        − 0.5 * sum_i E[c_i(W_ji, x_i)]

Use expectations of squared costs over distributions, not costs evaluated only at estimated mean traits. These differ when uncertainty is present. For non-monotonic costs, E[(y − Z)^2] = (y − E[Z])^2 + Var(Z).

Define a separate decision score:

    decision_score(j) = expected_happiness(j) − liar_penalty(j)

Rank eligible candidates by this score, including both new and known partners in one ranking. Respect `self.list_limit` and use deterministic tie-breaking. The simulator interprets the returned names as an interest subset, not a guaranteed ranked assignment; ranking matters primarily for who is included. In small markets the cap can include everyone, so changing ordering alone may do little.

**EXPERIMENT:** interest subset size and score cutoff; minimum acceptable score; controlled exploration among new partners versus strictly highest-score selection; uncertainty treatment for reported values; revised acceptance threshold and its decay. Strict exploitation can prevent the observation phase from collecting diverse samples.

## 5. Advertisement policy

Advertise the truthful largest x_i among monotonic dimensions i < m. Break ties by dimension index. If m = 0, provisionally retain the existing most-extreme non-monotonic fallback.

The advertisement has a separate desired-trait field. The request does not specify changing it; provisionally retain the current truthful most-extreme non-monotonic desired trait, with the existing all-dimensions fallback.

**EXPERIMENT:** highest monotonic value versus current extreme non-monotonic advertisement; fallback when m = 0; alternative desired-trait selection. Highest monotonic is a hypothesis to test, not inherently a more robust signal: its usefulness depends on others' demands and their skepticism.

## 6. Replace liar exclusion with a soft penalty

Replace permanent `liars` filtering and automatic refusal with a per-partner suspicion/evidence record and a nonnegative penalty measured in happiness-score units.

Candidate evidence includes changed projections, changed stated ideals, contradiction of advertisements, and apparent copying. Copying/similarity is weak evidence because honest matches can resemble each other. Avoid treating all changes or similarities as equally convincing.

Apply the decision score consistently to candidate ranking, repeat requests, and proposal eligibility. Suspicion alone must not set `incompatible=True`. Provisionally also avoid permanent blocking from a temporarily negative risk-adjusted estimate; distinguish a revisable poor estimate from an irreversible incompatibility decision.

If unreliable reports are shrunk toward the population model as well as penalized, measure whether this double-counts risk. When deception is introduced, comparisons must use the projection and ideal we actually showed that partner, not always our true x and y.

**EXPERIMENT:** evidence weights; similarity cutoff or removal of similarity evidence; penalty scaling with d and B; penalty cap and decay; recovery after consistent reports; report shrinkage; whether any high-confidence negative-score condition warrants permanent incompatibility. The old discount on high monotonic observations must be explicitly retained, replaced, or removed and tested rather than silently stacked with the new penalty.

## 7. Observation and proposal policy

Remove the remembered-proposal shortcut. During observation, record proposals but do not automatically reciprocate. After observation, apply the ordinary acceptance policy even to someone who previously proposed.

Provisional interpretation: observation means no proposals, including disabling the fourth-date shortcut during this phase. Repeat requests remain possible, but can reduce unique-partner exploration. If the intent is only to remove the remembered-proposal exception, retaining the fourth-date exception is an alternative to confirm before implementation.

A candidate joint schedule to evaluate is:

    target_unique = min(N − 1, max(1, ceil(a * sqrt(N))), max(1, floor(b * T)))
    deadline = max(0, min(floor(c * T), T − reserve_rounds))

End observation when the unique-partner target is reached OR the deadline is reached. Permit immediate acceptance mode when the deadline is zero. Reserve enough time for at least two dates with a new partner if possible; a run with fewer than two rounds cannot produce a legal marriage. Count unique people, not repeated dates, toward learning targets.

Use updated round/time state during repeat dates too. A pending proposal is learned only after the current decisions, and marriage requires both sides to propose on the same date; receiving one is not a persistent offer the engine automatically accepts.

**EXPERIMENT:** coefficients a, b, c; reserved rounds; OR versus confidence-based termination; minimum effective observations; repeats during exploration; fourth-date shortcut after observation; whether acceptance thresholds should adapt to the ratio of remaining rounds to initial N. The existing gamma can already be small on round 1 when T is much smaller than N.

## 8. Conditional late-game deception

Interpret “lying” initially as changing projected traits, while continuing to state our true ideal. Lying about desired traits is a separate extension, not assumed in this plan.

Start a deceptive persona only on a first date with a partner, and only if all conditions hold:

1. The run is sufficiently late, with enough time left for a second date and possible marriage.
2. Preference estimates have enough effective observations.
3. Estimated demand differs materially from our true traits in at least one unadvertised dimension.
4. A proposed projection is predicted to improve the chance of a worthwhile marriage enough to justify detection risk.
5. Estimated true marriage happiness remains acceptable; a better-looking projection must not substitute for our true x in our own happiness calculation.

Never change the advertised own dimension. Keep every value a valid integer in 0–10. Store a per-partner projection on first meeting and reuse it on repeat dates; this also means a relationship begun honestly remains honest. “Only lie to new people” is interpreted as only initiating a lie with new people, not reverting to truth on the second date.

At `project()` time we have the partner's ad and prior population data, but not their current date's stated ideal. Do not use information that arrives later in `describe_ideal()` or `decide()` to choose today's projection.

For candidate projections, use the preference model to estimate the partner's perceived benefit. Keep this distinct from our true expected marriage happiness. One possible research objective is:

    gain = [P(marry | deceptive projection) − P(marry | honest projection)]
           * expected_true_happiness − estimated_detection_cost

This probability is not supplied by the engine and is not directly identifiable from a few dates. Start with an explicitly labeled response heuristic and validate it against different opponent policies; do not report heuristic score gain as measured marriage probability.

**EXPERIMENT:** late-game cutoff; minimum samples; demand-gap definition and threshold; number of modified dimensions; projection magnitude; minimum predicted gain; response model; detection cost; preserving personal ideals versus also misreporting them. Exact mirroring may itself trigger opponents' detectors even if the story is consistent.

## 9. Implementation sequence

1. Preserve the existing player as the experimental baseline; record its source revision and run settings.
2. Add separate trait/preference estimators, per-partner contributions, and expected-cost scoring.
3. Add the advertisement and unified candidate-ranking changes.
4. Replace liar exclusions and audit all permanent incompatibility paths.
5. Remove proposal shortcuts during observation and add the joint N/T schedule.
6. Add per-partner projection memory and optional deception, initially disabled for ablations.
7. Validate and compare components before choosing final thresholds. Keep changes inside Group 2's implementation and appropriate tests; do not change simulator rules.

## 10. Validation and experiments

Correctness checks: separate histogram updates; advertisement conditioning; expected squared-cost arithmetic; unique-partner counting; repeat-date updates; no suspicion-only blacklist; observation-phase proposal rules; second-date proposal legality; fixed advertisements; consistent per-partner projections; m = 0, m = d, N = 2, and very short runs; no shared state; reproducibility; no player faults.

Compare baseline, estimator-only, advertisement-only, soft-penalty-only, revised observation-only, all changes without deception, and all changes with deception. Use paired seeds and held-out seeds across small/large N, short/long T, different N/T ratios, bonuses, dimensions, monotonic counts, and trait distributions. Test both Group 2-only markets and mixed markets with honest and deceptive opponents. Self-play alone cannot establish resistance to other strategies.

Primary outcome: fraction of participants with positive true happiness. Secondary outcome: average true happiness. Also record marriage rate, negative marriages, marriage timing, unique partners observed, faults, runtime, false suspicion in honest markets, and deception activation/results. Report variability across seeds; do not select thresholds from one favorable run.

Every item marked EXPERIMENT remains a tunable hypothesis. Any new numeric cutoff or policy choice introduced during implementation must be added to this register with its rationale and comparison baseline.

## Sources

- [Current Group 2 player](https://github.com/ananyakapoor19/coms4444-marriage-market/blob/main/players/player_2/player.py)
- [Player interface and callback timing](https://github.com/ananyakapoor19/coms4444-marriage-market/blob/main/models/player.py)
- [Simulator documentation](https://github.com/ananyakapoor19/coms4444-marriage-market/blob/main/README.md)
- [Assignment rules](https://www.cs.columbia.edu/~kar/4444f26/node21.html)
