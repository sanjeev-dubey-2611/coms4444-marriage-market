

from collections import Counter
from math import sqrt

from models.player import (
    Advertisement,
    Candidate,
    DateContext,
    DateOutcome,
    Decision,
    GameContext,
    PlayerSnapshot,
    RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import MAX_VALUE, VALUES, Profile


class Player5(BasePlayer):
    # blacklisting behavior
    UNANSWERED_LIMIT = 2
    BLOCK_FIRST_DATE = 0.25

    # Deception assumptions
    LIE_PROJECT = 0
    LIE_IDEAL = 0
    BASE_LIE_TRAIT = 0.08
    BASE_LIE_IDEAL = 0.12

    # Population histogram prior
    PRIOR_WEIGHT = 12.0

    # History weighting, older observations count less
    HISTORY_DECAY = 0.70

    # Exploration and candidate selection
    EXPLORE_FRACTION = 0.30
    MIN_EXPLORE_SLOTS = 1
    CONFIDENCE_PENALTY = 0.65

    # Proposal thresholds
    BAR_END = 0.75
    SAFETY = 0.02
    ASPIRE = 0.0
    THINNING_PRESSURE = 0.45
    EXPLORE = 10
    EXPLORE_BAR = 0.80

    # Repeat dates
    AGAIN_FRACTION = 0.55
    INFO_AGAIN_WEIGHT = 0.12

    def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
        super().__init__(snapshot, ctx)

        self.observed: dict[int, list[Profile]] = {}
        self.stated: dict[int, list[Profile]] = {}
        self.ads: dict[int, Advertisement] = {}

        self.estimates: dict[int, float] = {}
        self.uncertainties: dict[int, float] = {}

        self.date_counts: dict[int, int] = {}
        self.last_seen_round: dict[int, int] = {}

        self.i_proposed: dict[int, bool] = {}
        self.they_proposed_ever: set[int] = set()
        self.their_proposal_rounds: dict[int, list[int]] = {}
        self.my_proposal_rounds: dict[int, list[int]] = {}

        self.their_unanswered: dict[int, int] = {}
        self.my_unanswered: dict[int, int] = {}

        self.blocked: set[int] = set()
        self.singles = self.people
        self.current_round = 0

   
    # Population histograms
  

    def _histogram(
        self,
        source: dict[int, list[Profile]],
        dim: int,
    ) -> dict[int, float]:
        """Population estimate using each person's latest observation.

        The prior is deliberately stronger than before because the people
        we have dated are a selected sample, not a random sample.
        """
        counts: Counter[int] = Counter(
            entries[-1][dim]
            for entries in source.values()
            if entries
        )

        total = sum(counts.values()) + self.PRIOR_WEIGHT
        spread = self.PRIOR_WEIGHT / len(VALUES)

        return {
            v: (counts[v] + spread) / total
            for v in VALUES
        }

    def trait_histogram(self, dim: int) -> dict[int, float]:
        return self._histogram(self.observed, dim)

    def desire_histogram(self, dim: int) -> dict[int, float]:
        return self._histogram(self.stated, dim)

    
    # Compatibility calculation
    

    def _decrement(self, dim: int, want: int, have: int) -> int:
        gap = want - have

        if dim < self.monotonic and gap <= 0:
            return 0

        return gap * gap

    def _weighted_distribution(
        self,
        entries: list[Profile],
        dim: int,
        lie: float,
        prior: dict[int, float],
    ) -> dict[int, float]:
        """Discount older observations rather than counting dates equally."""
        if not entries:
            return prior.copy()

        weights = [
            self.HISTORY_DECAY ** (len(entries) - 1 - i)
            for i in range(len(entries))
        ]

        total_weight = sum(weights)
        counts = {v: 0.0 for v in VALUES}

        for entry, weight in zip(entries, weights):
            counts[entry[dim]] += weight

        empirical = {
            v: counts[v] / total_weight
            for v in VALUES
        }

        return {
            v: (1.0 - lie) * empirical[v] + lie * prior[v]
            for v in VALUES
        }

    def _varies(self, entries: list[Profile], dim: int) -> bool:
        return len({entry[dim] for entry in entries}) > 1

    def _trait_lie(self, q: int, dim: int) -> float:
        entries = self.observed[q]
        p = self.BASE_LIE_TRAIT

        # Changing projections are evidence of unreliability, but consistent
        # projections are NOT proof of honesty
        if self._varies(entries, dim):
            p = max(p, 0.55)

        ad = self.ads.get(q)
        if ad is not None and ad.own_dim == dim:
            if ad.own_value != entries[-1][dim]:
                p = max(p, 0.65)

        # Extreme monotonic values are possible, but less likely under the
        # observed population distribution
        if dim < self.monotonic:
            hist = self.trait_histogram(dim)
            value = entries[-1][dim]
            tail = sum(prob for v, prob in hist.items() if v >= value)

            if tail < 0.05:
                p = max(p, 0.30)

        return min(p, 0.85)

    def _ideal_lie(self, q: int, dim: int) -> float:
        entries = self.stated[q]
        p = self.BASE_LIE_IDEAL

        if self._varies(entries, dim):
            p = max(p, 0.55)

        ad = self.ads.get(q)
        if ad is not None and ad.seek_dim == dim:
            if ad.seek_value != entries[-1][dim]:
                p = max(p, 0.45)

        return min(p, 0.85)

    def trait_distribution(self, q: int, dim: int) -> dict[int, float]:
        return self._weighted_distribution(
            self.observed[q],
            dim,
            self._trait_lie(q, dim),
            self.trait_histogram(dim),
        )

    def desire_distribution(self, q: int, dim: int) -> dict[int, float]:
        return self._weighted_distribution(
            self.stated[q],
            dim,
            self._ideal_lie(q, dim),
            self.desire_histogram(dim),
        )

    def _compatibility_components(self, q: int):
        """Return expected happiness and uncertainty in that estimate."""
        total = float(self.bonus)
        variance = 0.0

        for dim in range(self.dims):
            traits = self.trait_distribution(q, dim)
            trait_penalties = {
                v: self._decrement(dim, self.ideal[dim], v)
                for v in VALUES
            }

            mean_trait = sum(
                p * trait_penalties[v]
                for v, p in traits.items()
            )

            var_trait = sum(
                p * (trait_penalties[v] - mean_trait) ** 2
                for v, p in traits.items()
            )

            total -= mean_trait
            variance += var_trait

            wants = self.desire_distribution(q, dim)
            desire_penalties = {
                v: self._decrement(dim, v, self.profile[dim])
                for v in VALUES
            }

            mean_desire = sum(
                p * desire_penalties[v]
                for v, p in wants.items()
            )

            var_desire = sum(
                p * (desire_penalties[v] - mean_desire) ** 2
                for v, p in wants.items()
            )

            total -= 0.5 * mean_desire
            variance += 0.25 * var_desire

        # Repeated observations are correlated, so the effective sample
        # size grows more slowly than the number of dates
        n = max(1, len(self.observed.get(q, [])))
        effective_n = 1.0 + 0.5 * (n - 1)

        uncertainty = sqrt(max(0.0, variance)) / sqrt(effective_n)

        return total, uncertainty

    def estimate(self, q: int) -> float:
        return self._compatibility_components(q)[0]

    def confidence_score(self, q: int) -> float:
        """Conservative compatibility estimate for ranking decisions."""
        mean, uncertainty = self._compatibility_components(q)
        return mean - self.CONFIDENCE_PENALTY * uncertainty

    def expected_random(self) -> float:
        total = float(self.bonus)

        for dim in range(self.dims):
            traits = self.trait_histogram(dim)
            total -= sum(
                p * self._decrement(dim, self.ideal[dim], v)
                for v, p in traits.items()
            )

            wants = self.desire_histogram(dim)
            total -= 0.5 * sum(
                p * self._decrement(dim, w, self.profile[dim])
                for w, p in wants.items()
            )

        return total

    
    # Mutual proposal estimation
    

    def mutual_proposal_probability(self, q: int, date_number: int) -> float:
        """Estimate whether q is likely to propose on the current date.

        Available signals:
        - Whether q has proposed before.
        - How recently q proposed.
        - Whether we have reached repeat dates.

        This is intentionally a heuristic, not a claim that we can observe
        private intentions or repeat-date requests.
        """
        probability = 0.12

        # Reaching another date is evidence of continued mutual interest.
        probability += min(0.22, max(0, date_number - 1) * 0.08)

        proposal_rounds = self.their_proposal_rounds.get(q, [])

        if proposal_rounds:
            probability += 0.20

            age = self.current_round - proposal_rounds[-1]

            if age <= 2:
                probability += 0.22
            elif age <= 5:
                probability += 0.12
            elif age <= 10:
                probability += 0.05

            # Repeated proposals are stronger evidence than one old proposal
            probability += min(0.12, 0.04 * (len(proposal_rounds) - 1))

        # If we have repeatedly proposed without reciprocation, reduce
        # our expectation that this person will propose
        unanswered = self.my_unanswered.get(q, 0)
        probability -= 0.12 * unanswered

        return min(0.85, max(0.05, probability))

    
    # Proposal thresholds
    

    def _pressure(self, rounds_left: int) -> float:
        elapsed = (
            (self.rounds - rounds_left) / max(1, self.rounds)
        )

        time_pressure = elapsed / self.BAR_END

        thinning = self.THINNING_PRESSURE * (
            1 - self.singles / max(1, self.people)
        )

        return min(1.0, max(time_pressure, thinning))

    def proposal_bar(self, rounds_left: int, q: int | None = None) -> float:
        safety = self.SAFETY * self.bonus

        # Use confidence-adjusted estimates so one optimistic observation
        # cannot inflate the threshold for every candidate
        others = sorted(
            self.confidence_score(k)
            for k in self.estimates
            if k != q and k not in self.blocked
        )

        reference = self.expected_random()

        if others:
            reference = max(reference, others[len(others) // 2])

        reference = min(
            max(reference, self.ASPIRE * self.bonus),
            float(self.bonus),
        )

        bar = max(
            safety,
            (1 - self._pressure(rounds_left)) * reference,
        )

        return bar

    
    # Advertisements
    

    def advertise(self) -> Advertisement:
        """Advertise a marketable real trait and an achievable preference."""
        if not self.monotonic:
            return Advertisement(
                own_dim=0,
                own_value=self.profile[0],
                seek_dim=0,
                seek_value=self.ideal[0],
            )

        # Pick the monotonic trait with the strongest market position
        # A high percentile is more meaningful than simply comparing
        # raw values across dimensions
        def percentile(dim: int, value: int) -> float:
            hist = self.trait_histogram(dim)
            return sum(p for v, p in hist.items() if v <= value)

        own_dim = max(
            range(self.monotonic),
            key=lambda d: percentile(d, self.profile[d]),
        )

        # Advertise the preference that the largest share of the population
        # is expected to satisfy. This makes the ad more attainable
        def satisfaction_rate(dim: int) -> float:
            hist = self.trait_histogram(dim)
            return sum(
                p for v, p in hist.items()
                if v >= self.ideal[dim]
            )

        seek_dim = max(
            range(self.monotonic),
            key=satisfaction_rate,
        )

        return Advertisement(
            own_dim=own_dim,
            own_value=self.profile[own_dim],
            seek_dim=seek_dim,
            seek_value=self.ideal[seek_dim],
        )

    def _ad_cost(self, ad: Advertisement) -> float:
        """Estimate the compatibility information available from an ad."""
        own = self._decrement(
            ad.own_dim,
            self.ideal[ad.own_dim],
            ad.own_value,
        )

        their = self._decrement(
            ad.seek_dim,
            ad.seek_value,
            self.profile[ad.seek_dim],
        )

        return own + 0.5 * their

    def _unknown_candidate_score(self, ad: Advertisement) -> float:
        """Prior-based compatibility estimate for someone not yet dated."""
        score = self.expected_random()

        # The ad reveals two dimensions. Replace their population-average
        # contribution with the advertised values
        dim = ad.own_dim
        hist = self.trait_histogram(dim)

        old = sum(
            p * self._decrement(dim, self.ideal[dim], v)
            for v, p in hist.items()
        )
        new = self._decrement(dim, self.ideal[dim], ad.own_value)
        score += old - new

        dim = ad.seek_dim
        hist = self.desire_histogram(dim)

        old = 0.5 * sum(
            p * self._decrement(dim, w, self.profile[dim])
            for w, p in hist.items()
        )
        new = 0.5 * self._decrement(
            dim,
            ad.seek_value,
            self.profile[dim],
        )
        score += old - new

        return score

    
    # Candidate selection
    

    def choose_dates(
        self,
        candidates: tuple[Candidate, ...],
        turn: RoundContext,
    ) -> list[int]:
        self.singles = turn.singles
        self.current_round = turn.round

        known = []
        unknown = []

        for c in candidates:
            self.ads[c.index] = c.ad

            if c.index in self.blocked:
                continue

            if c.index in self.estimates:
                mean = self.estimates[c.index]
                confidence = self.confidence_score(c.index)

                # Keep known candidates with positive expected compatibility
                if mean > 0:
                    known.append((confidence, mean, c.index))
            else:
                prior_score = self._unknown_candidate_score(c.ad)
                unknown.append((prior_score, c.index))

        known.sort(reverse=True)
        unknown.sort(reverse=True)

        limit = self.list_limit

        # Explicitly reserve slots for new candidates. This prevents the
        # known-candidate list from permanently crowding out exploration
        explore_slots = min(
            len(unknown),
            max(
                self.MIN_EXPLORE_SLOTS if unknown and limit > 0 else 0,
                int(limit * self.EXPLORE_FRACTION),
            ),
        )

        selected = [
            index for _, index in unknown[:explore_slots]
        ]

        remaining = limit - len(selected)

        selected.extend(
            index for _, _, index in known[:remaining]
        )

        # If we do not have enough known candidates, use additional unknowns
        if len(selected) < limit:
            already_selected = set(selected)

            for _, index in unknown:
                if index not in already_selected:
                    selected.append(index)
                    already_selected.add(index)

                if len(selected) >= limit:
                    break

        return selected[:limit]

    
    # Player interface
    

    def project(self, date: DateContext) -> Profile:
        if not self.LIE_PROJECT:
            return self.profile

        heard = self.stated.get(date.partner)

        if heard:
            return tuple(heard[-1])

        shown = list(self.profile)

        for dim in range(self.monotonic):
            shown[dim] = min(MAX_VALUE, shown[dim] + 1)

        ad = date.partner_ad
        shown[ad.seek_dim] = min(
            MAX_VALUE,
            ad.seek_value + (
                2 if ad.seek_dim < self.monotonic else 0
            ),
        )

        return tuple(shown)

    def describe_ideal(
        self,
        date: DateContext,
        observed: Profile,
    ) -> Profile:
        if not self.LIE_IDEAL:
            return self.ideal

        return tuple(observed)

    def decide(
        self,
        date: DateContext,
        observed: Profile,
        stated_ideal: Profile,
    ) -> Decision:
        q = date.partner
        self.current_round = date.round

        self.observed.setdefault(q, []).append(observed)
        self.stated.setdefault(q, []).append(stated_ideal)

        self.ads[q] = date.partner_ad
        self.date_counts[q] = self.date_counts.get(q, 0) + 1
        self.last_seen_round[q] = date.round
        self.i_proposed[q] = False

        if q in self.blocked:
            return Decision(incompatible=True)

        mean, uncertainty = self._compatibility_components(q)
        self.estimates[q] = mean
        self.uncertainties[q] = uncertainty

        # Preserve the original early-blacklisting rule
        block_line = (
            -self.BLOCK_FIRST_DATE * self.bonus
            if date.date_number == 1
            else 0.0
        )

        if mean < block_line:
            self.blocked.add(q)
            return Decision(incompatible=True)

        bar = self.proposal_bar(date.rounds_left, q)

        # Early in the game, require a strong match unless this person
        # has already shown interest
        if (
            date.round <= self.EXPLORE
            and q not in self.they_proposed_ever
        ):
            bar = max(bar, self.EXPLORE_BAR * self.bonus)

        # A recent proposal is useful evidence, but never a guarantee
        mutual_probability = self.mutual_proposal_probability(
            q,
            date.date_number,
        )

        # Lower the bar modestly when there is evidence of mutual interest
        # Do not multiply the happiness estimate by probability: the bar
        # already represents the value of continuing to wait
        adjusted_bar = bar * (1.0 - 0.35 * mutual_probability)

        propose = (
            date.date_number >= 2
            and mean > adjusted_bar
        )

        # Request another date when the match is promising or when another
        # observation could materially reduce uncertainty
        again_bar = max(
            self.SAFETY * self.bonus,
            self.AGAIN_FRACTION * adjusted_bar,
        )

        information_value = (
            self.INFO_AGAIN_WEIGHT * uncertainty
        )

        again = (
            mean + information_value > again_bar
        )

        if propose:
            self.i_proposed[q] = True
            self.my_proposal_rounds.setdefault(q, []).append(date.round)

        return Decision(
            again=again or propose,
            propose=propose,
        )

    def on_outcome(self, outcome: DateOutcome) -> None:
        q = outcome.partner

        if outcome.married:
            return

        mine = self.i_proposed.get(q, False)

        if outcome.partner_proposed:
            self.they_proposed_ever.add(q)
            self.their_proposal_rounds.setdefault(q, []).append(
                self.current_round
            )

            if not mine:
                self.their_unanswered[q] = (
                    self.their_unanswered.get(q, 0) + 1
                )

                if self.their_unanswered[q] >= self.UNANSWERED_LIMIT:
                    self.blocked.add(q)

        if mine and not outcome.partner_proposed:
            self.my_unanswered[q] = (
                self.my_unanswered.get(q, 0) + 1
            )

            if self.my_unanswered[q] >= self.UNANSWERED_LIMIT:
                self.blocked.add(q)
