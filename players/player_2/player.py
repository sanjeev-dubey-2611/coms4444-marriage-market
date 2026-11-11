"""Group 2: population learning, soft suspicion, and conditional deception.

Defaults selected using mixed markets with N=50, B=150, T=250.
All learned state is private to each instance.
"""

import math
from dataclasses import dataclass

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
from models.profile import VALUES, Profile, decrement


@dataclass(frozen=True)
class StrategyConfig:
    # Population estimation.
    seed_weight: float = 1.0
    smoothing: float = 0.0

    # Observation phase.
    observe_sqrt: float = 0.5
    observe_sample_fraction: float = 0.25
    observe_round_fraction: float = 0.37
    reserve_rounds: int = 2

    # Date selection and acceptance.
    interest_multiple: float = 1.0
    sample_size: int = 50
    top_quantile: float = 0.95

    # Soft suspicion penalty.
    change_weight: float = 1.0
    ad_weight: float = 1.0
    copy_weight: float = 0.15
    similarity_cutoff: float = 0.98
    suspicion_decay: float = 0.8
    suspicion_cap: float = 3.0
    penalty_per_dimension: float = 2.0

    # Conditional deception.
    deception_enabled: bool = False
    late_fraction: float = 0.7
    min_samples: int = 5
    demand_gap: float = 4.0
    max_shift: int = 2
    max_lie_dimensions: int = 2
    min_perceived_gain: float = 5.0


def similar(a: Profile, b: Profile, cutoff: float) -> bool:
    if tuple(a) == tuple(b):
        return True

    ca = [v - 5 for v in a]
    cb = [v - 5 for v in b]
    norm = math.sqrt(
        sum(v * v for v in ca) * sum(v * v for v in cb)
    )

    return bool(
        norm
        and sum(x * y for x, y in zip(ca, cb, strict=True)) / norm
        >= cutoff
    )


class Population:
    """One sample per partner; repeat reports replace earlier reports."""

    def __init__(self, own: Profile, config: StrategyConfig) -> None:
        self.counts = [
            [config.smoothing for _ in VALUES] for _ in own
        ]

        for row, value in zip(self.counts, own, strict=True):
            row[value] += config.seed_weight

        self.reports: dict[int, Profile] = {}
        self.effective_samples = [0] * len(own)

    def update(self, partner: int, report: Profile) -> None:
        previous = self.reports.get(partner)

        for dim, value in enumerate(report):
            if previous is not None:
                self.counts[dim][previous[dim]] -= 1.0
            else:
                self.effective_samples[dim] += 1

            self.counts[dim][value] += 1.0

        self.reports[partner] = tuple(report)

    def probabilities(self) -> list[list[float]]:
        return [
            [value / sum(row) for value in row]
            for row in self.counts
        ]


class Player2(BasePlayer):
    def __init__(
        self,
        snapshot: PlayerSnapshot,
        ctx: GameContext,
        config: StrategyConfig | None = None,
    ) -> None:
        super().__init__(snapshot, ctx)

        self.config = config or StrategyConfig()
        self.traits = Population(self.profile, self.config)
        self.preferences = Population(self.ideal, self.config)

        self.suspicion: dict[int, float] = {}
        self.evidence: dict[int, dict[str, float]] = {}
        self.estimates: dict[int, float] = {}
        self.personas: dict[int, Profile] = {}
        self.my_described: dict[int, Profile] = {}
        self.proposed_to_me: set[int] = set()
        self.deceptions_started = 0

        # Last observable count; repeat-date contexts do not provide it.
        self.singles = self.people

        self.observe_target, self.observe_deadline = (
            self._observation_limits()
        )
        self.observing = self.observe_deadline > 0
        self.ad = self._advertisement()
        self._refresh_model()

    def _observation_limits(self) -> tuple[int, int]:
        c = self.config

        target = min(
            self.people - 1,
            max(1, math.ceil(c.observe_sqrt * math.sqrt(self.people))),
            max(1, math.floor(c.observe_sample_fraction * self.rounds)),
        )

        deadline = max(
            0,
            min(
                math.floor(c.observe_round_fraction * self.rounds),
                self.rounds - c.reserve_rounds,
            ),
        )

        return target, deadline

    def _update_phase(self, round_number: int) -> None:
        if (
            len(self.traits.reports) >= self.observe_target
            or round_number >= self.observe_deadline
        ):
            self.observing = False

    def _advertisement(self) -> Advertisement:
        free = range(self.monotonic, self.dims) or range(self.dims)

        if self.monotonic:
            own = max(
                range(self.monotonic),
                key=lambda i: (self.profile[i], -i),
            )
        else:
            own = max(
                free,
                key=lambda i: (abs(self.profile[i] - 5), -i),
            )

        seek = max(
            free,
            key=lambda i: (abs(self.ideal[i] - 5), -i),
        )

        return Advertisement(
            own,
            self.profile[own],
            seek,
            self.ideal[seek],
        )

    def advertise(self) -> Advertisement:
        return self.ad

    def _gap(self, dim: int, want: int, have: int) -> int:
        gap = want - have
        return max(0, gap) ** 2 if dim < self.monotonic else gap**2

    def _score(self, profile: Profile, ideal: Profile) -> float:
        # Always use our true profile, even when projecting a persona.
        return (
            self.bonus
            - decrement(self.ideal, profile, self.monotonic)
            - 0.5 * decrement(ideal, self.profile, self.monotonic)
        )

    def _refresh_model(self) -> None:
        self.trait_probs = self.traits.probabilities()
        self.preference_probs = self.preferences.probabilities()

        self.expected_mine = [
            sum(
                p * self._gap(i, self.ideal[i], v)
                for v, p in enumerate(row)
            )
            for i, row in enumerate(self.trait_probs)
        ]

        self.expected_theirs = [
            sum(
                p * self._gap(i, v, self.profile[i])
                for v, p in enumerate(row)
            )
            for i, row in enumerate(self.preference_probs)
        ]

        self.sample_scores = sorted(
            self._score(
                tuple(
                    self.rng.choices(VALUES, weights=row)[0]
                    for row in self.trait_probs
                ),
                tuple(
                    self.rng.choices(VALUES, weights=row)[0]
                    for row in self.preference_probs
                ),
            )
            for _ in range(self.config.sample_size)
        )

    def _ad_score(self, ad: Advertisement) -> float:
        mine = (
            sum(self.expected_mine)
            - self.expected_mine[ad.own_dim]
            + self._gap(
                ad.own_dim,
                self.ideal[ad.own_dim],
                ad.own_value,
            )
        )

        theirs = (
            sum(self.expected_theirs)
            - self.expected_theirs[ad.seek_dim]
            + self._gap(
                ad.seek_dim,
                ad.seek_value,
                self.profile[ad.seek_dim],
            )
        )

        return self.bonus - mine - 0.5 * theirs

    def _penalty(self, partner: int) -> float:
        return (
            self.dims
            * self.config.penalty_per_dimension
            * self.suspicion.get(partner, 0.0)
        )

    def _decision_score(
        self,
        partner: int,
        ad: Advertisement,
    ) -> float:
        return (
            self.estimates.get(partner, self._ad_score(ad))
            - self._penalty(partner)
        )

    def _threshold(self, rounds_left: int) -> float:
        horizon = max(1, min(self.rounds, self.people - 1))

        gamma = max(
            0.0,
            min(
                1.0,
                min(rounds_left - 1, self.singles - 1) / horizon,
            ),
        )

        pool = sorted(
            self.sample_scores
            + [
                value - self._penalty(partner)
                for partner, value in self.estimates.items()
            ]
        )

        if not pool:
            return 0.0

        index = min(
            len(pool) - 1,
            int(self.config.top_quantile * gamma * len(pool)),
        )

        return max(0.0, pool[index])

    def _update_suspicion(
        self,
        date: DateContext,
        observed: Profile,
        stated: Profile,
    ) -> None:
        c = self.config
        partner = date.partner

        previous = self.traits.reports.get(partner)
        previous_ideal = self.preferences.reports.get(partner)

        changed = float(
            previous is not None and previous != tuple(observed)
        )
        changed += float(
            previous_ideal is not None
            and previous_ideal != tuple(stated)
        )

        ad = date.partner_ad
        contradiction = float(
            observed[ad.own_dim] != ad.own_value
        )
        contradiction += float(
            stated[ad.seek_dim] != ad.seek_value
        )

        copied = float(
            similar(
                stated,
                self.personas.get(partner, self.profile),
                c.similarity_cutoff,
            )
        )

        if previous is not None and partner in self.my_described:
            copied += float(
                similar(
                    observed,
                    self.my_described[partner],
                    c.similarity_cutoff,
                )
            )

        self.evidence[partner] = {
            "changed": changed,
            "ad": contradiction,
            "copy": copied,
        }

        new_evidence = (
            changed * c.change_weight
            + contradiction * c.ad_weight
            + copied * c.copy_weight
        )

        # Repeated identical evidence sets a floor rather than
        # accumulating indefinitely. Consistency allows recovery.
        self.suspicion[partner] = min(
            c.suspicion_cap,
            max(
                self.suspicion.get(partner, 0.0) * c.suspicion_decay,
                new_evidence,
            ),
        )

    def choose_dates(
        self,
        candidates: tuple[Candidate, ...],
        turn: RoundContext,
    ) -> list[int]:
        self.singles = turn.singles
        self._update_phase(turn.round)

        ranked = sorted(
            candidates,
            key=lambda c: (
                -self._decision_score(c.index, c.ad),
                c.index,
            ),
        )

        count = max(
            1,
            math.ceil(
                self.config.interest_multiple * math.sqrt(self.people)
            ),
        )

        if self.list_limit is not None:
            count = min(count, self.list_limit)

        return [c.index for c in ranked[:count]]

    def _demand_cost(
        self,
        dim: int,
        value: int,
        ad: Advertisement | None = None,
    ) -> float:
        if ad is not None and dim == ad.seek_dim:
            return float(self._gap(dim, ad.seek_value, value))

        return sum(
            p * self._gap(dim, want, value)
            for want, p in enumerate(self.preference_probs[dim])
        )

    def project(self, date: DateContext) -> Profile:
        # Keep the same persona for every date with this partner.
        if date.partner in self.personas:
            return self.personas[date.partner]

        persona = list(self.profile)
        c = self.config

        if (
            c.deception_enabled
            and date.date_number == 1
            and date.rounds_left >= 2
            and date.round >= math.ceil(c.late_fraction * self.rounds)
            and min(self.preferences.effective_samples) >= c.min_samples
            and self._decision_score(
                date.partner, date.partner_ad
            ) > 0
        ):
            changes = []

            for dim, actual in enumerate(self.profile):
                # Never contradict our advertised own characteristic.
                if dim == self.ad.own_dim:
                    continue

                options = range(
                    max(0, actual - c.max_shift),
                    min(10, actual + c.max_shift) + 1,
                )

                best = min(
                    options,
                    key=lambda v: (
                        self._demand_cost(dim, v),
                        abs(v - actual),
                        v,
                    ),
                )

                population_gain = (
                    self._demand_cost(dim, actual)
                    - self._demand_cost(dim, best)
                )

                partner_gain = (
                    self._demand_cost(dim, actual, date.partner_ad)
                    - self._demand_cost(dim, best, date.partner_ad)
                )

                if population_gain >= c.demand_gap and partner_gain > 0:
                    changes.append((partner_gain, dim, best))

            changes.sort(key=lambda item: (-item[0], item[1]))
            selected = changes[: c.max_lie_dimensions]

            # Perceived-cost improvement, not marriage probability.
            if (
                selected
                and sum(gain for gain, _, _ in selected)
                >= c.min_perceived_gain
            ):
                for _, dim, value in selected:
                    persona[dim] = value

                self.deceptions_started += 1

        self.personas[date.partner] = tuple(persona)
        return self.personas[date.partner]

    def describe_ideal(
        self,
        date: DateContext,
        observed: Profile,
    ) -> Profile:
        return self.ideal

    def decide(
        self,
        date: DateContext,
        observed: Profile,
        stated_ideal: Profile,
    ) -> Decision:
        self._update_suspicion(date, observed, stated_ideal)

        self.traits.update(date.partner, observed)
        self.preferences.update(date.partner, stated_ideal)
        self.my_described[date.partner] = self.ideal
        self.estimates[date.partner] = self._score(
            observed, stated_ideal
        )

        # Refresh after every date, including reserved repeat dates.
        self._refresh_model()
        self._update_phase(date.round)

        score = self._decision_score(
            date.partner, date.partner_ad
        )

        if score <= 0 or self.observing:
            return Decision()

        if score >= self._threshold(date.rounds_left):
            return Decision(
                again=True,
                propose=date.date_number >= 2,
            )

        return Decision()

    def on_outcome(self, outcome: DateOutcome) -> None:
        if outcome.partner_proposed:
            self.proposed_to_me.add(outcome.partner)