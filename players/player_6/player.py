"""Secretary-style marriage market strategy with learned field distributions.

Drop-in replacement for players/player_6/player.py.
"""

from __future__ import annotations

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
from models.profile import Profile, decrement


@dataclass
class _Partner:
	observed: Profile
	desired: Profile
	score: float
	trust: float = 1.0
	dates: int = 1
	our_proposals: int = 0
	their_proposals: int = 0
	mutual_repeats: int = 0


class Player6(BasePlayer):
	"""Learn early, then pursue the first candidate exceeding the sample best.

	A reversible denied set is deliberately *not* the engine's permanent
	incompatible flag. The game does not reveal who requested us, so repeated
	dates / proposals are the only observable interest signals. All state is
	private to this instance.
	"""

	EXPLORE_FRACTION = 0.30
	EARLY_EXCEPTION_RATIO = 0.65  # score >= 65% of bonus
	PRIOR_STRENGTH = 3.0
	EARLY_CONFIDENCE = 0.72
	END_MARGIN = 0.01
	ONE_SIDED_LIMIT = 3

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		# Need two rounds for the two dates required to propose.
		self.explore_rounds = min(
			math.ceil(self.rounds * self.EXPLORE_FRACTION),
			max(0, self.rounds - 2),
		)
		self.explore_people = max(1, math.ceil(max(1, self.people - 1) * self.EXPLORE_FRACTION))
		self.partners: dict[int, _Partner] = {}
		self.denied: set[int] = set()
		self.proposed_to_me: set[int] = set()
		self.sample_scores: list[float] = []
		self.sample_ids: set[int] = set()
		self.best_sample = float('-inf')
		self.current_round = 1
		self.current_singles = self.people
		self.traits = [[0] * 11 for _ in range(self.dims)]
		self.desires = [[0] * 11 for _ in range(self.dims)]
		self.ad_traits = [[0] * 11 for _ in range(self.dims)]
		self.ad_desires = [[0] * 11 for _ in range(self.dims)]
		self.ad_seen: set[int] = set()
		self._ad: Advertisement | None = None

	def _gap(self, dim: int, want: int, have: int) -> int:
		difference = want - have
		if dim < self.monotonic:
			difference = max(0, difference)
		return difference * difference

	def _penalty(self, observed: Profile, desired: Profile) -> float:
		return float(
			decrement(self.ideal, observed, self.monotonic)
			+ 0.5 * decrement(desired, self.profile, self.monotonic)
		)

	def _prior(self, dim: int) -> list[float]:
		# Weak neutral prior; not access to the actual hidden distribution.
		# A little smoothing prevents ad samples dominating early estimates.
		if dim < self.monotonic:
			weights = [math.exp(-(((v - 5) / 3.0) ** 2) / 2) for v in range(11)]
		else:
			weights = [1.0] * 11
		z = sum(weights)
		return [w / z for w in weights]

	def _pmf(self, dim: int, *, desire: bool = False) -> list[float]:
		direct = self.desires[dim] if desire else self.traits[dim]
		ads = self.ad_desires[dim] if desire else self.ad_traits[dim]
		prior = self._prior(dim)
		# Full-date observations get greater weight than advertised claims.
		weights = [direct[v] + 0.25 * ads[v] + self.PRIOR_STRENGTH * prior[v] for v in range(11)]
		total = sum(weights)
		return [w / total for w in weights]

	def _expected_gap(self, dim: int, want: int, *, desire: bool) -> float:
		return sum(
			probability
			* (self._gap(dim, value, self.profile[dim]) if desire else self._gap(dim, want, value))
			for value, probability in enumerate(self._pmf(dim, desire=desire))
		)

	def _observe_ads(self, candidates: tuple[Candidate, ...]) -> None:
		for candidate in candidates:
			if candidate.index in self.ad_seen:
				continue
			self.ad_seen.add(candidate.index)
			ad = candidate.ad
			self.ad_traits[ad.own_dim][ad.own_value] += 1
			self.ad_desires[ad.seek_dim][ad.seek_value] += 1

	def _remember(self, person: int, observed: Profile, desired: Profile) -> None:
		existing = self.partners.get(person)
		if existing is not None:
			for dim in range(self.dims):
				self.traits[dim][existing.observed[dim]] -= 1
				self.desires[dim][existing.desired[dim]] -= 1

		# The game lets opponents misreport, so penalize changed claims.
		trust = existing.trust if existing else 1.0
		if existing is not None:
			mismatch = sum(a != b for a, b in zip(existing.observed, observed, strict=True))
			if mismatch:
				trust *= max(0.40, 1.0 - mismatch / max(1, self.dims))

		score = self.bonus - self._penalty(observed, desired)
		self.partners[person] = _Partner(
			observed=observed,
			desired=desired,
			score=score,
			trust=trust,
			dates=existing.dates + 1 if existing else 1,
			our_proposals=existing.our_proposals if existing else 0,
			their_proposals=existing.their_proposals if existing else 0,
			mutual_repeats=existing.mutual_repeats if existing else 0,
		)

		for dim in range(self.dims):
			self.traits[dim][observed[dim]] += 1
			self.desires[dim][desired[dim]] += 1

		if person not in self.sample_ids and self._exploring():
			self.sample_ids.add(person)
			self.sample_scores.append(score)
			self.best_sample = max(self.best_sample, score)

	def _exploring(self, round_number: int | None = None) -> bool:
		round_number = self.current_round if round_number is None else round_number
		# Whichever quota is exhausted first: time OR distinct dated people.
		return round_number <= self.explore_rounds and len(self.sample_ids) < self.explore_people

	def _progress(self, round_number: int, singles: int) -> float:
		time_used = (round_number - 1) / max(1, self.rounds - 1)
		scarcity = 1.0 - singles / max(1, self.people)
		return max(0.0, min(1.0, max(time_used, scarcity)))

	def _threshold(self, round_number: int, singles: int) -> float:
		progress = self._progress(round_number, singles)
		# Reservation softens monotonically toward zero, never negative.
		# The sample champion is the secretary-problem reference value.
		reference = max(
			self.best_sample if math.isfinite(self.best_sample) else 0.0,
			0.35 * self.bonus,
		)

		if self._exploring(round_number):
			return max(self.EARLY_EXCEPTION_RATIO * self.bonus, reference)

		urgency = max(
			0.0,
			(progress - self.EXPLORE_FRACTION) / (1 - self.EXPLORE_FRACTION),
		)
		# Initially just exceed the best exploration match, later allow
		# positive matches as potential partners disappear.
		return max(
			self.END_MARGIN,
			reference * (1.0 - urgency) ** 1.4,
		)

	def _refresh_denied(self, round_number: int, singles: int) -> None:
		threshold = self._threshold(round_number, singles)
		self.denied = {
			person
			for person, state in self.partners.items()
			if state.score <= 0 or state.score < threshold
		}

	def _estimate_unknown(
		self,
		candidate: Candidate,
		expected_ours: list[float] | None = None,
		expected_theirs: list[float] | None = None,
	) -> float:
		if expected_ours is None:
			expected_ours = [
				self._expected_gap(dim, self.ideal[dim], desire=False) for dim in range(self.dims)
			]
		if expected_theirs is None:
			expected_theirs = [self._expected_gap(dim, 0, desire=True) for dim in range(self.dims)]

		total = sum(expected_ours) + 0.5 * sum(expected_theirs)
		ad = candidate.ad
		total += (
			self._gap(ad.own_dim, self.ideal[ad.own_dim], ad.own_value) - expected_ours[ad.own_dim]
		)
		total += 0.5 * (
			self._gap(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
			- expected_theirs[ad.seek_dim]
		)
		return self.bonus - total

	def advertise(self) -> Advertisement:
		# The one permitted +2 exaggeration is applied only here, at startup.
		# All date projections and stated ideals remain truthful.
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda dim: self.profile[dim])
		value = self.profile[own]

		if own < self.monotonic:
			value = min(10, value + 2)

		# Pick the dimension with the largest expected cost if unmet.
		seek = max(
			range(self.dims),
			key=lambda dim: self._expected_gap(dim, self.ideal[dim], desire=False),
		)
		self._ad = Advertisement(own, value, seek, self.ideal[seek])
		return self._ad

	def project(self, date: DateContext) -> Profile:
		return self.profile

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		return self.ideal

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self.current_round = turn.round
		self.current_singles = turn.singles
		self._observe_ads(candidates)
		self._refresh_denied(turn.round, turn.singles)

		exploring = self._exploring(turn.round)
		threshold = self._threshold(turn.round, turn.singles)

		# Compute empirical expectations once per round, not for every candidate.
		expected_ours = [
			self._expected_gap(dim, self.ideal[dim], desire=False) for dim in range(self.dims)
		]
		expected_theirs = [self._expected_gap(dim, 0, desire=True) for dim in range(self.dims)]

		ranked: list[tuple[float, int]] = []

		for candidate in candidates:
			idx = candidate.index
			known = self.partners.get(idx)

			if known is not None:
				if idx in self.denied:
					continue

				value = known.score

				if known.mutual_repeats:
					value += min(0.12 * self.bonus, known.mutual_repeats * 2)
				if idx in self.proposed_to_me:
					value += min(0.08 * self.bonus, 4.0)

			else:
				value = self._estimate_unknown(candidate, expected_ours, expected_theirs)

				# During discovery, seek distinct people rather than re-dates.
				if exploring:
					value += 0.10 * max(1.0, self.bonus)

				# Unknown ads are uncertain; do not treat forecasts as fact.
				value -= (1 - self.EARLY_CONFIDENCE) * max(0.0, value)

			# Mutual dating priority is valuable, but avoid low-scoring repeats.
			if known is not None and (
				known.score < threshold
				or (
					not exploring
					and math.isfinite(self.best_sample)
					and abs(threshold - self.best_sample) <= 1e-9
					and known.score <= self.best_sample
				)
			):
				continue

			ranked.append((value, idx))

		ranked.sort(key=lambda item: (-item[0], item[1]))
		limit = len(ranked) if self.list_limit is None else self.list_limit
		return [person for _, person in ranked[:limit]]

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		self.current_round = date.round
		was_exploring = self._exploring(date.round)

		self._remember(date.partner, observed, stated_ideal)
		self._refresh_denied(date.round, self.current_singles)

		state = self.partners[date.partner]
		threshold = self._threshold(date.round, self.current_singles)
		score = state.score

		# Hard safety bound: never propose with nonpositive predicted payoff.
		if score <= 0:
			self.denied.add(date.partner)
			return Decision()

		early_exception = score >= self.EARLY_EXCEPTION_RATIO * self.bonus
		explore = was_exploring

		if explore and not early_exception:
			return Decision()

		# After sampling, accept the first person better than the observed
		# champion, subject to the falling reservation threshold.
		# A secretary benchmark requires strictly better than the sample
		# champion initially; after the threshold relaxes, equality is fine.
		champion = self.best_sample
		strict_champion = (
			not explore and math.isfinite(champion) and abs(threshold - champion) <= 1e-9
		)
		eligible = score > threshold if strict_champion else score >= threshold

		if not eligible:
			self.denied.add(date.partner)
			return Decision()

		# Avoid endlessly proposing to people who never reciprocate.
		if state.our_proposals >= self.ONE_SIDED_LIMIT and state.their_proposals == 0:
			return Decision(again=False)

		propose = date.date_number >= 2
		if propose:
			state.our_proposals += 1

		return Decision(again=True, propose=propose)

	def on_outcome(self, outcome: DateOutcome) -> None:
		state = self.partners.get(outcome.partner)
		if state is None:
			return

		if outcome.partner_proposed:
			self.proposed_to_me.add(outcome.partner)
			state.their_proposals += 1

		if outcome.again:
			state.mutual_repeats += 1
