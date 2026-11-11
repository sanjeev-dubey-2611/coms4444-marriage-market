"""Group 9 marriage-market player."""

from functools import cache
from itertools import combinations

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
from models.profile import VALUES, Profile, decrement, happiness


class Player9(BasePlayer):
	"""Show honest profiles, shortlist by estimated happiness, and follow up briefly."""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.date_scores: dict[int, float] = {}
		self.proposed_to_us: set[int] = set()
		self.singles = self.people
		# Use the team's truthful ad convention only in supported games.
		eligible = (
			8 <= self.people <= 22
			and self.monotonic >= 2
			and self.rounds >= 2 * self.people + 6
			and self.bonus > 0
		)
		self.match_round = min(2 * self.people, self.rounds // 2)
		self.coordinating: bool | None = None if eligible else False
		self.reports: dict[int, tuple[Profile, Profile] | None] = {}
		self.target_chosen = False
		self.target: int | None = None
		# Treat each undisclosed value as uniform over 0–10 when scoring an ad.
		self.unknown_trait_cost = tuple(
			sum(self._cost(dim, want, value) for value in VALUES) / len(VALUES)
			for dim, want in enumerate(self.ideal)
		)
		self.unknown_preference_cost = tuple(
			sum(self._cost(dim, value, have) for value in VALUES) / len(VALUES)
			for dim, have in enumerate(self.profile)
		)

	def _cost(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		return (max(0, gap) if dim < self.monotonic else gap) ** 2

	def advertise(self) -> Advertisement:
		# Show the trait with the lowest expected cost to a partner, and our costliest need.
		own = min(range(self.dims), key=self.unknown_preference_cost.__getitem__)
		if self.coordinating is not False:
			own = 1
		seek = max(range(self.dims), key=self.unknown_trait_cost.__getitem__)
		return Advertisement(own, self.profile[own], seek, self.ideal[seek])

	def _threshold(self, rounds_left: int) -> float:
		time_left = (rounds_left - 1) / max(1, self.rounds - 1)
		# Repeat dates use the last count observed while making a shortlist.
		options_left = (self.singles - 2) / max(1, self.people - 2)
		return 0.4 * self.bonus * min(time_left, options_left)

	def _ad_score(self, ad: Advertisement) -> float:
		# Rank by how much the advertised values improve on a fully unknown partner.
		trait_gain = self.unknown_trait_cost[ad.own_dim] - self._cost(
			ad.own_dim, self.ideal[ad.own_dim], ad.own_value
		)
		preference_gain = self.unknown_preference_cost[ad.seek_dim] - self._cost(
			ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim]
		)
		return trait_gain + preference_gain / 2

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self.singles = turn.singles
		if self.coordinating is None:
			self.coordinating = len(candidates) == self.people - 1 and all(
				candidate.ad.own_dim == 1 for candidate in candidates
			)
		if self.coordinating:
			if turn.round < self.match_round:
				fresh = sorted(
					(c for c in candidates if c.dates == 0),
					key=lambda c: self._ad_score(c.ad),
					reverse=True,
				)
				width = 6 if self.list_limit is None else min(6, self.list_limit)
				return [c.index for c in fresh[:width]]
			if not self.target_chosen:
				self.target = self._matching_target(candidates)
				self.target_chosen = True
			if self.target is not None and any(c.index == self.target for c in candidates):
				return [self.target]

		if self.bonus <= 0:
			return []

		threshold = self._threshold(turn.rounds_left)
		known = []
		fresh = []
		for candidate in candidates:
			index = candidate.index
			score = self.date_scores.get(index)
			if score is None:
				if turn.rounds_left > 1:
					fresh.append((self._ad_score(candidate.ad), self.rng.random(), index))
				continue
			if score > (0 if index in self.proposed_to_us else threshold):
				known.append((score, index))

		# Ranking determines who makes the shortlist. The engine treats it as a set.
		known.sort(reverse=True)
		fresh.sort(reverse=True)
		width = 3 if self.list_limit is None else min(3, self.list_limit)
		return [entry[-1] for entry in (known + fresh)[:width]]

	def _matching_target(self, candidates: tuple[Candidate, ...]) -> int | None:
		"""Find our partner in a matching that maximizes pair count, then happiness."""
		profiles = {self.index: (self.profile, self.ideal)}
		for candidate in candidates:
			report = self.reports.get(candidate.index)
			if report is not None:
				profiles[candidate.index] = report
		indices = sorted(profiles)
		weights = {}
		for first, second in combinations(range(len(indices)), 2):
			profile_a, ideal_a = profiles[indices[first]]
			profile_b, ideal_b = profiles[indices[second]]
			cost_a = decrement(ideal_a, profile_b, self.monotonic)
			cost_b = decrement(ideal_b, profile_a, self.monotonic)
			happy_a = self.bonus - cost_a - cost_b / 2
			happy_b = self.bonus - cost_b - cost_a / 2
			if happy_a > 0 and happy_b > 0:
				weights[first, second] = happy_a + happy_b

		choices: dict[int, int | None] = {}

		@cache
		def solve(mask: int) -> tuple[int, float]:
			if not mask:
				return 0, 0.0
			bit = mask & -mask
			first = bit.bit_length() - 1
			rest = mask ^ bit
			best, partner = solve(rest), None
			remaining = rest
			while remaining:
				bit = remaining & -remaining
				remaining ^= bit
				second = bit.bit_length() - 1
				weight = weights.get((first, second))
				if weight is not None:
					count, total = solve(rest ^ bit)
					if (count + 1, total + weight) > best:
						best, partner = (count + 1, total + weight), second
			choices[mask] = partner
			return best

		mask = (1 << len(indices)) - 1
		solve(mask)
		solve.cache_clear()
		while mask:
			bit = mask & -mask
			first = bit.bit_length() - 1
			second = choices[mask]
			if indices[first] == self.index:
				return None if second is None else indices[second]
			mask ^= bit
			if second is not None:
				if indices[second] == self.index:
					return indices[first]
				mask ^= 1 << second
		return None

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		score = happiness(
			self.bonus, self.profile, self.ideal, observed, stated_ideal, self.monotonic
		)
		self.date_scores[date.partner] = score

		has_next_round = date.rounds_left > 1
		if self.coordinating and date.round < self.match_round:
			report = observed, stated_ideal
			if self.reports.setdefault(date.partner, report) != report:
				self.reports[date.partner] = None
			return Decision()
		if date.partner == self.target:
			partner_score = happiness(
				self.bonus, observed, stated_ideal, self.profile, self.ideal, self.monotonic
			)
			if score > 0 and partner_score > 0:
				return Decision(
					again=has_next_round and date.date_number <= 2,
					propose=date.date_number >= 2,
				)
			self.target = None
		if date.date_number == 1:
			return Decision(again=has_next_round and score > 0)

		threshold = (
			0.0 if date.partner in self.proposed_to_us else self._threshold(date.rounds_left)
		)
		propose = score > threshold
		# A proposal on date two can get one immediate follow-up.
		again = has_next_round and date.date_number == 2 and propose
		return Decision(again=again, propose=propose)

	def on_outcome(self, outcome: DateOutcome) -> None:
		if outcome.partner_proposed:
			self.proposed_to_us.add(outcome.partner)
		elif outcome.partner == self.target and outcome.date_number >= 2:
			self.target = None
