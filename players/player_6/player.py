"""Explore honestly, then adapt presentation and pursue positive-score matches."""

import math

from models.player import (
	Advertisement,
	Candidate,
	DateContext,
	Decision,
	GameContext,
	PlayerSnapshot,
	RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import Profile, decrement


class Player6(BasePlayer):
	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		# Instance settings: penalty / B <= 0.8 leaves a 20% estimated margin.
		self.exploration_fraction = 0.25
		self.penalty_ratio = 0.8
		self.max_projection_shift = 2
		# Leave at least two rounds for a first date followed by a proposal.
		self.exploration_rounds = min(
			math.ceil(self.rounds * self.exploration_fraction), max(0, self.rounds - 2)
		)
		self.observations: dict[int, tuple[Profile, Profile]] = {}
		self.trait_counts = [[0] * 11 for _ in range(self.dims)]
		self.preference_counts = [[0] * 11 for _ in range(self.dims)]

	def advertise(self) -> Advertisement:
		# Ads cannot change later, so keep this initial signal honest.
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		seek = max(range(self.dims), key=lambda i: abs(self.ideal[i] - 5))
		return Advertisement(own, self.profile[own], seek, self.ideal[seek])

	def _remember(self, partner: int, observed: Profile, stated_ideal: Profile) -> None:
		previous = self.observations.get(partner)
		if previous is not None:
			for counts, values in zip(
				(self.trait_counts, self.preference_counts), previous, strict=True
			):
				for dim, value in enumerate(values):
					counts[dim][value] -= 1
		self.observations[partner] = (observed, stated_ideal)
		for counts, values in zip(
			(self.trait_counts, self.preference_counts), (observed, stated_ideal), strict=True
		):
			for dim, value in enumerate(values):
				counts[dim][value] += 1

	def _penalty(self, observed: Profile, stated_ideal: Profile) -> float:
		return decrement(self.ideal, observed, self.monotonic) + 0.5 * decrement(
			stated_ideal, self.profile, self.monotonic
		)

	def _gap(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		return max(0, gap) ** 2 if dim < self.monotonic else gap**2

	def _candidate_penalty(self, candidate: Candidate) -> float:
		if candidate.index in self.observations:
			return self._penalty(*self.observations[candidate.index])
		# Unknown dimensions use the empirical distribution, with a uniform prior
		# of total weight one. Ads replace only the dimensions they disclose.
		total = 0.0
		for dim in range(self.dims):
			traits = self.trait_counts[dim]
			preferences = self.preference_counts[dim]
			mine = (
				self._gap(dim, self.ideal[dim], candidate.ad.own_value)
				if dim == candidate.ad.own_dim
				else sum(
					(count + 1 / 11) * self._gap(dim, self.ideal[dim], value)
					for value, count in enumerate(traits)
				)
				/ (sum(traits) + 1)
			)
			theirs = (
				self._gap(dim, candidate.ad.seek_value, self.profile[dim])
				if dim == candidate.ad.seek_dim
				else sum(
					(count + 1 / 11) * self._gap(dim, value, self.profile[dim])
					for value, count in enumerate(preferences)
				)
				/ (sum(preferences) + 1)
			)
			total += mine + 0.5 * theirs
		return total

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		if turn.round <= self.exploration_rounds:
			# Prioritize unseen people without selecting for compatibility yet.
			choices = list(candidates)
			self.rng.shuffle(choices)
			choices.sort(key=lambda c: c.dates)
		else:
			choices = sorted(candidates, key=lambda c: (self._candidate_penalty(c), c.index))
			choices = [
				c
				for c in choices
				if c.index not in self.observations
				or self._acceptable(self._penalty(*self.observations[c.index]))
			]
		return [c.index for c in choices[: self.list_limit]]

	def project(self, date: DateContext) -> Profile:
		if date.round <= self.exploration_rounds:
			return self.profile
		projected = []
		for dim, value in enumerate(self.profile):
			if dim < self.monotonic:
				value = min(10, value + self.max_projection_shift)
			else:
				counts = self.preference_counts[dim]
				if sum(counts):
					target = round(sum(v * n for v, n in enumerate(counts)) / sum(counts))
					value += max(
						-self.max_projection_shift, min(self.max_projection_shift, target - value)
					)
			projected.append(value)
		return tuple(projected)

	def _acceptable(self, penalty: float) -> bool:
		return penalty <= self.penalty_ratio * self.bonus and penalty < self.bonus

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		self._remember(date.partner, observed, stated_ideal)
		if date.round <= self.exploration_rounds:
			return Decision()
		if not self._acceptable(self._penalty(observed, stated_ideal)):
			# Signals may change, so avoid permanently blocking on an estimate.
			return Decision()
		return Decision(again=True, propose=date.date_number >= 2)
