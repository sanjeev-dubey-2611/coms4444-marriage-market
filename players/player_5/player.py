"""Choose between a repeat date and searching the remaining market."""

from collections import deque
from dataclasses import dataclass
from math import sqrt
from statistics import fmean

from models.player import (
	Advertisement,
	Candidate,
	DateContext,
	DateOutcome,
	Decision,
	GameContext,
	Player,
	PlayerSnapshot,
	RoundContext,
)
from models.profile import Profile, decrement


def unknown_ideal_cost(profile: Profile, monotonic: int) -> float:
	"""Expected partner penalty when each unknown target is uniform on 0–10."""
	return sum(
		fmean((max(0, want - have) if dim < monotonic else want - have) ** 2 for want in range(11))
		for dim, have in enumerate(profile)
	)


def mirrors_projection(stated_ideal: Profile, projection: Profile) -> bool:
	"""Use exact copying as a signal when there are at least three traits."""
	return len(projection) >= 3 and stated_ideal == projection


@dataclass
class Courtship:
	first_profile: Profile
	own_loss: int
	score: float = 0
	partner_score: float = 0
	offer_round: int = 0
	declined_repeats: int = 0
	declined_proposals: int = 0
	retry_round: int = 0
	asked: bool = False
	proposed: bool = False


class Player5(Player):
	"""Retain viable matches and use observed alternatives to decide when to commit."""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.history: dict[int, Courtship] = {}
		self.first_scores: deque[float] = deque(maxlen=64)
		self.available: tuple[Candidate, ...] = ()
		self.prior_ideal_loss = unknown_ideal_cost(self.profile, self.monotonic)
		self.unknown_trait_cost = tuple(
			sum(self._loss(i, target, value) for value in range(11)) / 11
			for i, target in enumerate(self.ideal)
		)
		self.unknown_seek_cost = tuple(
			sum(self._loss(i, target, value) for target in range(11)) / 11
			for i, value in enumerate(self.profile)
		)

	def _loss(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		return (max(0, gap) if dim < self.monotonic else gap) ** 2

	def advertise(self) -> Advertisement:
		own = min(range(self.dims), key=self.unknown_seek_cost.__getitem__)
		seek = max(range(self.dims), key=self.unknown_trait_cost.__getitem__)
		return Advertisement(own, self.profile[own], seek, self.ideal[seek])

	def _ad_value(self, ad: Advertisement) -> float:
		own = self.unknown_trait_cost[ad.own_dim] - self._loss(
			ad.own_dim, self.ideal[ad.own_dim], ad.own_value
		)
		theirs = self.unknown_seek_cost[ad.seek_dim] - self._loss(
			ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim]
		)
		return own + 0.35 * theirs

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self.available = candidates
		known = []
		fresh = []
		for candidate in candidates:
			match = self.history.get(candidate.index)
			if match is None:
				if turn.rounds_left >= 2:
					fresh.append(candidate)
			elif match.score > 0 and match.retry_round <= turn.round:
				known.append(candidate)
		known.sort(
			key=lambda candidate: self._match_value(self.history[candidate.index], turn.round),
			reverse=True,
		)
		fresh.sort(key=lambda candidate: self._ad_value(candidate.ad), reverse=True)
		width = max(1, int(sqrt(len(candidates))))
		bar = self._search_value(turn.rounds_left, len(fresh))
		ready = [c for c in known if self._match_value(self.history[c.index], turn.round) >= bar]
		choices = (ready or known) if ready or not fresh else fresh
		return [candidate.index for candidate in choices[:width][: self.list_limit]]

	def _search_value(self, rounds_left: int, fresh: int) -> float:
		attempts = min(fresh, rounds_left // 2)
		if attempts == 0:
			return 0.0
		if not self.first_scores:
			return max(0.0, self.bonus * 0.25)
		# A recent upper-quartile score estimates the upside of a new date.
		scores = sorted(self.first_scores)
		upper = scores[3 * (len(scores) - 1) // 4]
		return max(0.0, upper) * (1 - 0.65 ** min(attempts, 5))

	def _match_value(self, match: Courtship, round_number: int) -> float:
		if match.offer_round and round_number - match.offer_round <= 3:
			return 0.9 * match.score
		fit = 0.6 if match.partner_score > 0 else 0.3
		return fit * match.score / (1 + match.declined_proposals)

	def _reservation(self, date: DateContext) -> float:
		# The last visible pool can be stale during a run of repeat dates.
		fresh = sum(c.index != date.partner and c.index not in self.history for c in self.available)
		known = max(
			(
				self._match_value(match, date.round)
				for c in self.available
				if c.index != date.partner
				and (match := self.history.get(c.index)) is not None
				and match.retry_round <= date.round
			),
			default=0.0,
		)
		if date.rounds_left <= 1:
			return 0.0
		return max(known, self._search_value(date.rounds_left - 1, fresh))

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		own_loss = decrement(self.ideal, observed, self.monotonic)
		partner_loss = decrement(stated_ideal, self.profile, self.monotonic)
		match = self.history.get(date.partner)
		if match is None:
			match = self.history[date.partner] = Courtship(observed, own_loss)
		else:
			match.own_loss = max(match.own_loss, own_loss)
		if mirrors_projection(stated_ideal, self.profile):
			partner_loss = 0.25 * partner_loss + 0.75 * self.prior_ideal_loss
		match.score = self.bonus - match.own_loss - partner_loss / 2
		match.partner_score = self.bonus - partner_loss - match.own_loss / 2
		if date.date_number == 1:
			self.first_scores.append(match.score)
		bar = self._reservation(date)
		match.proposed = date.date_number >= 2 and match.score > 0 and match.score >= bar
		match.asked = (
			date.rounds_left > 1
			and match.score > 0
			and match.score >= bar
			and (date.rounds_left <= 2 or self._match_value(match, date.round) >= bar)
		)
		if (
			match.score <= 0
			and observed == match.first_profile
			and date.rounds_left > 2
			and len(self.available) > 2
			and own_loss >= self.bonus
		):
			return Decision(incompatible=True)
		return Decision(again=match.asked, propose=match.proposed)

	def on_outcome(self, outcome: DateOutcome) -> None:
		match = self.history[outcome.partner]
		if outcome.partner_proposed:
			match.offer_round = outcome.round
			match.declined_proposals = 0
		elif match.proposed:
			match.declined_proposals += 1
		if outcome.again:
			match.declined_repeats = 0
		elif match.asked and not outcome.married:
			match.declined_repeats += 1
			match.retry_round = outcome.round + min(4, match.declined_repeats + 1)
