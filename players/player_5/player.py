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

	# One uniform pseudo-observation per value. Ads are selected claims, so
	# even a large market contributes at most two date-equivalents per field.
	PRIOR_STRENGTH = 11.0
	AD_WEIGHT = 0.25
	AD_MASS_LIMIT = 2.0

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.history: dict[int, Courtship] = {}
		self.first_scores: deque[float] = deque(maxlen=64)
		self.available: tuple[Candidate, ...] = ()
		self.current_singles = self.people
		self.traits = [[0] * 11 for _ in range(self.dims)]
		self.desires = [[0] * 11 for _ in range(self.dims)]
		self.ad_traits = [[0] * 11 for _ in range(self.dims)]
		self.ad_desires = [[0] * 11 for _ in range(self.dims)]
		self.observed_ads: dict[int, Advertisement] = {}
		self.observed_dates: set[int] = set()
		self._refresh_costs()

	def _pmf(self, dim: int, *, desire: bool = False) -> tuple[float, ...]:
		direct = self.desires[dim] if desire else self.traits[dim]
		ads = self.ad_desires[dim] if desire else self.ad_traits[dim]
		ad_weight = min(self.AD_WEIGHT, self.AD_MASS_LIMIT / max(1, sum(ads)))
		weights = [
			count + ad_weight * advertised + self.PRIOR_STRENGTH / 11
			for count, advertised in zip(direct, ads, strict=True)
		]
		total = sum(weights)
		return tuple(weight / total for weight in weights)

	def _refresh_costs(self) -> None:
		"""Cache expectations after new evidence, never once per candidate."""
		self.unknown_trait_cost = tuple(
			sum(p * self._loss(i, target, value) for value, p in enumerate(self._pmf(i)))
			for i, target in enumerate(self.ideal)
		)
		self.unknown_seek_cost = tuple(
			sum(
				p * self._loss(i, target, value)
				for target, p in enumerate(self._pmf(i, desire=True))
			)
			for i, value in enumerate(self.profile)
		)
		self.prior_ideal_loss = sum(self.unknown_seek_cost)

	def _observe_ads(self, candidates: tuple[Candidate, ...]) -> None:
		changed = False
		for candidate in candidates:
			person = candidate.index
			if person in self.observed_ads or person in self.observed_dates:
				continue
			ad = candidate.ad
			self.observed_ads[person] = ad
			self.ad_traits[ad.own_dim][ad.own_value] += 1
			self.ad_desires[ad.seek_dim][ad.seek_value] += 1
			changed = True
		if changed:
			self._refresh_costs()

	def _observe_date(self, person: int, observed: Profile, desired: Profile) -> None:
		# Keep one first-date sample per person, not one sample per repeat.
		# A full observation replaces that person's partial ad evidence.
		if person in self.observed_dates:
			return
		self.observed_dates.add(person)
		ad = self.observed_ads.pop(person, None)
		if ad is not None:
			self.ad_traits[ad.own_dim][ad.own_value] -= 1
			self.ad_desires[ad.seek_dim][ad.seek_value] -= 1
		for dim, value in enumerate(observed):
			self.traits[dim][value] += 1
		# Exact copying already triggers our lie safeguard; do not train its
		# fallback estimate on the same suspicious claim it is discounting.
		if not mirrors_projection(desired, self.profile):
			for dim, value in enumerate(desired):
				self.desires[dim][value] += 1
		self._refresh_costs()

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
		# Candidates omit reserved repeat pairs and incompatible people. Only
		# the public singles count measures people actually leaving the market.
		self.current_singles = turn.singles
		self._observe_ads(candidates)
		known = []
		fresh = []
		for candidate in candidates:
			match = self.history.get(candidate.index)
			if match is None:
				if turn.rounds_left >= 2:
					fresh.append(candidate)
			# On the last round, there is no later chance to retry a known match.
			elif match.score > 0 and (turn.rounds_left == 1 or match.retry_round <= turn.round):
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
		if not fresh:
			width = len(choices)
		return [candidate.index for candidate in choices[:width][: self.list_limit]]

	def _search_value(self, rounds_left: int, fresh: int) -> float:
		attempts = min(fresh, rounds_left // 2)
		if attempts == 0:
			return 0.0
		# This still distinguishes a full and depleted market after the five-
		# attempt cap is reached. Square root makes the discount gradual.
		remaining = max(0.0, min(1.0, (self.current_singles - 1) / max(1, self.people - 1)))
		market_factor = sqrt(remaining)
		if not self.first_scores:
			return max(0.0, self.bonus * 0.25) * market_factor
		# A recent upper-quartile score estimates the upside of a new date.
		scores = sorted(self.first_scores)
		upper = scores[3 * (len(scores) - 1) // 4]
		return max(0.0, upper) * (1 - 0.65 ** min(attempts, 5)) * market_factor

	def _match_value(self, match: Courtship, round_number: int) -> float:
		if match.offer_round and round_number - match.offer_round <= 3:
			return 0.9 * match.score
		fit = 0.6 if match.partner_score > 0 else 0.3
		return fit * match.score / (1 + match.declined_proposals)

	def _search_reservation(self, date: DateContext) -> float:
		if date.rounds_left <= 2:
			return 0.0
		fresh = sum(c.index != date.partner and c.index not in self.history for c in self.available)
		return self._search_value(date.rounds_left - 1, fresh)

	def _reservation(self, date: DateContext) -> float:
		if date.rounds_left <= 1:
			return 0.0
		# The last visible pool can be stale during a run of repeat dates.
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
		return max(known, self._search_reservation(date))

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		self._observe_date(date.partner, observed, stated_ideal)
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
		# A proposal alone does not reserve the next round.
		match.proposed = (
			date.date_number >= 2
			and match.score > 0
			and match.score >= 0.25 * bar
			and (match.score >= bar or match.score >= self._search_reservation(date))
		)
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
