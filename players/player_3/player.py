import math

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


class Player3(BasePlayer):
	MIN_DATES = 2
	EXPLORE_FRACTION = 0.65
	RESERVATION_PERCENTILE = 0.70
	MIN_SAMPLES = 5

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self._commit_round = max(1, int(self.rounds * self.EXPLORE_FRACTION))
		self._est: dict[int, float] = {}
		self._seen: list[float] = []
		self._proposed_to_me: set[int] = set()
		self._mutual_again: dict[int, int] = {}

	def advertise(self) -> Advertisement:
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		seek = max(range(self.dims), key=lambda i: abs(self.ideal[i] - 5))
		return Advertisement(
			own_dim=own,
			own_value=self.profile[own],
			seek_dim=seek,
			seek_value=self.ideal[seek],
		)

	def _gap(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		if dim < self.monotonic and gap <= 0:
			return 0
		return gap * gap

	def _estimate(self, observed: Profile, stated_ideal: Profile) -> float:
		mine = decrement(self.ideal, observed, self.monotonic)
		theirs = decrement(stated_ideal, self.profile, self.monotonic)
		return self.bonus - mine - theirs / 2

	def _ad_cost(self, ad: Advertisement) -> float:
		mine = self._gap(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		theirs = self._gap(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		return mine + theirs / 2

	@staticmethod
	def _percentile(values: list[float], q: float) -> float:
		s = sorted(values)
		if not s:
			return 0.0
		k = (len(s) - 1) * q
		lo = math.floor(k)
		hi = math.ceil(k)
		if lo == hi:
			return s[int(k)]
		return s[lo] * (hi - k) + s[hi] * (k - lo)

	def _reservation(self, rounds_left: int) -> float:
		frac = max(0.0, min(1.0, (rounds_left - 1) / max(1, self.rounds - 1)))
		if len(self._seen) >= self.MIN_SAMPLES:
			base = self._percentile(self._seen, self.RESERVATION_PERCENTILE)
		else:
			base = 0.5 * self.bonus
		return max(0.0, base) * frac

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		if not candidates:
			return []
		exploring = turn.round <= self._commit_round
		known = sorted(
			(c for c in candidates if self._est.get(c.index, 0.0) > 0),
			key=lambda c: self._est[c.index],
			reverse=True,
		)
		fresh = sorted(
			(c for c in candidates if c.index not in self._est),
			key=lambda c: (self._ad_cost(c.ad), c.index),
		)
		order = [c.index for c in known]
		order += [c.index for c in (fresh if exploring else fresh[:3])]

		seen: set[int] = set()
		chosen: list[int] = []
		for i in order:
			if i not in seen:
				seen.add(i)
				chosen.append(i)
		if self.list_limit is not None:
			chosen = chosen[: self.list_limit]
		return chosen

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		est = self._estimate(observed, stated_ideal)
		self._est[date.partner] = est
		self._seen.append(est)

		if est <= 0:
			return Decision(again=False, propose=False)

		bar = self._reservation(date.rounds_left)
		willing = (
			date.partner in self._proposed_to_me or self._mutual_again.get(date.partner, 0) >= 2
		)
		propose = date.date_number >= self.MIN_DATES and est > 0 and (est > bar or willing)

		exploring = date.round <= self._commit_round
		pursue = est > max(bar, 0.5 * self.bonus) if exploring else est > 0.0
		return Decision(again=pursue or propose, propose=propose)

	def on_outcome(self, outcome: DateOutcome) -> None:
		if outcome.partner_proposed:
			self._proposed_to_me.add(outcome.partner)
		if outcome.again:
			self._mutual_again[outcome.partner] = self._mutual_again.get(outcome.partner, 0) + 1
