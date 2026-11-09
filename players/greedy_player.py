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


class GreedyPlayer(BasePlayer):
	"""Takes everything at face value.

	Honest advertisement, honest behaviour, honest description. After each date
	it plugs what it observed and what it was told into the handout's
	happiness formula, asks again when that estimate is positive, and proposes
	once the estimate clears a bar that falls from half the bonus to zero as
	rounds run out.

	This is deliberately shallow: it is a sanity baseline for the simulator and
	a starting point for discussion, not a serious strategy. It never doubts a
	projected profile or a described ideal, never lies itself, and never learns
	anything about the population from the people it meets.
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.estimates: dict[int, float] = {}
		self.proposed_to_me: set[int] = set()

	def advertise(self) -> Advertisement:
		# Show off the monotonic dimension we score best on, and 
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		
		# ask for the dimension where our ideal is furthest from the middle of the scale.
		seek = max(range(self.dims), key=lambda i: abs(self.ideal[i] - 5))
		
		return Advertisement(
			own_dim=own, own_value=self.profile[own], seek_dim=seek, seek_value=self.ideal[seek]
		)

	def _gap(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		if dim < self.monotonic and gap <= 0:
			return 0
		return gap * gap

	def _ad_cost(self, ad: Advertisement) -> float:
		"""What the two advertised dimensions alone would cost us, by the formula."""
		mine = self._gap(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		theirs = self._gap(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		return mine + theirs / 2

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		chosen = [c.index for c in candidates if self.estimates.get(c.index, 0.0) > 0]
		fresh = sorted(
			(c for c in candidates if c.index not in self.estimates),
			key=lambda c: (self._ad_cost(c.ad), c.index),
		)
		chosen += [c.index for c in fresh[: max(1, len(fresh) // 2)]]
		return chosen[: self.list_limit]

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		mine = decrement(self.ideal, observed, self.monotonic)
		theirs = decrement(stated_ideal, self.profile, self.monotonic)
		estimate = self.bonus - mine - theirs / 2
		self.estimates[date.partner] = estimate

		if estimate <= 0:
			return Decision(incompatible=True)

		bar = 0.5 * self.bonus * (date.rounds_left - 1) / max(1, self.rounds - 1)
		if date.partner in self.proposed_to_me:
			bar = 0.0
		propose = date.date_number >= 2 and estimate > bar
		return Decision(again=True, propose=propose)

	def on_outcome(self, outcome: DateOutcome) -> None:
		if outcome.partner_proposed:
			self.proposed_to_me.add(outcome.partner)
