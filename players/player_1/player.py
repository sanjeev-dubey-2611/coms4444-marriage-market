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

# Only marry when the estimated score clears this share of the bonus.
MARGIN_SHARE = 0.1
# Stop chasing someone who ignores this many of our proposals.
MAX_UNANSWERED = 5


class Player1(BasePlayer):
	"""Honest Verifier.

	Always tells the truth (the default `project` and `describe_ideal`).
	Drops anyone caught lying. Re-dates people who stay truthful and are a
	good enough match, proposes from the second date on, and marries them.

	Honest people show the same numbers on every date, so a lie is anything
	that contradicts what the person showed or said before:
	- their behaviour or stated ideal does not match their own ad,
	- the ideal they describe is exactly our profile (flattery),
	- their projection or stated ideal changes between dates.
	A liar who shows the same fake profile every time cannot be caught.
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.margin = max(0.0, MARGIN_SHARE * self.bonus)
		# partner -> (first projection, first stated ideal)
		self.first_seen: dict[int, tuple[Profile, Profile]] = {}
		# partner -> estimated score, only for people who passed every check
		self.estimates: dict[int, float] = {}
		self.unanswered: dict[int, int] = {}
		self.proposed_to: set[int] = set()

	def advertise(self) -> Advertisement:
		# Honest ad: our best monotonic dimension, and the dimension where our
		# ideal is most extreme (the one a random partner is likeliest to miss).
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		seek = max(range(self.dims), key=lambda i: abs(self.ideal[i] - 5))
		return Advertisement(
			own_dim=own, own_value=self.profile[own], seek_dim=seek, seek_value=self.ideal[seek]
		)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		# Verified people first (best estimate first), then strangers ranked by ad.
		verified = sorted(
			(c for c in candidates if c.index in self.estimates),
			key=lambda c: (-self.estimates[c.index], c.index),
		)
		fresh = sorted(
			(c for c in candidates if c.index not in self.first_seen),
			key=lambda c: (self._ad_cost(c.ad), c.index),
		)
		chosen = [c.index for c in verified + fresh]
		return chosen[: self.list_limit]

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		partner = date.partner
		self.proposed_to.discard(partner)

		if self._is_lying(partner, date.partner_ad, observed, stated_ideal):
			self.estimates.pop(partner, None)
			return Decision(incompatible=True)
		self.first_seen.setdefault(partner, (observed, stated_ideal))

		estimate = self._estimate(observed, stated_ideal)
		if estimate <= self.margin or self.unanswered.get(partner, 0) >= MAX_UNANSWERED:
			self.estimates.pop(partner, None)
			return Decision(incompatible=True)
		self.estimates[partner] = estimate

		propose = date.date_number >= 2
		if propose:
			self.proposed_to.add(partner)
		return Decision(again=True, propose=propose)

	def on_outcome(self, outcome: DateOutcome) -> None:
		if outcome.partner in self.proposed_to and not outcome.married:
			self.unanswered[outcome.partner] = self.unanswered.get(outcome.partner, 0) + 1

	def _is_lying(
		self, partner: int, ad: Advertisement, observed: Profile, stated_ideal: Profile
	) -> bool:
		if observed[ad.own_dim] != ad.own_value or stated_ideal[ad.seek_dim] != ad.seek_value:
			return True
		if tuple(stated_ideal) == tuple(self.profile):
			return True
		first = self.first_seen.get(partner)
		return first is not None and first != (observed, stated_ideal)

	def _estimate(self, observed: Profile, stated_ideal: Profile) -> float:
		"""Our score if they are who they say they are."""
		mine = decrement(self.ideal, observed, self.monotonic)
		theirs = decrement(stated_ideal, self.profile, self.monotonic)
		return self.bonus - mine - theirs / 2

	def _ad_cost(self, ad: Advertisement) -> float:
		"""What the two advertised dimensions alone would cost us."""
		mine = self._gap(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		theirs = self._gap(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		return mine + theirs / 2

	def _gap(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		if dim < self.monotonic and gap <= 0:
			return 0
		return gap * gap
