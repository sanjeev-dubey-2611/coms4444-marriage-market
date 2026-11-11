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

# No fixed positive floor; the declining acceptance bar still applies.
FLOOR_SHARE = 0.0
# The bar starts at this percentile of the first-date scores we have seen ...
BAR_PERCENTILE = 0.6
# ... or at this share of the bonus until we have MIN_SAMPLES of them.
BAR_START_SHARE = 0.5
MIN_SAMPLES = 5
# In the last share of rounds, accept any partner whose score is above 0.
ENDGAME_FRACTION = 0.2
# Untrusted partners become acceptable once this share of rounds is gone ...
RELAX_FRACTION = 0.5
# ... or when we know fewer than this many trusted partners above the bar.
MIN_TRUSTED_OPTIONS = 2
# A lie that costs us at most this share of the bonus is forgiven (endgame: the larger one).
LIE_TOLERANCE_SHARE = 0.1
LIE_TOLERANCE_ENDGAME_SHARE = 0.2
# Propose only if this partner is within this share of the bonus of our best other option.
BEST_SLACK_SHARE = 0.1
# A proposal from them counts as a signal for this many rounds.
WILLING_ROUNDS = 2
# Stop counting someone as an option after this many ignored proposals.
MAX_IGNORED = 3
# Lower the bar as the market empties, not only as rounds pass.
MARKET_CLOCK = True


class Player1(BasePlayer):
	"""Trust but verify, relax over time.

	We are always honest ourselves. Every partner is scored with the handout's
	formula on what they showed us, after "repricing" anything that looks like a
	lie at its most pessimistic plausible value:
	- a trait that differs between their ad and their behaviour, or between dates,
	- a stated ideal that is an exact copy of our profile (flattery), together with
	the trait our ad asked for.
	The size of that correction is the lie's cost. Small lies are forgiven; partners
	with bigger lies are only considered later in the game.

	We propose only to partners above a personal bar that falls as rounds run out,
	and only if nobody we have dated and could still marry looks clearly better.
	In the endgame, anyone whose repriced score is above 0 is acceptable.
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.floor = max(0.0, FLOOR_SHARE * self.bonus)
		self.my_ad = self._make_ad()
		# partner -> distinct (projection, stated ideal) reports, first one first
		self.reports: dict[int, list[tuple[Profile, Profile]]] = {}
		self.ads: dict[int, Advertisement] = {}
		# partner -> (repriced score, lie cost)
		self.scores: dict[int, tuple[float, float]] = {}
		self.blocked: set[int] = set()
		self.samples: list[float] = []
		# What typical people look like, from first dates without red flags.
		self.dim_values: list[list[int]] = [[] for _ in range(self.dims)]
		self.their_costs: list[int] = []
		self.proposed_to_me: dict[int, int] = {}
		self.ignored: dict[int, int] = {}
		self.proposed_to: set[int] = set()
		self.available: set[int] = set()
		self.singles_left = 1.0  # share of the market still single, as of our last choice

	# ------------------------------------------------------------------ moves

	def advertise(self) -> Advertisement:
		return self.my_ad

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self.available = {c.index for c in candidates}
		self.singles_left = turn.singles / max(1, self.people)
		bar = self._bar(turn.rounds_left)
		endgame = self._endgame(turn.rounds_left)
		relaxed = self._relaxed(turn.rounds_left, bar)

		floor = self._floor(turn.rounds_left)
		# People who ignore our proposals sort last, but stay listed: dropping them
		# measured worse, since slow committers (e.g. Player 5) often come round.
		known = sorted(
			(c.index for c in candidates if c.index in self.scores and c.index not in self.blocked),
			key=lambda i: (self._unresponsive(i), -self.scores[i][0], i),
		)
		good = [
			i for i in known if self.scores[i][0] >= bar and self._eligible(i, relaxed, endgame)
		]
		fresh = sorted(
			(c for c in candidates if c.index not in self.reports),
			key=lambda c: (self._ad_cost(c.ad), c.index),
		)
		chosen = good + [c.index for c in fresh]
		# Weaker known partners are a fallback once strangers run out or time does.
		if endgame or not fresh:
			chosen += [
				i
				for i in known
				if i not in good
				and self.scores[i][0] > floor
				and self._eligible(i, relaxed, endgame)
			]
		return chosen[: self.list_limit]

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		partner = date.partner
		self.ads[partner] = date.partner_ad
		first_date = partner not in self.reports
		reports = self.reports.setdefault(partner, [])
		if (observed, stated_ideal) not in reports:
			reports.append((observed, stated_ideal))

		score, lie_cost = self._reprice(partner)
		self.scores[partner] = (score, lie_cost)
		if first_date:
			self.samples.append(score)
			self._learn(partner, observed, stated_ideal)

		rounds_left = date.rounds_left
		bar = self._bar(rounds_left)
		floor = self._floor(rounds_left)
		endgame = self._endgame(rounds_left)
		relaxed = self._relaxed(rounds_left, bar)

		# Block only clear negatives, and only while there is time to find others.
		if score <= 0 and not endgame:
			self.blocked.add(partner)
			return Decision(incompatible=True)

		eligible = self._eligible(partner, relaxed, endgame)
		willing = date.round - self.proposed_to_me.get(partner, -100) <= WILLING_ROUNDS
		best_other = self._best_other(partner, relaxed, endgame)

		if endgame or willing:
			want = eligible and score > floor
		else:
			want = eligible and score >= bar and score >= best_other - BEST_SLACK_SHARE * self.bonus

		propose = want and date.date_number >= 2
		if propose:
			self.proposed_to.add(partner)
		# Another date is how we verify and propose; keep it for people we'd marry.
		return Decision(again=want, propose=propose)

	def on_outcome(self, outcome: DateOutcome) -> None:
		partner = outcome.partner
		if outcome.partner_proposed:
			self.proposed_to_me[partner] = outcome.round
		if partner in self.proposed_to and not outcome.married:
			self.ignored[partner] = self.ignored.get(partner, 0) + 1
		self.proposed_to.discard(partner)

	# --------------------------------------------------------------- scoring

	def _reprice(self, partner: int) -> tuple[float, float]:
		"""(score after pricing suspicious claims pessimistically, how much that cost)."""
		reports = self.reports[partner]
		ad = self.ads[partner]
		flattered = all(tuple(ideal) == tuple(self.profile) for _, ideal in reports)

		mine = 0.0
		for dim in range(self.dims):
			options = {obs[dim] for obs, _ in reports}
			if ad.own_dim == dim:
				options.add(ad.own_value)  # ad and behaviour disagree: take the worse
			cost = max(self._gap(dim, self.ideal[dim], value) for value in options)
			if flattered and dim == self.my_ad.seek_dim:
				# Average penalties, not traits, so population variance contributes.
				cost = max(cost, self._typical_cost(dim))
			mine += cost

		honest_ideals = [ideal for _, ideal in reports if tuple(ideal) != tuple(self.profile)]
		if honest_ideals:
			theirs = max(decrement(ideal, self.profile, self.monotonic) for ideal in honest_ideals)
		else:
			theirs = self._typical_their_cost()
		score = self.bonus - mine - theirs / 2

		shown = max(
			self.bonus
			- decrement(self.ideal, obs, self.monotonic)
			- decrement(ideal, self.profile, self.monotonic) / 2
			for obs, ideal in reports
		)
		return score, max(0.0, shown - score)

	def _learn(self, partner: int, observed: Profile, stated_ideal: Profile) -> None:
		"""Record what a typical person looks like, skipping reports with red flags."""
		ad = self.ads[partner]
		if tuple(stated_ideal) == tuple(self.profile):
			return
		if observed[ad.own_dim] != ad.own_value:
			return
		for dim, value in enumerate(observed):
			self.dim_values[dim].append(value)
		self.their_costs.append(decrement(stated_ideal, self.profile, self.monotonic))

	def _typical_cost(self, dim: int) -> float:
		"""Expected mismatch against observed traits, with a uniform fallback.

		The hidden P is unavailable. With no usable observations, every legal
		value receives equal weight; otherwise use the empirical distribution.
		"""
		values = self.dim_values[dim] or VALUES
		return sum(self._gap(dim, self.ideal[dim], value) for value in values) / len(values)

	def _typical_their_cost(self) -> float:
		if self.their_costs:
			return sum(self.their_costs) / len(self.their_costs)
		# Unknown desired traits go in the "want" argument. This direction
		# matters for monotonic traits, where only unmet desires incur a cost.
		cost = 0.0
		for dim, have in enumerate(self.profile):
			values = self.dim_values[dim] or VALUES
			cost += sum(self._gap(dim, want, have) for want in values) / len(values)
		return cost

	# ------------------------------------------------------------ thresholds

	def _endgame(self, rounds_left: int) -> bool:
		return rounds_left <= max(2, ENDGAME_FRACTION * self.rounds)

	def _floor(self, rounds_left: int) -> float:
		return 0.0 if self._endgame(rounds_left) else self.floor

	def _bar(self, rounds_left: int) -> float:
		"""A personal reservation score that falls as rounds run out or the market empties."""
		if len(self.samples) >= MIN_SAMPLES:
			ordered = sorted(self.samples)
			base = ordered[int(BAR_PERCENTILE * (len(ordered) - 1))]
		else:
			base = BAR_START_SHARE * self.bonus
		left = max(0.0, (rounds_left - 1) / max(1, self.rounds - 1))
		if MARKET_CLOCK:
			left = min(left, self.singles_left)
		return max(self._floor(rounds_left), base * left)

	def _tolerance(self, endgame: bool) -> float:
		share = LIE_TOLERANCE_ENDGAME_SHARE if endgame else LIE_TOLERANCE_SHARE
		return share * self.bonus

	def _trusted(self, partner: int, endgame: bool) -> bool:
		return self.scores[partner][1] <= self._tolerance(endgame)

	def _eligible(self, partner: int, relaxed: bool, endgame: bool) -> bool:
		return relaxed or self._trusted(partner, endgame)

	def _relaxed(self, rounds_left: int, bar: float) -> bool:
		if rounds_left <= (1 - RELAX_FRACTION) * self.rounds:
			return True
		trusted = sum(
			1
			for i, (score, _) in self.scores.items()
			if i in self.available
			and i not in self.blocked
			and score >= bar
			and self._trusted(i, False)
		)
		return trusted < MIN_TRUSTED_OPTIONS

	def _unresponsive(self, partner: int) -> bool:
		return self.ignored.get(partner, 0) >= MAX_IGNORED

	def _best_other(self, partner: int, relaxed: bool, endgame: bool) -> float:
		"""Best score among other people we've dated who still look available."""
		scores = [
			score
			for i, (score, _) in self.scores.items()
			if i != partner
			and i in self.available
			and i not in self.blocked
			and not self._unresponsive(i)
			and self._eligible(i, relaxed, endgame)
		]
		return max(scores, default=float('-inf'))

	# ------------------------------------------------------------- utilities

	def _make_ad(self) -> Advertisement:
		# Honest ad: our best monotonic dimension, and the dimension where our
		# ideal is most extreme (the one a random partner is likeliest to miss).
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		seek = max(range(self.dims), key=lambda i: abs(self.ideal[i] - 5))
		return Advertisement(
			own_dim=own, own_value=self.profile[own], seek_dim=seek, seek_value=self.ideal[seek]
		)

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
