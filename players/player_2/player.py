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
from models.profile import MAX_VALUE, VALUES, Profile, decrement

# Prior for every dimension: a discretised normal, worth this many observations.
PRIOR_MEAN = 5.0
PRIOR_SD = 2.0
PRIOR_WEIGHT = 10.0

# Monotonic dimensions are where liars inflate values. An observed value above
# the prior mean counts for less, down to (1 - MONO_LIAR_DISCOUNT) at value 10.
MONO_LIAR_DISCOUNT = 0.5

# Synthetic people drawn from the learned distributions each round, used to
# build the distribution of scores the threshold is read from.
SAMPLE_SIZE = 200

# The threshold is the gamma * TOP_QUANTILE quantile of the score distribution.
TOP_QUANTILE = 0.95

# Hard cap on the observe phase, as a fraction of rounds (the classic 37%), so
# a short run never spends all its time observing.
OBSERVE_ROUND_CAP = 0.37

# After more than this many dates with the same person, stop waiting: propose
# if their estimate is positive and has not dropped, otherwise walk away.
REPEAT_PROPOSE_AFTER = 3

# Two profiles whose centred cosine similarity reaches this are treated as a match.
SIMILARITY_CUTOFF = 0.98


def _normal_prior() -> list[float]:
	weights = [math.exp(-((v - PRIOR_MEAN) ** 2) / (2 * PRIOR_SD**2)) for v in VALUES]
	total = sum(weights)
	return [PRIOR_WEIGHT * w / total for w in weights]


def _similar(a: Profile, b: Profile) -> bool:
	"""True if a and b are equal, or nearly parallel once centred on the middle value.

	Centring matters: raw 0-10 vectors are all positive, so their plain cosine
	similarity is high even for unrelated profiles.
	"""
	if tuple(a) == tuple(b):
		return True
	mid = MAX_VALUE / 2
	ca = [x - mid for x in a]
	cb = [x - mid for x in b]
	na = math.sqrt(sum(x * x for x in ca))
	nb = math.sqrt(sum(x * x for x in cb))
	if na == 0 or nb == 0:
		return False
	return sum(x * y for x, y in zip(ca, cb, strict=True)) / (na * nb) >= SIMILARITY_CUTOFF


class Player2(BasePlayer):
	"""Honest, liar-aware, distribution-aware, observe-then-choose player.

	Honest: advertises its most extreme non-monotonic value and wish, projects its
	true profile and describes its true ideal.

	Distribution-aware: keeps a per-dimension histogram of values, seeded with a
	normal prior and updated from ads and projections. High values in monotonic
	dimensions are discounted, since that is where liars inflate.

	Liar-aware: flags people whose projection or stated ideal changes between
	dates, whose stated ideal mirrors our projection, or whose projection mirrors
	an ideal we described earlier. Liars are never proposed to or asked for.

	Observe-then-choose: observes until it has dated floor(sqrt(N)) unique people,
	then proposes to the first person whose estimate meets a threshold that decays
	with gamma = min(rounds_left, singles - 1) / (N - 1).
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		# Per-person memory.
		self.estimates: dict[int, float] = {}
		self.estimate_history: dict[int, list[float]] = {}
		self.projections: dict[int, list[Profile]] = {}
		self.stated_ideals: dict[int, list[Profile]] = {}
		self.my_described: dict[int, list[Profile]] = {}
		self.liars: set[int] = set()
		self.proposed_to_me: set[int] = set()
		self.best_estimate = -math.inf

		# Per-dimension value histograms (pseudo-counts over 0-10).
		self.counts = [_normal_prior() for _ in range(self.dims)]
		self.ads_seen: set[int] = set()
		self.projections_seen: set[int] = set()

		# Observe phase.
		self.observe_target = max(1, math.isqrt(self.people))
		self.observe_round_cap = max(1, math.ceil(OBSERVE_ROUND_CAP * self.rounds))
		self.observing = True

		# Threshold inputs, refreshed each round.
		self.singles = self.people
		self.sample_scores: list[float] = []
		self.expected_mine: list[float] = [0.0] * self.dims
		self.expected_theirs: list[float] = [0.0] * self.dims
		self._refresh_model()

	# ------------------------------------------------------------------ honesty

	def advertise(self) -> Advertisement:
		"""Show our most extreme non-monotonic value and ask for our most extreme
		non-monotonic wish. Both are true, and honesty there helps both sides."""
		free = range(self.monotonic, self.dims)
		showable = free if len(free) else range(self.dims)
		own = max(showable, key=lambda i: (abs(self.profile[i] - PRIOR_MEAN), -i))
		seek = max(showable, key=lambda i: (abs(self.ideal[i] - PRIOR_MEAN), -i))
		return Advertisement(
			own_dim=own, own_value=self.profile[own], seek_dim=seek, seek_value=self.ideal[seek]
		)

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		self.my_described.setdefault(date.partner, []).append(self.ideal)
		return self.ideal

	# ------------------------------------------------------------ distributions

	def _observe_value(self, dim: int, value: int) -> None:
		weight = 1.0
		if dim < self.monotonic and value > PRIOR_MEAN:
			weight -= MONO_LIAR_DISCOUNT * (value - PRIOR_MEAN) / (MAX_VALUE - PRIOR_MEAN)
		self.counts[dim][value] += weight

	def _gap(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		if dim < self.monotonic and gap <= 0:
			return 0
		return gap * gap

	def _refresh_model(self) -> None:
		"""Recompute expected per-dimension costs and the sampled score distribution."""
		probs = []
		for dim in range(self.dims):
			total = sum(self.counts[dim])
			probs.append([c / total for c in self.counts[dim]])
		self.expected_mine = [
			sum(p * self._gap(d, self.ideal[d], v) for v, p in zip(VALUES, probs[d], strict=True))
			for d in range(self.dims)
		]
		self.expected_theirs = [
			sum(p * self._gap(d, v, self.profile[d]) for v, p in zip(VALUES, probs[d], strict=True))
			for d in range(self.dims)
		]
		scores = []
		for _ in range(SAMPLE_SIZE):
			profile = tuple(self.rng.choices(VALUES, weights=probs[d])[0] for d in range(self.dims))
			ideal = tuple(self.rng.choices(VALUES, weights=probs[d])[0] for d in range(self.dims))
			scores.append(self._score(profile, ideal))
		self.sample_scores = scores

	# ---------------------------------------------------------------- estimates

	def _score(self, their_profile: Profile, their_ideal: Profile) -> float:
		mine = decrement(self.ideal, their_profile, self.monotonic)
		theirs = decrement(their_ideal, self.profile, self.monotonic)
		return self.bonus - mine - theirs / 2

	def _ad_score(self, ad: Advertisement) -> float:
		"""Expected score of someone we have not dated: the ad's two dimensions are
		taken as stated, every other dimension is the learned expectation."""
		mine = sum(self.expected_mine)
		mine += self._gap(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		mine -= self.expected_mine[ad.own_dim]
		theirs = sum(self.expected_theirs)
		theirs += self._gap(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		theirs -= self.expected_theirs[ad.seek_dim]
		return self.bonus - mine - theirs / 2

	def _gamma(self, rounds_left: int) -> float:
		chances = min(rounds_left, self.singles - 1)
		return max(0.0, min(1.0, chances / max(1, self.people - 1)))

	def _threshold(self, rounds_left: int) -> float:
		"""A quantile of all scores we know of (sampled and real) that falls as
		rounds and singles run out. Never below zero."""
		pool = sorted(self.sample_scores + list(self.estimates.values()))
		if not pool:
			return 0.0
		q = self._gamma(rounds_left) * TOP_QUANTILE
		return max(0.0, pool[min(len(pool) - 1, int(q * len(pool)))])

	# ----------------------------------------------------------- liar detection

	def _detect_lie(self, partner: int, observed: Profile, stated: Profile) -> bool:
		past_proj = self.projections.get(partner, [])
		past_ideal = self.stated_ideals.get(partner, [])
		our_past = self.my_described.get(partner, [])[:-1]  # exclude today's
		if any(tuple(p) != tuple(observed) for p in past_proj):
			return True
		if any(tuple(i) != tuple(stated) for i in past_ideal):
			return True
		if _similar(stated, self.profile):  # mirrors what we projected today
			return True
		return any(_similar(observed, ours) for ours in our_past)

	# ---------------------------------------------------------------- the round

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self.singles = turn.singles
		if turn.round > self.observe_round_cap:
			self.observing = False
		for c in candidates:
			if c.index not in self.ads_seen:
				self.ads_seen.add(c.index)
				self._observe_value(c.ad.own_dim, c.ad.own_value)
		self._refresh_model()

		pool = [c for c in candidates if c.index not in self.liars]
		fresh = sorted(
			(c for c in pool if c.index not in self.estimates),
			key=lambda c: (-self._ad_score(c.ad), c.index),
		)
		known = sorted(
			(c for c in pool if self.estimates.get(c.index, 0.0) > 0),
			key=lambda c: (-self.estimates[c.index], c.index),
		)
		if self.observing:
			# Grow the number of unique people seen, then revisit the good ones.
			chosen = fresh + known
		else:
			bar = self._threshold(turn.rounds_left)
			good = [c for c in known if self.estimates[c.index] >= bar]
			rest = [c for c in known if self.estimates[c.index] < bar]
			chosen = good + fresh + rest
		return [c.index for c in chosen][: self.list_limit]

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		partner = date.partner
		# Close the observe phase once enough unique people have been dated
		# (estimates has one entry per unique partner, so repeats never count).
		if self.observing and (
			len(self.estimates) >= self.observe_target or date.round > self.observe_round_cap
		):
			self.observing = False

		lied = self._detect_lie(partner, observed, stated_ideal)
		self.projections.setdefault(partner, []).append(tuple(observed))
		self.stated_ideals.setdefault(partner, []).append(tuple(stated_ideal))
		if partner not in self.projections_seen:
			self.projections_seen.add(partner)
			for dim, value in enumerate(observed):
				self._observe_value(dim, value)

		estimate = self._score(observed, stated_ideal)
		self.estimates[partner] = estimate
		past = self.estimate_history.setdefault(partner, [])
		decreased = bool(past) and estimate < max(past)
		past.append(estimate)
		if lied:
			self.liars.add(partner)

		if estimate < 0:
			return Decision(incompatible=True)
		# Long repeat streaks: a dropping score ends it, a steady one gets a proposal.
		repeated = date.date_number > REPEAT_PROPOSE_AFTER
		if repeated and decreased:
			return Decision(incompatible=True)
		if partner in self.liars:
			return Decision()

		best = estimate >= self.best_estimate
		self.best_estimate = max(self.best_estimate, estimate)
		can_propose = date.date_number >= 2

		if repeated and estimate > 0:
			return Decision(again=True, propose=True)
		# They already proposed to us: accept in either phase unless their score dropped.
		if partner in self.proposed_to_me and not decreased:
			return Decision(again=True, propose=can_propose)

		if self.observing:
			return Decision(again=best and estimate > 0)

		if estimate > 0 and estimate >= self._threshold(date.rounds_left):
			return Decision(again=True, propose=can_propose)
		return Decision(again=best and estimate > 0)

	def on_outcome(self, outcome: DateOutcome) -> None:
		if outcome.partner_proposed:
			self.proposed_to_me.add(outcome.partner)
		if self.observing and len(self.estimates) >= self.observe_target:
			self.observing = False
