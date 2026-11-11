"""Starting point for a group's player.

Copy this whole directory to ``players/player_<k>/`` using your group number,
then rename the class to ``Player<k>``. Group 4 would end up with
``players/player_4/player.py`` containing ``class Player4``. The registry looks
for exactly that; nothing else needs editing.

Keep the ``__init__.py``. Discovery uses ``pkgutil.iter_modules``, which only
reports directories that have one, so a group directory without it is silently
invisible to the simulator - no error, just a player that never turns up.

This directory is not itself discovered - the registry only matches
``player_<digits>`` - so the template can never appear in a run as a competitor.
"""

from math import ceil, sqrt

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

# Dirichlet prior pseudo-count per value: 1 is a flat prior over 0-10.
PRIOR_COUNT = 1.0

# How our proposal bar is set (see _bar):
#   'original'    - Sachi's schedule: 0.45*B at the start, falling to 0.05*B.
#   'scaled'      - that schedule, shrunk when our odds of a good match are low.
#   'reservation' - the happiness we can still expect to beat by searching on.
#   'optimal'     - optimal stopping: backward induction over the full joint
#                   (our happiness, their happiness) distribution of a random
#                   partner. The bar is the value of searching on, W(k-1).
BAR_MODE = 'optimal'

# How choose_dates picks people we have never met (see _ad_score):
#   'original'       - a random half of them.
#   'adfit_half'     - the half whose ads promise the most happiness.
#   'adfit_quarter'  - the best quarter.
#   'adfit_positive' - everyone whose ad promises positive happiness.
#   'topk'           - one shortlist of everyone, met or not, best first.
DATE_MODE = 'adfit_half'
# 'scaled': odds at or above this keep the full schedule; below it, the bar
# shrinks in proportion.
ODDS_REFERENCE = 0.5
# 'reservation': share of the people we meet who would also commit to us.
COMMIT_RATE = 0.5

# 'optimal': multiplier on the computed bar. Sweep 0.8-1.2 in simulation,
# since the model assumes partners behave like us.
BAR_SCALE = 1.0
# 'optimal': the primary goal is to end up married with positive happiness, so
# a marriage with h > 0 is worth h + POSITIVE_BONUS * B in the recursion and
# staying single is worth 0. 0 maximizes pure expected happiness (this made the
# bar too high); large values approach 'accept anything positive'.
# Sweep 0, 0.25, 0.5, 1.0.
POSITIVE_BONUS = 0.5
# 'optimal': the recursion assumes a fresh partner every round and a pool that
# never thins. In reality good people marry others, so trust only this share of
# the remaining rounds. 1.0 = take the model at face value.
HORIZON_SCALE = 0.5
# --- Liars and safety (see _cautious_gap, decide) ---
# Players like us shade the one trait our ad asks for by up to this many steps.
# We assume the true value is that much worse than what we saw on that trait.
LIE_SLACK = 2
# How much we believe the shading happened (0 = take projections at face value).
LIE_WEIGHT = 1.0
# Repeat dates: penalise partners whose scores jump around between dates.
STD_PENALTY = 1.0
# Their projected profile moved more than this (sum of |changes|) since the last date.
CONSISTENCY_TOL = 4
# Their ad's own value is at least this far from what we observed on that trait.
AD_MISMATCH = 3
# Extra happiness (fraction of B) we subtract for partners caught lying.
SUSPECT_FRAC = 0.10
# Ads boasting on a monotonic trait are only partly trusted in _ad_score.
AD_TRUST = 0.7
# --- Marry for positive benefit ---
# We never propose below MARGIN_FRAC * B (a safety margin above zero, since our
# estimate of their traits can be off).
MARGIN_FRAC = 0.05
# The proposal bar never exceeds CAP_FRAC * B: holding out for a perfect match
# risks ending single, which scores 0 on the primary goal.
CAP_FRAC = 0.65
# The bar falls to the margin over the final URGENCY rounds.
URGENCY = 8
# Ask for another date when the estimate beats AGAIN_FRAC * bar (a second date
# is the cheapest way to catch a liar).
AGAIN_FRAC = 0.6

# --- Exploit a high bonus ---
# When B is large almost every marriage is positive, so the primary goal is
# saturated and what separates players is average happiness. We estimate how
# likely we are to find a mutually positive partner (see _saturation) and blend
# the bar from 'safety first' (capped, positive bonus) toward 'maximise average
# happiness' (pure expected value, cap CAP_MAX) as that likelihood rises.
CAP_MAX = 0.95
# Share of the remaining fresh dates we count on when estimating that
# likelihood; below 1 because good partners marry others and ads mislead.
SAT_SCALE = 0.03
# Same idea for second dates: when good matches are plentiful, only ask again
# for matches we would marry. This one can safely be larger.
AGAIN_SAT_SCALE = 0.1

# After their proposal acceptance is certain, so we lower the bar to
# 0.05 * B as the original code did.
LOWER_BAR_ON_PROPOSAL = True
# project(): stay honest with people we have met and rate below our bar.
# Untested idea, off by default.
HONEST_IF_UNWANTED = False
# 'optimal': cost bins are this fraction of B wide (coarsening keeps the
# pair list small so the recursion stays fast).
BIN_FRACTION = 1 / 40
# 'optimal': ignore (a, b) combinations rarer than this.
MIN_PROB = 1e-9


class Player8(BasePlayer):
	"""Rename me to Player<k>, where <k> is your group number.

	One instance of this class is one person. Your code may be used for several
	people in the same simulation - perhaps all of them - with one instance per
	person. Do not share state between instances (class attributes, module
	globals, files): that is exactly the kind of bypass the handout forbids.
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)

		# super() has already set these from ctx and snapshot:
		#
		#   self.index      which person you are (0-based); others refer to you by it
		#   self.id         your UUID, stable for the whole simulation
		#   self.profile    your true profile T_p, a tuple of d ints in 0-10
		#   self.ideal      your desired profile D_p, a tuple of d ints in 0-10
		#   self.people     N, how many people are in the simulation
		#   self.dims       d, the number of dimensions
		#   self.monotonic  m, the first m dimensions are monotonic
		#   self.rounds     t, how many rounds of dating there are
		#   self.bonus      B, the marriage bonus
		#   self.rng        private seeded random generator; use instead of random
		#   self.list_limit maximum returned names, or None if disabled
		#
		# The engine constructs you once, before round 1. Anything you want to
		# carry between rounds lives on self, so initialise it here.
		# Added estimates and proposed_to_me to keep track of the estimated happiness with each candidate and whether they proposed to me.
		self.history: dict[int, list[Profile]] = {}
		self.estimates: dict[int, float] = {}
		self.proposed_to_me: set[int] = set()

		# Bayesian estimate of the trait distribution P, one per dimension.
		# Every true profile and ideal is drawn from the same P, so our own pair
		# and each partner's projection and stated ideal are samples of it.
		# Model: categorical over 0-10 with a Dirichlet prior, so the posterior
		# is prior pseudo-counts plus one per value observed.
		self.counts: list[list[float]] = [[PRIOR_COUNT] * 11 for _ in range(self.dims)]
		self.sampled: set[int] = set()  # partners already counted
		# Caches, all cleared whenever the belief changes (see _observe).
		self._outlook: list[tuple[float, float]] | None = None  # see _outlook_dist
		self._expected: tuple[list[float], list[float]] | None = None  # see _ad_score
		self._pair_cache: list[tuple[float, float, float]] | None = None  # see _pairs
		self._W: dict[float, list[float]] = {}  # see _values, keyed by positive bonus
		self._pmut: float | None = None  # see _saturation
		self.observed: dict[int, list[Profile]] = {}  # what each partner projected
		self.suspect: set[int] = set()  # partners we caught lying
		self._ad = self._pick_ad()
		self._observe(self.profile)
		self._observe(self.ideal)
		self.unmet = self.people - 1  # candidates we have never dated, from choose_dates

	def advertise(self) -> Advertisement:
		"""Called once, before round 1. Fixed for the whole simulation.

		Post one of your own characteristics and one characteristic you are
		looking for. Neither has to be true.
		"""
		return self._ad

	def _pick_ad(self) -> Advertisement:
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		# Seek the trait we care most about; on ties prefer a non-monotonic one,
		# where an exact match matters (from Sachi's version).
		seek = max(range(self.dims), key=lambda i: (abs(self.ideal[i] - 5), i >= self.monotonic))
		return Advertisement(
			own_dim=own, own_value=self.profile[own], seek_dim=seek, seek_value=self.ideal[seek]
		)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		"""Name the people you would like to date this round.

		Called once per round while you are single, unless you are already on a
		repeat date because you and your last date both asked for one.

		WHAT YOU CAN SEE

			candidate.index         who they are, stable for the whole run
			candidate.ad            their advertisement, possibly a lie
			candidate.dates         how many times you two have dated so far
			turn.round              this round, 1-based
			turn.rounds_left        rounds remaining, including this one
			turn.singles            how many people are still single

		HOW PAIRING WORKS

			1. Pairs who both asked for another date are paired first.
			2. Pairs who listed each other are paired at random.
			3. Pairs where only one listed the other are paired at random.
			4. Everyone left is paired at random.

		Nobody is ever paired with someone either side marked incompatible, so
		in rare cases you sit a round out.
		"""
		good = [c.index for c in candidates if self.estimates.get(c.index, 0.0) > 0]
		fresh = [c for c in candidates if c.index not in self.estimates]
		self.rng.shuffle(fresh)  # random tie-breaks for the stable sorts below
		if DATE_MODE == 'original':
			picked = fresh[: max(2, len(fresh) // 2)]
		elif DATE_MODE == 'topk':
			# One ranking for everyone: what we learned on dates for people we
			# have met, what their ad suggests for everyone else.
			pool = [c for c in candidates if c.index in good] + fresh
			pool.sort(key=lambda c: -self.estimates.get(c.index, self._ad_score(c.ad)))
			width = max(3, ceil(1.5 * sqrt(len(candidates))))
			good, picked = [], pool[:width]
		else:
			fresh.sort(key=lambda c: -self._ad_score(c.ad))
			if DATE_MODE == 'adfit_half':
				picked = fresh[: max(2, len(fresh) // 2)]
			elif DATE_MODE == 'adfit_quarter':
				picked = fresh[: max(2, len(fresh) // 4)]
			else:  # 'adfit_positive'
				picked = [c for c in fresh if self._ad_score(c.ad) > 0] or fresh[:2]
		chosen = good + [c.index for c in picked]
		if not chosen:
			chosen = [c.index for c in candidates]
		self.unmet = sum(1 for c in candidates if c.dates == 0)
		return chosen[: self.list_limit]

	def project(self, date: DateContext) -> Profile:
		"""The profile your behaviour projects on this date.

		Your date senses exactly what you return here. It does not have to be
		your true profile, but your final happiness is scored on true profiles
		only - and you also lose half of whatever your partner loses.

		We stay honest except on the one trait the partner's ad says they seek,
		which we nudge up to 2 steps toward their wish (from Sachi's version).
		Monotonic traits only move up, never past the wish or 10.

		New: we only shade the truth for people we still want. If we already
		dated this person and our estimate is below our bar, luring them into
		a match we would not take only costs us half their decrement.
		"""
		known = self.estimates.get(date.partner)
		if HONEST_IF_UNWANTED and known is not None:
			rounds_left = getattr(date, 'rounds_left', None)
			if rounds_left is not None and known <= self._bar(rounds_left):
				return tuple(self.profile)
		projected = list(self.profile)
		dim, value = date.partner_ad.seek_dim, date.partner_ad.seek_value
		if dim < self.monotonic:
			projected[dim] = min(10, max(projected[dim], min(value, projected[dim] + 2)))
		else:
			projected[dim] += max(-2, min(2, value - projected[dim]))
		return tuple(projected)

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		"""At the end of the date, describe the ideal partner you are seeking.

		``observed`` is what your date projected. Honest or not, it is up to you.
		"""
		return self.ideal

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		"""After the date: ask again, propose, or mark them incompatible.

		``observed`` is what they projected; ``stated_ideal`` is the ideal
		partner they described. ``date.date_number`` counts this date.

		RULES

			propose       only legal when date.date_number >= 2
			again         if you both ask, you are paired first next round
			incompatible  you will never be paired with them again

		If you both propose on the same date you are married and leave the pool
		for good. incompatible cannot be combined with again or propose.

		IF YOU GET IT WRONG

		An exception, a malformed return value, or taking longer than the
		--timeout budget is recorded as a fault and the engine substitutes a
		safe default: your true profile for project, your true ideal for
		describe_ideal, nobody for choose_dates, and Decision() (no again, no
		proposal, no incompatibility) for decide. Your failure never affects the
		other groups.
		"""
		# Cautious happiness: on the trait our ad asks for, partners may have
		# shaded the truth toward our wish, so assume the worst they could hide.
		mine = sum(self._cautious_gap(i, observed[i]) for i in range(self.dims))
		theirs = decrement(stated_ideal, self.profile, self.monotonic)
		h = self.bonus - mine - theirs / 2

		# Average over all dates with this person.
		# Bayesian update, once per person.
		if date.partner not in self.sampled:
			self.sampled.add(date.partner)
			self._observe(observed)
			self._observe(stated_ideal)

		# Liar checks: a projection that jumps between dates, or an ad that
		# does not match what we see, marks them as someone to trust less.
		seen = self.observed.setdefault(date.partner, [])
		if seen and sum(abs(a - b) for a, b in zip(seen[-1], observed)) > CONSISTENCY_TOL:
			self.suspect.add(date.partner)
		ad = getattr(date, 'partner_ad', None)
		if ad is not None and abs(observed[ad.own_dim] - ad.own_value) >= AD_MISMATCH:
			self.suspect.add(date.partner)
		seen.append(observed)

		hist = self.history.setdefault(date.partner, [])
		hist.append(h)
		mean = sum(hist) / len(hist)
		estimate = mean
		if len(hist) >= 2:
			var = sum((x - mean) ** 2 for x in hist) / len(hist)
			estimate -= STD_PENALTY * sqrt(var)
		if date.partner in self.suspect:
			estimate -= SUSPECT_FRAC * self.bonus
		self.estimates[date.partner] = estimate

		# Never marry for a negative benefit.
		if estimate <= 0:
			return Decision(incompatible=True)

		margin = MARGIN_FRAC * self.bonus
		bar = self._bar(date.rounds_left)
		if LOWER_BAR_ON_PROPOSAL and date.partner in self.proposed_to_me:
			bar = margin  # acceptance is certain; take any clearly positive match
		propose = date.date_number >= 2 and estimate > bar
		# A second date is the cheapest way to catch a liar, so be a bit lenient
		# when good matches are scarce. When they are plentiful, repeat dates
		# crowd out fresh ones, so only ask again for matches we would marry.
		sat = self._saturation(date.rounds_left, AGAIN_SAT_SCALE) if BAR_MODE == 'optimal' else 0.0
		again_frac = AGAIN_FRAC + (1 - AGAIN_FRAC) * sat
		floor = margin if BAR_MODE == 'optimal' else 0.0
		again = propose or estimate > max(floor, again_frac * bar)
		return Decision(again=again, propose=propose)

	# ------------------------------------------------------------------ odds

	def _cautious_gap(self, dim: int, seen: int) -> float:
		"""Gap in one dimension, allowing for a shaded projection.

		Only the trait our ad seeks is exposed to shading, and shading always
		moves the projection toward our wish, so the truth lies on the far side.
		Taking the worst case there penalised everyone who shaded (most of the
		market), so instead we average the gap over the values the truth could
		take, weighted by our belief about P, and blend that with the face-value
		reading (LIE_WEIGHT is how much we believe the shading happened).
		"""
		want = self.ideal[dim]
		plain = self._gap(want, seen, dim)
		if LIE_SLACK <= 0 or LIE_WEIGHT <= 0 or dim != self._ad.seek_dim:
			return plain
		lo, hi = max(0, seen - LIE_SLACK), min(10, seen + LIE_SLACK)
		if dim < self.monotonic:
			options = [seen] if seen > want else list(range(lo, seen + 1))
		elif seen < want:
			options = list(range(lo, seen + 1))
		elif seen > want:
			options = list(range(seen, hi + 1))
		else:
			options = list(range(max(0, want - LIE_SLACK), min(10, want + LIE_SLACK) + 1))
		belief = self._belief(dim)
		weights = [belief[v] for v in options]
		total = sum(weights)
		expected = sum(w * self._gap(want, v, dim) for w, v in zip(weights, options)) / total
		return (1 - LIE_WEIGHT) * plain + LIE_WEIGHT * expected

	def _gap(self, want: int, have: int, dim: int) -> int:
		"""One dimension's share of a decrement, as the scorer computes it."""
		if dim < self.monotonic and have >= want:
			return 0
		return (want - have) ** 2

	def _observe(self, values: Profile) -> None:
		"""Bayesian update: one observed profile adds one count per dimension."""
		for i, v in enumerate(values):
			self.counts[i][v] += 1
		self._outlook = None
		self._expected = None
		self._pair_cache = None
		self._W = {}
		self._pmut = None

	def _belief(self, dim: int) -> list[float]:
		"""Posterior predictive for one dimension: P(value = v) = count_v / total."""
		row = self.counts[dim]
		total = sum(row)
		return [c / total for c in row]

	def _outlook_dist(self) -> list[tuple[float, float]]:
		"""Our happiness with a random partner, as (happiness, probability) pairs.

		A random partner's true profile and ideal are independent draws from our
		estimate of P, so each dimension adds an independent cost. Convolving
		them gives the exact distribution of B - our decrement - their decrement / 2.
		Costs are kept doubled so every key is an integer.
		"""
		if self._outlook is None:
			dist = {0: 1.0}
			for i in range(self.dims):
				belief = self._belief(i)
				mine: dict[int, float] = {}
				theirs: dict[int, float] = {}
				for v, p in enumerate(belief):
					k = 2 * self._gap(self.ideal[i], v, i)  # their trait v vs our ideal
					mine[k] = mine.get(k, 0.0) + p
					k = self._gap(v, self.profile[i], i)  # their wish v vs our trait
					theirs[k] = theirs.get(k, 0.0) + p
				for part in (mine, theirs):
					step: dict[int, float] = {}
					for a, pa in dist.items():
						for b, pb in part.items():
							step[a + b] = step.get(a + b, 0.0) + pa * pb
					dist = step
			self._outlook = sorted((self.bonus - k / 2, p) for k, p in dist.items())
		return self._outlook

	def _ad_score(self, ad: Advertisement) -> float:
		"""Expected happiness with someone we only know by their ad.

		Every dimension costs what a random person would cost on average under
		our estimate of P, except the two the ad reveals: their advertised trait
		replaces our guess of their value, and their sought value replaces our
		guess of what they want there.
		"""
		if self._expected is None:
			mine, theirs = [], []
			for i in range(self.dims):
				belief = self._belief(i)
				mine.append(sum(p * self._gap(self.ideal[i], v, i) for v, p in enumerate(belief)))
				theirs.append(
					sum(p * self._gap(v, self.profile[i], i) for v, p in enumerate(belief))
				)
			self._expected = (mine, theirs)
		mine, theirs = list(self._expected[0]), list(self._expected[1])
		claimed = self._gap(self.ideal[ad.own_dim], ad.own_value, ad.own_dim)
		if ad.own_dim < self.monotonic:
			# Boasting on a trait everyone wants is cheap, so trust it only partly.
			claimed = AD_TRUST * claimed + (1 - AD_TRUST) * mine[ad.own_dim]
		mine[ad.own_dim] = claimed
		theirs[ad.seek_dim] = self._gap(ad.seek_value, self.profile[ad.seek_dim], ad.seek_dim)
		return self.bonus - sum(mine) - sum(theirs) / 2

	def _odds(self) -> float:
		"""Chance a random partner would give us positive happiness."""
		return sum(p for h, p in self._outlook_dist() if h > 0)

	# --------------------------------------------------------- optimal stopping

	def _cost_dist(self, cost) -> dict[int, float]:
		"""Distribution of a summed per-dimension cost for a random partner.

		``cost(i, v)`` is the cost in dimension i if the partner's relevant
		value there is v (drawn from our belief). Dimensions are independent,
		so the total is a convolution. Totals are binned to BIN_FRACTION * B to
		keep the support small; the bin index is the key.
		"""
		width = max(1.0, self.bonus * BIN_FRACTION)
		dist: dict[int, float] = {0: 1.0}
		for i in range(self.dims):
			part: dict[int, float] = {}
			for v, p in enumerate(self._belief(i)):
				k = cost(i, v)
				part[k] = part.get(k, 0.0) + p
			step: dict[int, float] = {}
			for x, px in dist.items():
				for y, py in part.items():
					step[x + y] = step.get(x + y, 0.0) + px * py
			dist = step
		binned: dict[int, float] = {}
		for x, px in dist.items():
			b = int(round(x / width))
			binned[b] = binned.get(b, 0.0) + px
		# Convert bin index back to a representative cost (key = cost in grid units).
		return {int(round(b * width)): p for b, p in binned.items() if p > MIN_PROB}

	def _pairs(self) -> list[tuple[float, float, float]]:
		"""Joint distribution of (our happiness, their happiness, probability).

		a = our decrement (their traits vs our ideal), b = their decrement
		(their ideal vs our traits). Assuming they judge us as we judge them:
			our happiness   = B - a - b/2
			their happiness = B - b - a/2
		"""
		if self._pair_cache is None:
			a_dist = self._cost_dist(lambda i, v: self._gap(self.ideal[i], v, i))
			b_dist = self._cost_dist(lambda i, v: self._gap(v, self.profile[i], i))
			self._pair_cache = [
				(self.bonus - a - b / 2, self.bonus - b - a / 2, pa * pb)
				for a, pa in a_dist.items()
				for b, pb in b_dist.items()
				if pa * pb > MIN_PROB and a + b / 2 < self.bonus and b + a / 2 < self.bonus
			]
		return self._pair_cache

	def _utility(self, h: float, pb: float) -> float:
		"""Value of ending married at happiness h (single = 0)."""
		return h + pb * self.bonus if h > 0 else 0.0

	def _saturation(self, rounds_left: int, scale: float | None = None) -> float:
		"""Chance we still find a partner who is positive for both of us.

		Near 1 when B is high (nearly everyone is a positive match), so the
		primary goal is safe and we can hold out for happiness. Near 0 when
		positive matches are scarce, so we marry for any positive benefit.
		"""
		if self._pmut is None:
			self._pmut = min(1.0, sum(p for _, _, p in self._pairs()))
		draws = (SAT_SCALE if scale is None else scale) * min(self.unmet, max(0, (rounds_left - 1) // 2))
		if draws <= 0:
			return 0.0
		return 1.0 - (1.0 - self._pmut) ** draws

	def _values(self, pb: float) -> list[float]:
		"""W[k]: expected final happiness with k rounds left, playing optimally.

		W[0] = W[1] = 0 (a marriage needs two dates). With r = W[k-1], the
		value of searching on, a random partner marries us only if both sides
		beat r; otherwise we walk away after one round and hold r.
		"""
		if pb not in self._W:
			pairs = self._pairs()
			W = [0.0, 0.0]
			for _ in range(2, self.rounds + 1):
				r = W[-1]
				# Pairs not listed (someone's h <= 0) always end in a walk-away worth r.
				gain = 0.0
				for h, h2, p in pairs:
					u1, u2 = self._utility(h, pb), self._utility(h2, pb)
					if u1 > r and u2 > r:
						gain += p * (u1 - r)
				W.append(r + gain)
			self._W[pb] = W
		return self._W[pb]

	def _search_bar(self, rounds_left: int, pb: float, cap: float) -> float:
		"""Value of searching on, as a threshold on our happiness."""
		W = self._values(pb)
		# We cannot search forever: each fresh person we can still meet
		# uses at most two rounds, so cap the horizon by the finite pool.
		horizon = min(int((rounds_left - 1) * HORIZON_SCALE), 2 * self.unmet)
		horizon = min(max(horizon, 0), len(W) - 1)
		# W is in utility units; utility(h) > W  <=>  h > W - pb * B  (and h > 0).
		raw = max(0.0, W[horizon] - pb * self.bonus) * BAR_SCALE
		return max(MARGIN_FRAC * self.bonus, min(raw, cap * self.bonus))

	def _bar(self, rounds_left: int) -> float:
		"""How good a match must be before we propose."""
		frac = (rounds_left - 1) / max(1, self.rounds - 1)
		schedule = self.bonus * (0.05 + 0.4 * frac)
		if BAR_MODE == 'scaled':
			return schedule * min(1.0, self._odds() / ODDS_REFERENCE)
		if BAR_MODE == 'optimal':
			margin = MARGIN_FRAC * self.bonus
			sat = self._saturation(rounds_left)
			careful = self._search_bar(rounds_left, POSITIVE_BONUS, CAP_FRAC)
			greedy = self._search_bar(rounds_left, 0.0, CAP_MAX)
			bar = (1 - sat) * careful + sat * greedy
			# Time pressure: marrying takes two dates, so near the end settle
			# for anything clearly positive. Less needed when matches are plentiful.
			window = max(3.0, URGENCY * (1 - sat))
			urgency = max(0.0, min(1.0, (rounds_left - 2) / window))
			return margin + (bar - margin) * urgency
		if BAR_MODE == 'reservation':
			# Fresh people we can still court (each needs two dates), of whom only
			# some would commit. Set the bar where at least one of them beats it
			# with even chances; scarce good matches mean a low bar.
			chances = COMMIT_RATE * min(self.unmet, rounds_left // 2)
			if chances < 1:
				return 0.0
			tail = 1 - 0.5 ** (1 / chances)  # each one must beat the bar this often
			above = 0.0
			for h, p in reversed(self._outlook_dist()):
				above += p
				if above >= tail:
					return max(0.0, h)
			return 0.0
		return schedule

	def on_outcome(self, outcome: DateOutcome) -> None:
		"""Told what happened after both of you decided.

		outcome.partner_proposed  they proposed to you this round
		outcome.married           you are now married (to them)
		outcome.again             you both asked; you meet again next round
		"""
		if outcome.partner_proposed:
			self.proposed_to_me.add(outcome.partner)