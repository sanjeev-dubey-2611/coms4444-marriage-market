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
BAR_MODE = 'reservation'

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
		self._observe(self.profile)
		self._observe(self.ideal)
		self._outlook: list[tuple[float, float]] | None = None  # cache, see _outlook_dist
		self._expected: tuple[list[float], list[float]] | None = None  # cache, see _ad_score
		self.unmet = self.people - 1  # candidates we have never dated, from choose_dates

	def advertise(self) -> Advertisement:
		"""Called once, before round 1. Fixed for the whole simulation.

		Post one of your own characteristics and one characteristic you are
		looking for. Neither has to be true.
		"""
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
		"""
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
		mine = decrement(self.ideal, observed, self.monotonic)
		theirs = decrement(stated_ideal, self.profile, self.monotonic)
		h = self.bonus - mine - theirs / 2

		# Average over all dates with this person.
		# Bayesian update, once per person.
		if date.partner not in self.sampled:
			self.sampled.add(date.partner)
			self._observe(observed)
			self._observe(stated_ideal)

		hist = self.history.setdefault(date.partner, [])
		hist.append(h)
		estimate = sum(hist) / len(hist)
		self.estimates[date.partner] = estimate

		if estimate <= 0:
			return Decision(incompatible=True)

		bar = self._bar(date.rounds_left)
		if date.partner in self.proposed_to_me:
			bar = 0.05 * self.bonus
		propose = date.date_number >= 2 and estimate > bar
		# Only ask for another date if they already clear the proposal bar;
		# repeat dates spent on lukewarm matches cost rounds we could search with.
		again = propose or estimate > bar
		return Decision(again=again, propose=propose)

	# ------------------------------------------------------------------ odds

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
		mine[ad.own_dim] = self._gap(self.ideal[ad.own_dim], ad.own_value, ad.own_dim)
		theirs[ad.seek_dim] = self._gap(ad.seek_value, self.profile[ad.seek_dim], ad.seek_dim)
		return self.bonus - sum(mine) - sum(theirs) / 2

	def _odds(self) -> float:
		"""Chance a random partner would give us positive happiness."""
		return sum(p for h, p in self._outlook_dist() if h > 0)

	def _bar(self, rounds_left: int) -> float:
		"""How good a match must be before we propose."""
		frac = (rounds_left - 1) / max(1, self.rounds - 1)
		schedule = self.bonus * (0.05 + 0.4 * frac)
		if BAR_MODE == 'scaled':
			return schedule * min(1.0, self._odds() / ODDS_REFERENCE)
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
