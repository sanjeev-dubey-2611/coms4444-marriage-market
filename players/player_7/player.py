"""Group 7: honest dating, then marriage by a shared matching rule.

If both people on a date are honest, the date reveals the other person's true
profile and true ideal, so each of us can work out exactly how happy *both*
people would be in that marriage. After dating most of the market, a person
therefore knows the whole compatibility graph among the singles they have met:
an edge joins two people when both would end up with positive happiness.

Our primary goal is the share of people with positive happiness, which is the
size of a matching in that graph. Every copy of this player computes the same
deterministic matching from the same information and pursues only its assigned
partner, so copies agree on who marries whom without sharing any state. The
matching repeatedly pairs the person with the fewest compatible partners, so
people with many options do not take the only option of someone with few.

Copies recognise each other honestly: our ad's dimensions are a checksum of
our true profile and ideal, which a date reveals. Only recognised copies go
into the shared matching. Anyone else will not wait for a matching they do not
compute, so we take them once they beat both our assigned partner and a
patience bar: what we expect to end up with if we hold out for the people we
can still meet. The bar is reasoned dimension by dimension. Under the values
we have seen, each dimension has an expected cost and a variance, and since
dimensions are drawn independently these add up to the mean and spread of our
happiness with a random partner. The bar falls to zero as the unmet people
run out, so we stay picky early and never miss the end. We stop counting on
anyone who snubs us twice, and we block outsiders who are hopeless for us so
the engine stops spending our dates on them.

1. Explore: list people we have not met, ranked by how well their ad fits us.
2. Commit: once we have met everyone, list only our assigned partner, ask for
   repeat dates, and propose from the second date on.
3. Fall back: if the assigned partner does not respond, or time is running
   out, propose to any compatible person, best first.

We never propose unless our own happiness would be positive. A partner whose
projected profile changes between dates or contradicts their ad is treated as
no better than a random person and is left out of the shared graph. A partner
whose stated ideal is just our own profile is flattering us: we assume an
average ideal for them and that the value our ad asked for was nudged by 2.
"""

import math
from statistics import NormalDist

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

VALUES = range(11)
NORMAL = NormalDist()

# Pseudo-counts per value before any date: a broad bell around 5, so the first
# ad-based estimates are sensible and real observations soon dominate.
PRIOR = [0.5 + 1.5 * math.exp(-((v - 5) ** 2) / 12.5) for v in VALUES]

# Rounds without a date with our assigned partner before we set them aside,
# and how long they stay set aside.
PATIENCE = 4
COOLDOWN = 10
# Rounds without meeting anyone new before we stop exploring: whoever is left
# has probably marked us incompatible, so we can never be paired with them.
STALL = 5
# In the last few rounds, propose to anyone compatible.
ENDGAME = 6
# A person not seen among the candidates for this many rounds has married.
GONE_AFTER = 3
# Unanswered proposals or refused repeat dates before we stop counting on
# someone outside our group: they keep dating us but will not marry us.
SNUBS = 2
# The chance that any one person we have yet to meet turns into a real option
# for us, when deciding how picky to be with outsiders.
CHANCE = 0.2
# A stated ideal within this distance of our own profile is flattery.
MIRROR = 1
# How far a flatterer may have nudged the value our ad asked for.
NUDGE = 2


def _ad(profile: Profile, ideal: Profile) -> Advertisement:
	"""Our honest ad, with both dimensions picked by a checksum of our profile
	and ideal. A date shows us a partner's true profile and ideal, and if they
	produce exactly the ad that partner showed, the partner runs this player.
	Other groups pick dimensions by value, so they rarely match by chance."""
	d = len(profile)
	h = sum(
		(i + 1) * (3 * p + 7 * q + 1) for i, (p, q) in enumerate(zip(profile, ideal, strict=True))
	)
	own, seek = h % d, h // d % d
	return Advertisement(own_dim=own, own_value=profile[own], seek_dim=seek, seek_value=ideal[seek])


class Player7(BasePlayer):
	"""One honest person who marries the partner a shared matching assigns."""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.counts = [list(PRIOR) for _ in range(self.dims)]
		self.costs: tuple[list[float], list[float]] | None = None
		# What each person showed and said on every date with us.
		self.shown: dict[int, list[Profile]] = {}
		self.stated: dict[int, list[Profile]] = {}
		self.ads: dict[int, Advertisement] = {}
		self.last_seen: dict[int, int] = {}
		self.proposed_to_me: set[int] = set()
		# Partners who ignored us, and how long we have waited on the current one.
		self.unresponsive: dict[int, int] = {}
		# How often each person ignored our proposal or our request to meet again.
		self.snubs: dict[int, int] = {}
		self.asked: dict[int, Decision] = {}
		self.last_new = 0
		self.exploring = True
		self.target: int | None = None
		self.waited = 0
		self.round = 0
		self.unmet = 0

	# ------------------------------------------------------------------
	# Scoring
	# ------------------------------------------------------------------

	def _cost(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		if dim < self.monotonic and gap <= 0:
			return 0
		return gap * gap

	def _expected_costs(self) -> tuple[list[float], list[float]]:
		"""Per dimension, our expected cost from a random partner's value, and a
		random partner's expected cost from ours, under the learned distribution."""
		if self.costs is None:
			mine, theirs = [], []
			for dim, row in enumerate(self.counts):
				total = sum(row)
				want, have = self.ideal[dim], self.profile[dim]
				mine.append(
					sum(n * self._cost(dim, want, v) for v, n in zip(VALUES, row, strict=True))
					/ total
				)
				theirs.append(
					sum(n * self._cost(dim, w, have) for w, n in zip(VALUES, row, strict=True))
					/ total
				)
			self.costs = (mine, theirs)
		return self.costs

	def _from_ad(self, ad: Advertisement) -> float:
		"""Our expected happiness with someone we have not met, from their ad."""
		mine, theirs = self._expected_costs()
		ours = sum(mine) - mine[ad.own_dim]
		ours += self._cost(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		their = sum(theirs) - theirs[ad.seek_dim]
		their += self._cost(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		return self.bonus - ours - their / 2

	def _trusted(self, who: int) -> bool:
		"""Honest people show the same profile every time, matching their ad."""
		if self._mirror(who):
			return False
		stated = self.stated[who]
		return self._consistent(who) and all(s == stated[0] for s in stated)

	def _consistent(self, who: int) -> bool:
		"""They show the same profile every time, matching their ad."""
		shown = self.shown[who]
		if any(p != shown[0] for p in shown):
			return False
		ad = self.ads.get(who)
		return ad is None or shown[0][ad.own_dim] == ad.own_value

	def _mirror(self, who: int) -> bool:
		"""Their stated ideal is (almost) exactly our own profile: flattery
		rather than a real ideal, which an honest person almost never has."""
		stated = self.stated[who][-1]
		return sum(abs(a - b) for a, b in zip(stated, self.profile, strict=True)) <= MIRROR

	def _with_me(self, who: int) -> tuple[float, float]:
		"""(our happiness, their happiness) if we married ``who``. For someone we
		do not trust, every dimension costs at least what a random person costs."""
		mine, theirs = self._expected_costs()
		if self._mirror(who) and self._consistent(who):
			# A flatterer's stated ideal tells us nothing, so assume an average
			# one. Their profile mostly stands, but the value our ad asked for
			# may have been nudged in our favour.
			seek = self.advertise().seek_dim
			shown = self.shown[who][0]
			ours = 0.0
			for dim in range(self.dims):
				near = range(max(0, shown[dim] - NUDGE), min(10, shown[dim] + NUDGE) + 1)
				if dim == seek:
					ours += max(self._cost(dim, self.ideal[dim], v) for v in near)
				else:
					ours += self._cost(dim, self.ideal[dim], shown[dim])
			their = sum(theirs)
			return self.bonus - ours - their / 2, self.bonus - their - ours / 2
		trusted = self._trusted(who)
		ours = 0.0
		for dim in range(self.dims):
			worst = max(self._cost(dim, self.ideal[dim], p[dim]) for p in self.shown[who])
			ours += worst if trusted else max(worst, mine[dim])
		their = max(decrement(s, self.profile, self.monotonic) for s in self.stated[who])
		return self.bonus - ours - their / 2, self.bonus - their - ours / 2

	def _optimistic(self, who: int) -> float:
		"""Our happiness with ``who`` if what they showed and said were true."""
		ours = min(
			sum(self._cost(d, self.ideal[d], p[d]) for d in range(self.dims))
			for p in self.shown[who]
		)
		their = min(decrement(s, self.profile, self.monotonic) for s in self.stated[who])
		return self.bonus - ours - their / 2

	def _pair(self, a: int, b: int) -> float | None:
		"""Total happiness if two other people married, or None unless both
		would be positive. Uses what each showed us, which honest people share."""
		ta, da = self.shown[a][0], self.stated[a][0]
		tb, db = self.shown[b][0], self.stated[b][0]
		ca = decrement(da, tb, self.monotonic)
		cb = decrement(db, ta, self.monotonic)
		ha, hb = self.bonus - ca - cb / 2, self.bonus - cb - ca / 2
		return ha + hb if ha > 0 and hb > 0 else None

	# ------------------------------------------------------------------
	# Matching
	# ------------------------------------------------------------------

	def _kin(self, who: int) -> bool:
		"""Whether ``who`` runs this player: honest, and their ad is exactly the
		one we would write for the profile and ideal they showed us."""
		return self._trusted(who) and self.ads.get(who) == _ad(
			self.shown[who][0], self.stated[who][0]
		)

	def _singles(self) -> list[int]:
		return [
			who
			for who in self.shown
			if self.round - self.last_seen.get(who, -GONE_AFTER) < GONE_AFTER and self._kin(who)
		]

	def _assign(self) -> int | None:
		"""Our partner in the matching of every single we know, or None.

		Every copy of this player that knows the same people builds the same
		graph and gets the same answer, so assignments agree."""
		me = self.index
		people = sorted(self._singles())
		edges: dict[int, set[int]] = {p: set() for p in people + [me]}
		weight: dict[tuple[int, int], float] = {}
		for p in people:
			if self.round - self.unresponsive.get(p, -COOLDOWN) < COOLDOWN or self._snubbed(p):
				continue
			ours, their = self._with_me(p)
			if ours > 0 and their > 0:
				edges[me].add(p)
				edges[p].add(me)
				weight[(min(me, p), max(me, p))] = ours + their
		for i, a in enumerate(people):
			for b in people[i + 1 :]:
				total = self._pair(a, b)
				if total is not None:
					edges[a].add(b)
					edges[b].add(a)
					weight[(min(a, b), max(a, b))] = total

		# Repeatedly match the person with the fewest compatible partners to
		# their neighbour with the fewest, breaking ties by total happiness.
		while True:
			live = [p for p in edges if edges[p]]
			if not live:
				return None
			u = min(live, key=lambda p: (len(edges[p]), p))
			v = min(edges[u], key=lambda p: (len(edges[p]), -weight[(min(u, p), max(u, p))], p))
			if me in (u, v):
				return v if u == me else u
			for x in (u, v):
				for y in edges[x]:
					edges[y].discard(x)
				edges[x] = set()

	def _market(self) -> tuple[float, float]:
		"""Mean and spread of our happiness with a random partner.

		Each dimension is reasoned about on its own. Under the values seen so
		far it has an expected cost and a variance, once for what we would lose
		on it and once for the half of a random partner's loss that we share.
		Dimensions are drawn independently, so the totals are just the sums."""
		mean, variance = self.bonus, 0.0
		for dim, row in enumerate(self.counts):
			total = sum(row)
			ours = [self._cost(dim, self.ideal[dim], v) for v in VALUES]
			theirs = [self._cost(dim, w, self.profile[dim]) / 2 for w in VALUES]
			for costs in (ours, theirs):
				first = sum(n * c for n, c in zip(row, costs, strict=True)) / total
				second = sum(n * c * c for n, c in zip(row, costs, strict=True)) / total
				mean -= first
				variance += second - first * first
		return mean, math.sqrt(variance)

	def _bar(self, rounds_left: int) -> float:
		"""The happiness an outsider must offer us now: what we expect to end up
		with if we hold out for the people we can still meet.

		Holding out with nobody left to meet is worth zero. Each further person
		is a random partner who turns into a real option with probability
		CHANCE, and raises the bar by the amount they are expected to beat it.
		It falls to zero as the unmet people run out."""
		mean, spread = self._market()
		bar = 0.0
		if spread <= 0:
			return bar
		for _ in range(min(self.unmet, max(0, rounds_left - ENDGAME))):
			z = (mean - bar) / spread
			gain = CHANCE * spread * (NORMAL.pdf(z) + z * NORMAL.cdf(z))
			if gain < 0.01:
				break
			bar += gain
		return bar

	def _snubbed(self, who: int) -> bool:
		if who in self.proposed_to_me or self._kin(who):
			return False
		return self.snubs.get(who, 0) >= SNUBS

	def _best_compatible(self) -> list[int]:
		"""Singles we would be positive with, best for us first. Outsiders count
		even when we think they would not be: whether to accept is their call."""
		scored = []
		for who in self.shown:
			if self.round - self.last_seen.get(who, -GONE_AFTER) >= GONE_AFTER:
				continue
			ours, their = self._with_me(who)
			likely = their > 0 or who in self.proposed_to_me
			if ours > 0 and (likely or not self._kin(who)):
				scored.append((not self._snubbed(who), likely, ours, who))
		return [who for *_, who in sorted(scored, reverse=True)]

	# ------------------------------------------------------------------
	# Interface
	# ------------------------------------------------------------------

	def advertise(self) -> Advertisement:
		return _ad(self.profile, self.ideal)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self.round = turn.round
		for c in candidates:
			self.ads[c.index] = c.ad
			self.last_seen[c.index] = turn.round
		fresh = [c for c in candidates if c.index not in self.shown]
		self.unmet = len(fresh)
		limit = self.list_limit if self.list_limit is not None else len(candidates)

		self._retarget()
		late = turn.rounds_left <= ENDGAME
		# Keep meeting new people while there is time to meet them all.
		exploring = (
			bool(fresh)
			and turn.round - self.last_new <= STALL
			and turn.rounds_left > len(fresh) + ENDGAME
		)
		self.exploring = exploring

		offered = {c.index for c in candidates}
		target = self.target if self.target in offered else None
		if target is not None and not exploring:
			return [target]
		chosen = [target] if target is not None else []
		later = []
		if late or not exploring:
			for w in self._best_compatible():
				if w in offered and w not in chosen:
					# Someone who keeps turning us down goes behind new faces.
					(later if self._snubbed(w) else chosen).append(w)
		fresh.sort(key=lambda c: (-self._from_ad(c.ad), c.index))
		chosen += [c.index for c in fresh] + later
		return chosen[:limit]

	def _retarget(self) -> None:
		target = self._assign()
		if target != self.target:
			self.target, self.waited = target, 0
		elif target is not None:
			self.waited += 1
			if self.waited > PATIENCE:
				self.unresponsive[target] = self.round
				self.target, self.waited = self._assign(), 0

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		who = date.partner
		self.round = date.round
		self.ads[who] = date.partner_ad
		self.last_seen[who] = date.round
		if who not in self.shown:
			self.last_new = date.round
			for dim, (v, w) in enumerate(zip(observed, stated_ideal, strict=True)):
				self.counts[dim][v] += 1
				self.counts[dim][w] += 1
			self.costs = None
		self.shown.setdefault(who, []).append(tuple(observed))
		self.stated.setdefault(who, []).append(tuple(stated_ideal))

		decision = self._decide(date, who)
		self.asked[who] = decision
		return decision

	def _decide(self, date: DateContext, who: int) -> Decision:
		ours, their = self._with_me(who)
		if ours <= 0:
			# An outsider who is hopeless for us even taken at face value gets
			# blocked, so the engine stops spending our dates on them. Copies
			# are never blocked: they must stay visible for the shared matching.
			if self._optimistic(who) <= 0 and not self._kin(who):
				return Decision(incompatible=True)
			return Decision()

		if who == self.target:
			self.waited = 0
			# A copy of us waits until both have met everyone, so the matching
			# is built from the whole market rather than the first few dates.
			if self.exploring and self._kin(who):
				return Decision()
			return Decision(again=True, propose=date.date_number >= 2)

		# Someone who already proposed, anyone compatible once the matching has
		# nobody for us, or anyone compatible near the end.
		kin = self._kin(who)
		willing = their > 0 or who in self.proposed_to_me or not kin
		late = date.rounds_left <= ENDGAME or self.target is None and not self.exploring
		bar = self._bar(date.rounds_left)
		# Someone outside our matching will not wait for us: take them if they
		# beat both the partner the matching offers and what the rest of the
		# market is likely to offer.
		if not kin:
			mine = self._with_me(self.target)[0] if self.target in self.shown else 0
			if ours >= max(mine, bar):
				return Decision(again=True, propose=date.date_number >= 2)
		if late or who in self.proposed_to_me and ours >= bar:
			return Decision(again=willing, propose=willing and date.date_number >= 2)
		return Decision()

	def on_outcome(self, outcome: DateOutcome) -> None:
		who = outcome.partner
		if outcome.partner_proposed:
			self.proposed_to_me.add(who)
		asked = self.asked.pop(who, None)
		snubbed = asked is not None and (
			asked.propose and not outcome.partner_proposed or asked.again and not outcome.again
		)
		if snubbed and not outcome.married:
			self.snubs[who] = self.snubs.get(who, 0) + 1
