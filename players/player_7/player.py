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

1. Explore: list people we have not met, ranked by how well their ad fits us.
2. Commit: list only our assigned partner, ask for repeat dates, and propose
   from the second date on.
3. Fall back: if the assigned partner does not respond, or time is running
   out, propose to any compatible person, best first.

We never propose unless our own happiness would be positive. A partner whose
projected profile changes between dates or contradicts their ad is treated as
no better than a random person and is left out of the shared graph.
"""

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

VALUES = range(11)

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
		shown, stated = self.shown[who], self.stated[who]
		if any(p != shown[0] for p in shown) or any(s != stated[0] for s in stated):
			return False
		ad = self.ads.get(who)
		return ad is None or shown[0][ad.own_dim] == ad.own_value

	def _with_me(self, who: int) -> tuple[float, float]:
		"""(our happiness, their happiness) if we married ``who``. For someone we
		do not trust, every dimension costs at least what a random person costs."""
		mine, _ = self._expected_costs()
		trusted = self._trusted(who)
		ours = 0.0
		for dim in range(self.dims):
			worst = max(self._cost(dim, self.ideal[dim], p[dim]) for p in self.shown[who])
			ours += worst if trusted else max(worst, mine[dim])
		their = max(decrement(s, self.profile, self.monotonic) for s in self.stated[who])
		return self.bonus - ours - their / 2, self.bonus - their - ours / 2

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

	def _singles(self) -> list[int]:
		return [
			who
			for who in self.shown
			if self.round - self.last_seen.get(who, -GONE_AFTER) < GONE_AFTER and self._trusted(who)
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
			if self.round - self.unresponsive.get(p, -COOLDOWN) < COOLDOWN:
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

	def _best_compatible(self) -> list[int]:
		"""Singles we would be positive with, best for us first."""
		scored = []
		for who in self.shown:
			if self.round - self.last_seen.get(who, -GONE_AFTER) >= GONE_AFTER:
				continue
			ours, their = self._with_me(who)
			if ours > 0 and (their > 0 or who in self.proposed_to_me):
				scored.append((ours, who))
		return [who for _, who in sorted(scored, reverse=True)]

	# ------------------------------------------------------------------
	# Interface
	# ------------------------------------------------------------------

	def advertise(self) -> Advertisement:
		# Show the dimension where our true value most lowers a seeker's expected
		# cost, and ask for the one where a random partner would cost us most.
		def relief(dim: int) -> float:
			v = self.profile[dim]
			return sum(
				n * (self._cost(dim, w, 5) - self._cost(dim, w, v))
				for w, n in zip(VALUES, PRIOR, strict=True)
			)

		def risk(dim: int) -> float:
			return sum(
				n * self._cost(dim, self.ideal[dim], v) for v, n in zip(VALUES, PRIOR, strict=True)
			)

		own = max(range(self.dims), key=relief)
		seek = max(range(self.dims), key=risk)
		return Advertisement(
			own_dim=own, own_value=self.profile[own], seek_dim=seek, seek_value=self.ideal[seek]
		)

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
		if late or not exploring:
			chosen += [w for w in self._best_compatible() if w in offered and w not in chosen]
		fresh.sort(key=lambda c: (-self._from_ad(c.ad), c.index))
		chosen += [c.index for c in fresh]
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

		ours, their = self._with_me(who)
		if ours <= 0:
			return Decision()

		if who == self.target:
			self.waited = 0
			return Decision(again=True, propose=date.date_number >= 2)

		# Someone who already proposed, anyone compatible once the matching has
		# nobody for us, or anyone compatible near the end.
		willing = their > 0 or who in self.proposed_to_me
		late = date.rounds_left <= ENDGAME or self.target is None and not self.exploring
		if who in self.proposed_to_me or late:
			return Decision(again=willing, propose=willing and date.date_number >= 2)
		return Decision()

	def on_outcome(self, outcome: DateOutcome) -> None:
		if outcome.partner_proposed:
			self.proposed_to_me.add(outcome.partner)
