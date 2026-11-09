"""Group 9: truthful public signaling, coordinated matching, and adaptive search."""

import os
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from math import ceil, exp, sqrt
from random import Random

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


@dataclass
class CommitmentModel:
	"""Beta-smoothed response rates; these say nothing about true compatibility.

	Separate proposals after our recent invitation from unsolicited proposals.
	A decision made simultaneously with our first proposal cannot react to it.
	"""

	repeat_yes: int = 0
	repeat_no: int = 0
	proposal_yes: int = 0
	proposal_no: int = 0
	invited_yes: int = 0
	invited_no: int = 0

	def repeat_probability(self, population: 'CommitmentModel | None' = None) -> float:
		prior = population.repeat_probability() if population is not None else 0.5
		return (2 * prior + self.repeat_yes) / (2 + self.repeat_yes + self.repeat_no)

	def proposal_probability(
		self, invited: bool, population: 'CommitmentModel | None' = None
	) -> float:
		yes = self.invited_yes if invited else self.proposal_yes
		no = self.invited_no if invited else self.proposal_no
		prior = population.proposal_probability(invited) if population is not None else 0.4
		return (3 * prior + yes) / (3 + yes + no)

	def observe_repeat(self, agreed: bool) -> None:
		self.repeat_yes += int(agreed)
		self.repeat_no += int(not agreed)

	def observe_proposal(self, proposed: bool, invited: bool) -> None:
		if invited:
			self.invited_yes += int(proposed)
			self.invited_no += int(not proposed)
		else:
			self.proposal_yes += int(proposed)
			self.proposal_no += int(not proposed)


@dataclass
class Availability:
	"""Candidate counts exclude reserved repeats; singles counts do not.

	Between observations, forecast availability with a decaying estimate. The
	forecast is not a claim that particular people married or rejected us.
	"""

	count: int
	observed_round: int = 0
	retention: float = 0.9

	def observe(self, count: int, round_number: int) -> None:
		if self.observed_round and round_number > self.observed_round:
			gap = round_number - self.observed_round
			ratio = (max(1, count) / max(1, self.count)) ** (1 / gap)
			self.retention = 0.5 * self.retention + 0.5 * min(1.0, ratio)
		self.count = count
		self.observed_round = round_number

	def alternatives(self, round_number: int) -> float:
		age = max(0, round_number - self.observed_round)
		forecast = self.count * min(0.95, self.retention) ** age
		return max(0.0, forecast - 1)  # Exclude the person on our current date.


@dataclass(frozen=True)
class Estimate:
	"""Scenario estimates, not calibrated probabilities or verified happiness."""

	positive: float
	mutual_positive: float
	mean_happiness: float
	reported_happiness: float | None = None
	partner_reported_happiness: float | None = None


@dataclass
class PartnerHistory:
	"""Bounded report memory plus observable courtship outcomes."""

	profiles: list[Profile] = field(default_factory=list)
	ideals: list[Profile] = field(default_factory=list)
	profile_changes: int = 0
	ideal_changes: int = 0
	last_profile: Profile | None = None
	last_ideal: Profile | None = None
	dates: int = 0
	commitment: CommitmentModel = field(default_factory=CommitmentModel)
	last_proposal_round: int = 0
	last_own_proposal_round: int = 0
	cooldown_until: int = 0
	asked_again: bool = False
	invited: bool = False
	estimate: Estimate | None = None
	outside: float = 0.0  # estimated number of better-than-us options among singles


@dataclass
class Disclosures:
	"""Per-partner messages; no state or random streams are shared across people."""

	first_profile: Profile | None = None
	first_ideal: Profile | None = None
	first_sent_profile: Profile | None = None
	first_sent_ideal: Profile | None = None
	previous_profile: Profile | None = None
	previous_ideal: Profile | None = None
	current_profile: Profile | None = None
	current_ideal: Profile | None = None


def maximum_penalty(profile: Profile, ideal: Profile, monotonic: int) -> float:
	"""Largest own decrement plus half their decrement over every possible date."""
	mine = sum(
		want * want if i < monotonic else max(want * want, (10 - want) ** 2)
		for i, want in enumerate(ideal)
	)
	theirs = sum(
		(10 - have) ** 2 if i < monotonic else max(have * have, (10 - have) ** 2)
		for i, have in enumerate(profile)
	)
	return mine + theirs / 2


class CompatibilityModel:
	"""A fixed private scenario bank and a per-instance advertisement cache."""

	def __init__(
		self, profile: Profile, ideal: Profile, monotonic: int, bonus: float, rng: Random
	) -> None:
		self.profile = profile
		self.ideal = ideal
		self.monotonic = monotonic
		self.bonus = bonus
		self.dims = len(profile)
		self.ad_estimates: dict[Advertisement, Estimate] = {}
		# Only our own T and D are genuine population samples. Other people's
		# selected, potentially deceptive reports do not update this prior.
		weights = [[1.0] * 11 for _ in range(self.dims)]
		for dim in range(self.dims):
			weights[dim][self.profile[dim]] += 1
			weights[dim][self.ideal[dim]] += 1
		self.prior_own_costs: list[tuple[int, ...]] = []
		self.prior_partner_costs: list[tuple[int, ...]] = []
		for _ in range(64):
			sampled_profile = tuple(rng.choices(range(11), weights=row)[0] for row in weights)
			sampled_ideal = tuple(rng.choices(range(11), weights=row)[0] for row in weights)
			self.prior_own_costs.append(
				tuple(self.loss(i, self.ideal[i], sampled_profile[i]) for i in range(self.dims))
			)
			self.prior_partner_costs.append(
				tuple(self.loss(i, sampled_ideal[i], self.profile[i]) for i in range(self.dims))
			)
		self.prior_totals = [
			(sum(mine), sum(theirs))
			for mine, theirs in zip(self.prior_own_costs, self.prior_partner_costs, strict=True)
		]

		self.private_positive = sum(
			bonus > own + partner / 2 for own, partner in self.prior_totals
		) / len(self.prior_totals)

	def loss(self, dim: int, want: int, have: int) -> int:
		"""Squared mismatch, with no surplus penalty on monotonic dimensions."""
		gap = want - have
		return 0 if dim < self.monotonic and gap <= 0 else gap * gap

	def _summarize(self, scenarios: list[tuple[int, int, float]]) -> Estimate:
		positive = mutual = mean = 0.0
		for mine, theirs, weight in scenarios:
			ours = self.bonus - mine - theirs / 2
			theirs_happiness = self.bonus - theirs - mine / 2
			positive += weight * (ours > 0)
			mutual += weight * (ours > 0 and theirs_happiness > 0)
			mean += weight * ours
		return Estimate(min(1.0, positive), min(1.0, mutual), mean)

	def ad_estimate(self, ad: Advertisement) -> Estimate:
		"""Ads weakly condition the prior; unadvertised dimensions remain unknown."""
		if ad not in self.ad_estimates:
			mine_ad = self.loss(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
			theirs_ad = self.loss(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
			scenarios = []
			for k, (mine, theirs) in enumerate(self.prior_totals):
				# Each advertised half is used in half the scenarios. The four
				# combinations occur equally often; no advertisement is verified.
				if k % 2 == 0:
					mine += mine_ad - self.prior_own_costs[k][ad.own_dim]
				if k % 4 < 2:
					theirs += theirs_ad - self.prior_partner_costs[k][ad.seek_dim]
				scenarios.append((mine, theirs, 1 / len(self.prior_totals)))
			self.ad_estimates[ad] = self._summarize(scenarios)
		return self.ad_estimates[ad]

	def date_estimate(self, history: PartnerHistory, ad: Advertisement) -> Estimate:
		"""Mix plausible reports and latent profiles; never average trait vectors.

		Computing squared losses inside each scenario preserves uncertainty costs.
		Distinct reports share the report weight, so changing a story does not
		erase older evidence. The four cases allow either disclosure to be false.
		"""
		profile_trust = max(0.4, 0.9 - 0.15 * history.profile_changes)
		ideal_trust = max(0.4, 0.9 - 0.15 * history.ideal_changes)
		if any(profile[ad.own_dim] != ad.own_value for profile in history.profiles):
			profile_trust -= 0.1
		if any(ideal[ad.seek_dim] != ad.seek_value for ideal in history.ideals):
			ideal_trust -= 0.1
		own_reports = [decrement(self.ideal, p, self.monotonic) for p in history.profiles]
		partner_reports = [decrement(d, self.profile, self.monotonic) for d in history.ideals]
		scenarios = []
		count = len(self.prior_totals)
		for k, (unknown_mine, unknown_theirs) in enumerate(self.prior_totals):
			mine = own_reports[k % len(own_reports)]
			theirs = partner_reports[(k // len(own_reports)) % len(partner_reports)]
			for own_cost, own_weight in ((mine, profile_trust), (unknown_mine, 1 - profile_trust)):
				for partner_cost, partner_weight in (
					(theirs, ideal_trust),
					(unknown_theirs, 1 - ideal_trust),
				):
					scenarios.append((own_cost, partner_cost, own_weight * partner_weight / count))
		estimate = self._summarize(scenarios)
		mine = sum(own_reports) / len(own_reports)
		theirs = sum(partner_reports) / len(partner_reports)
		return Estimate(
			estimate.positive,
			estimate.mutual_positive,
			estimate.mean_happiness,
			self.bonus - mine - theirs / 2,
			self.bonus - theirs - mine / 2,
		)


# Independent-search defaults. P9_PARAMS="MULT=1.0,FLOOR=0.2" can override
# them for reproducible experiments; instances never mutate these parameters.
PARAMS = {
	'FLOOR': 0.35,  # aspiration floor as a fraction of B
	'REF': 0.75,  # fraction of the reference opportunity
	'PCT': 0.6,  # percentile of first-date happiness used as the reference
	'MULT': 1.3,  # overall bar multiplier
	'PERS_MIN': 0.7,  # pickiness scale floor
	'PERS_MAX': 1.2,  # pickiness scale cap (squared)
	'PERS_SCALE': 0.55,  # private positive rate that maps to scale 1
	'RES_BASE': 0.60,  # proposal probability threshold base
	'RES_SPAN': 0.20,  # extra threshold while alternatives remain
	'AGAIN_ALT': 0.5,  # alternatives level above which a repeat needs readiness
	'COOLDOWN': 6,  # rounds a declined partner is down-weighted (was 3)
}
for _item in os.environ.get('P9_PARAMS', '').split(','):
	if '=' in _item:
		_key, _value = _item.split('=', 1)
		PARAMS[_key.strip()] = float(_value)


# Ridge-logistic gate fitted to public advertisements. This score is not proof
# of player identity or cooperation. The cutoff rejected all 816 noncooperative
# calibration examples; runtime candidate loss cancels exploration immediately.
_COOPERATION_WEIGHTS = (
	6.055266389963511,
	-5.2803245020372716,
	-4.60549213339715,
	-6.283389838550684,
	7.172524225026243,
	8.108675817897518,
	-4.51378117645401,
	-6.329398405952754,
	1.0490739039129067,
	-6.312140165294354,
	5.067821310282944,
	6.524643357057741,
)
_COOPERATION_CUTOFF = 0.9711576022742818


def _cooperation_score(ads: list[Advertisement]) -> float:
	"""Score an initial public-ad market without accessing opponent identities."""
	n = len(ads)
	features = (
		1.0,
		sum(ad.seek_dim < 2 for ad in ads) / n,
		sum(ad.own_value for ad in ads) / (10 * n),
		sum(abs(ad.seek_value - 5) for ad in ads) / (5 * n),
		sum(abs(ad.own_value - 5) for ad in ads) / (5 * n),
		sum(ad.own_value == 10 for ad in ads) / n,
		sum(ad.seek_value in (0, 10) for ad in ads) / n,
		*(sum(ad.seek_dim == dim for ad in ads) / n for dim in range(5)),
	)
	logit = sum(w * x for w, x in zip(_COOPERATION_WEIGHTS, features, strict=True))
	return 1 / (1 + exp(-max(-35.0, min(35.0, logit))))


def _match_large_market(count: int, weights: dict[tuple[int, int], float]) -> list[int]:
	"""Maximum positive-pair cardinality, followed by bounded happiness swaps."""
	neighbours = [[] for _ in range(count)]
	for a, b in weights:
		neighbours[a].append(b)
		neighbours[b].append(a)
	for a in range(count):
		neighbours[a].sort(key=lambda b: (-weights[min(a, b), max(a, b)], b))
	mate = [-1] * count
	for a, b in sorted(weights, key=lambda edge: (-weights[edge], edge)):
		if mate[a] < 0 and mate[b] < 0:
			mate[a] = b
			mate[b] = a

	def augment(root):
		parent = [-1] * count
		base = list(range(count))
		used = [False] * count

		def lca(a, b):
			path = [False] * count
			while True:
				a = base[a]
				path[a] = True
				if mate[a] < 0:
					break
				a = parent[mate[a]]
			while not path[base[b]]:
				b = parent[mate[base[b]]]
			return base[b]

		def mark(a, b, common, blossom):
			while base[a] != common:
				blossom[base[a]] = blossom[base[mate[a]]] = True
				parent[a] = b
				b = mate[a]
				a = parent[mate[a]]

		queue = deque([root])
		used[root] = True
		while queue:
			a = queue.popleft()
			for b in neighbours[a]:
				if base[a] == base[b] or mate[a] == b:
					continue
				if b == root or (mate[b] >= 0 and parent[mate[b]] >= 0):
					common = lca(a, b)
					blossom = [False] * count
					mark(a, b, common, blossom)
					mark(b, a, common, blossom)
					for i in range(count):
						if blossom[base[i]]:
							base[i] = common
							if not used[i]:
								used[i] = True
								queue.append(i)
				elif parent[b] < 0:
					parent[b] = a
					if mate[b] < 0:
						while b >= 0:
							a = parent[b]
							previous = mate[a] if a >= 0 else -1
							mate[b] = a
							if a >= 0:
								mate[a] = b
							b = previous
						return True
					used[mate[b]] = True
					queue.append(mate[b])
		return False

	for a in range(count):
		if mate[a] < 0:
			augment(a)

	# Keep cardinality fixed. These local swaps improve, but do not globally
	# maximize, happiness. A fixed iteration bound limits callback time.
	def weight(a, b):
		return weights.get((min(a, b), max(a, b)))

	for _ in range(count):
		pairs = [(a, b) for a, b in enumerate(mate) if a < b]
		best = 0
		change = None
		for i, (a, b) in enumerate(pairs):
			old = weight(a, b)
			for c in range(count):
				if mate[c] < 0:
					for u, v in [(a, b), (b, a)]:
						w = weight(u, c)
						if w is not None and w - old > best:
							best = w - old
							change = [(u, c), (v, -1)]
			for c, d in pairs[i + 1 :]:
				old_both = old + weight(c, d)
				for u, v in [(c, d), (d, c)]:
					x, y = weight(a, u), weight(b, v)
					if x is not None and y is not None and x + y - old_both > best:
						best = x + y - old_both
						change = [(a, u), (b, v)]
		if change is None:
			break
		for a, b in change:
			mate[a] = b
			if b >= 0:
				mate[b] = a
	return mate


class Player9(BasePlayer):
	"""Choose promising dates, learn willingness, and adapt to our own prospects.

	The compatibility model uses private traits and uncertain reports. Courtship
	counts estimate willingness independently. The happiness bar falls as the
	market disappears and is scaled to 70--144% by private compatibility scarcity.
	When initial public ads all follow our truthful convention, use a common
	exploration deadline and cardinality-first, happiness-second matching.
	The supported N50 mode uses a fitted public-ad gate and polynomial matching.
	Otherwise use independent reciprocal search immediately.
	We display our true profile and ideal; partners who use a decodable
	first-date swap or who mirror our messages back are still recognised.
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.history: dict[int, PartnerHistory] = {}
		self.disclosures: dict[int, Disclosures] = {}
		self.opportunities: dict[int, float] = {}
		self.availability = Availability(self.people - 1)
		self.available_ids: set[int] = set()
		self.population_commitment = CommitmentModel()
		self.last_requests: set[int] = set()
		self.request_round = 0
		self.request_trials = 0
		self.request_hits = 0
		self.fresh_positive = 0.5
		self.beliefs = CompatibilityModel(
			self.profile, self.ideal, self.monotonic, self.bonus, self.rng
		)
		self.guaranteed_positive = self.bonus > maximum_penalty(
			self.profile, self.ideal, self.monotonic
		)
		# Coordinate only in the market size and time range this mode supports.
		self.coordination_eligible = (
			8 <= self.people <= 22
			and self.monotonic >= 2
			and self.rounds >= 2 * self.people + 6
			and self.bonus > 0
		)
		# The statistical gate was evaluated only in this parameter domain.
		self.statistical_coordination = (
			self.people == 50
			and self.dims == 5
			and self.monotonic == 2
			and self.bonus == 150
			and self.rounds == 250
		)
		self.coordination_eligible |= self.statistical_coordination
		self.coordinating: bool | None = None
		self.exploring = False
		self.target: int | None = None
		self.target_chosen = False

	def advertise(self) -> Advertisement:
		"""Disclose a strength and costly preference, with the small-market convention."""
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(
			showable,
			key=lambda i: self.profile[i] if i < self.monotonic else abs(self.profile[i] - 5),
		)
		seek = max(
			range(self.dims),
			key=lambda i: sum(cost[i] for cost in self.beliefs.prior_own_costs),
		)
		if self.coordination_eligible and not self.statistical_coordination:
			# A public, truthful convention; matching is still based on dates.
			own = 1
		return Advertisement(own, self.profile[own], seek, self.ideal[seek])

	def _choose_independently(
		self, candidates: tuple[Candidate, ...], turn: RoundContext
	) -> list[int]:
		"""Rank candidates for inclusion; the engine ignores list order."""
		self.availability.observe(len(candidates), turn.round)
		self.available_ids = {candidate.index for candidate in candidates}
		self._refresh_market()
		if self.bonus <= 0:
			return []
		ranked = []
		for candidate in candidates:
			if turn.rounds_left == 1 and candidate.dates == 0:
				continue
			history = self.history.get(candidate.index)
			estimate = (
				history.estimate
				if history and history.estimate
				else self.beliefs.ad_estimate(candidate.ad)
			)
			quality = max(-1.0, min(1.0, estimate.mean_happiness / max(1, self.bonus)))
			# A preference-driven partner needs a favorable outcome too. Once
			# presentation is possible, true mutual compatibility is less decisive
			# for acceptance; our own happiness requirement remains unchanged.
			weight = 0.35
			if candidate.dates >= 1:
				weight = 0.0
			compatible = (1 - weight) * estimate.positive + weight * estimate.mutual_positive
			# Willingness is learned only after dates and ranking by it chased
			# willing-but-mediocre partners; happiness is higher without it.
			score = compatible * (1 + 0.5 * quality)
			if history and turn.round <= history.cooldown_until and turn.rounds_left > 2:
				score *= 0.25
			ranked.append((score, self.rng.random(), candidate))
		ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
		miss_rate = 1 - self.request_hits / self.request_trials if self.request_trials else 0
		width = min(len(ranked), max(2, ceil(1.5 * sqrt(len(ranked)) * (1 + 0.5 * miss_rate))))
		if self.list_limit is not None:
			width = min(width, self.list_limit)
		selected = [candidate.index for _, _, candidate in ranked[:width]]
		self.last_requests = set(selected)
		self.request_round = turn.round
		self.request_trials += int(bool(selected))
		fresh = [
			self.beliefs.ad_estimate(c.ad).positive
			for _, _, c in ranked[:width]
			if c.index not in self.history
		]
		if fresh:
			self.fresh_positive = sum(fresh) / len(fresh)
		return selected

	def project(self, date: DateContext) -> Profile:
		"""Show the true profile. Flattering displays won more marriages but
		worse ones; mean happiness is higher when partners decide on the truth."""
		messages = self._messages(date.partner)
		claim = self.profile
		messages.current_profile = claim
		if messages.first_sent_profile is None:
			messages.first_sent_profile = claim
		return claim

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		"""Describe the true ideal, on every date."""
		messages = self._messages(date.partner)
		claim = self.ideal
		messages.current_ideal = claim
		if messages.first_sent_ideal is None:
			messages.first_sent_ideal = claim
		return claim

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		profile_evidence, ideal_evidence = self._date_evidence(date, observed, stated_ideal)
		decision = self._assess_date(date, profile_evidence, ideal_evidence)
		assigned_courtship = self.coordinating and (self.exploring or self.target is not None)
		if self.guaranteed_positive and not assigned_courtship:
			# Preserve ordinary evidence and cooldown updates even when our private
			# traits prove every possible marriage strictly positive.
			history = self.history[date.partner]
			continuation = self._commitment_chance(
				history, date.date_number, date.rounds_left - 1, date.round + 1
			)
			search = self._search_value(date.rounds_left - 1, date.round + 1, date.partner)
			again = date.rounds_left > 1 and (
				self.availability.alternatives(date.round) == 0 or continuation >= search
			)
			history.asked_again = again
			propose = date.date_number >= 2
			if propose:
				history.last_own_proposal_round = date.round
			decision = Decision(again=again, propose=propose)
		messages = self._messages(date.partner)
		messages.previous_profile = messages.current_profile
		messages.previous_ideal = messages.current_ideal
		return decision

	def on_outcome(self, outcome: DateOutcome) -> None:
		"""Learn willingness after both people have made their decisions."""
		history = self.history[outcome.partner]
		model = history.commitment
		if outcome.date_number >= 2:
			self.population_commitment.observe_proposal(outcome.partner_proposed, history.invited)
			model.observe_proposal(outcome.partner_proposed, history.invited)
		if outcome.partner_proposed:
			history.last_proposal_round = outcome.round
		if outcome.married:
			return
		if history.asked_again:
			self.population_commitment.observe_repeat(outcome.again)
			model.observe_repeat(outcome.again)

	def _outside_options(self, partner: int) -> float:
		"""How many singles we know would make `partner` happier than we would.

		Uses only our own date evidence (exact profiles and ideals of people we
		dated); scaled up by the share of singles we have not dated.
		"""
		history = self.history.get(partner)
		if not history or history.last_profile is None or history.estimate is None:
			return 0.0
		with_us = history.estimate.partner_reported_happiness or 0.0
		known = rivals = 0
		for other_id in self.available_ids:
			if other_id == partner:
				continue
			other = self.history.get(other_id)
			if not other or other.last_profile is None:
				continue
			known += 1
			cost_j = decrement(history.last_ideal, other.last_profile, self.monotonic)
			cost_k = decrement(other.last_ideal, history.last_profile, self.monotonic)
			happy_j = self.bonus - cost_j - cost_k / 2
			happy_k = self.bonus - cost_k - cost_j / 2
			if happy_j > with_us and happy_k > 0:
				rivals += 1
		if not known:
			return 0.0
		return rivals * len(self.available_ids - {partner}) / known

	def _refresh_market(self) -> None:
		for partner, history in self.history.items():
			if partner in self.available_ids:
				history.outside = self._outside_options(partner)

	def _messages(self, partner: int) -> Disclosures:
		return self.disclosures.setdefault(partner, Disclosures())

	def _date_evidence(
		self, date: DateContext, observed: Profile, stated_ideal: Profile
	) -> tuple[Profile, Profile]:
		"""Interpret the display convention and keep copied messages as old evidence.

		A coherent advertisement permits the inverse monotonic swap. This is a
		signaling hypothesis, not verification of the partner's true traits.
		"""
		messages = self._messages(date.partner)
		ad = date.partner_ad
		if (
			messages.first_profile is None
			and ad.own_dim < self.monotonic
			and ad.own_value < 10
			and observed[ad.own_dim] == 10
		):
			decoded = list(observed)
			for dim in range(self.monotonic):
				if decoded[dim] == 10:
					decoded[dim] = ad.own_value
				elif decoded[dim] == ad.own_value:
					decoded[dim] = 10
			if (
				decoded[ad.own_dim] == ad.own_value
				and max(decoded[: self.monotonic]) == ad.own_value
			):
				observed = tuple(decoded)
		if messages.first_profile is None:
			messages.first_profile, messages.first_ideal = observed, stated_ideal
		if date.date_number >= 2:
			if observed in (messages.previous_ideal, messages.first_sent_ideal, self.ideal):
				observed = messages.first_profile
			if stated_ideal in (
				messages.previous_profile,
				messages.first_sent_profile,
				self.profile,
			):
				stated_ideal = messages.first_ideal
		return observed, stated_ideal

	def _aspiration(self, rounds_left: int, round_number: int, partner: int) -> float:
		"""A personal reservation value that relaxes with scarcity and time."""
		values = sorted(value for index, value in self.opportunities.items() if index != partner)
		reference = values[int(PARAMS['PCT'] * (len(values) - 1))] if values else 0.0
		threshold = max(PARAMS['FLOOR'] * self.bonus, PARAMS['REF'] * max(0.0, reference))
		fraction_left = (rounds_left - 1) / max(1, self.rounds - 1)
		promising = self.availability.alternatives(round_number) * self.fresh_positive
		breadth = min(1.0, promising / max(4, sqrt(self.people)))
		urgency = min(1.0, self.availability.retention / 0.9)
		bar = PARAMS['MULT'] * (threshold * fraction_left * breadth * urgency)
		personal = (
			min(PARAMS['PERS_MAX'], self.beliefs.private_positive / PARAMS['PERS_SCALE']) ** 2
		)
		return bar * max(PARAMS['PERS_MIN'], personal)

	def _commitment_chance(
		self, history: PartnerHistory | None, dates: int, rounds_left: int, round_number: int
	) -> float:
		"""Chance of commitment over at most six dates if we pursue this person.

		Both repeat agreement and a legal, simultaneous proposal are required.
		The geometric calculation is a decision heuristic, not a calibrated model.
		"""
		model = history.commitment if history else CommitmentModel()
		repeat = model.repeat_probability(self.population_commitment)
		alive, success = 1.0, 0.0
		for offset in range(min(6, rounds_left)):
			if dates + offset + 1 >= 2:
				recent_invitation = bool(
					history
					and history.last_own_proposal_round
					and round_number + offset - history.last_own_proposal_round <= 2
				)
				invited = recent_invitation or dates + offset >= 2
				proposal = model.proposal_probability(invited, self.population_commitment)
				# A partner proposes in proportion to how happy they said they
				# would be with us, not only to how often they said yes before.
				if (
					history
					and history.estimate
					and history.estimate.partner_reported_happiness is not None
				):
					fit = history.estimate.partner_reported_happiness / max(1.0, self.bonus)
					proposal *= max(0.1, min(1.0, 0.5 + fit))
				if history and history.last_proposal_round:
					age = round_number + offset - history.last_proposal_round
					proposal = max(proposal, 0.95 * 0.5 ** max(0, age - 1))
				success += alive * proposal
				alive *= 1 - proposal
			alive *= repeat
		return min(1.0, success)

	def _search_value(self, rounds_left: int, round_number: int, partner: int) -> float:
		"""Estimate success probability from fresh and known search alternatives."""
		alternatives = self.availability.alternatives(round_number)
		attempts = min(alternatives, max(0, rounds_left) // 2, 3)
		# Each fresh attempt consumes at least two dates. Do not count a
		# three-date proposal chance as if it cost only two rounds.
		new_commitment = self._commitment_chance(None, 0, min(2, rounds_left), round_number)
		per_attempt = self.fresh_positive * new_commitment
		value = 1 - (1 - per_attempt) ** attempts
		age = max(0, round_number - self.availability.observed_round)
		for candidate in self.available_ids - {partner}:
			history = self.history.get(candidate)
			if history and history.estimate and history.estimate.reported_happiness > 0:
				known = history.estimate.positive * self._commitment_chance(
					history, history.dates, rounds_left, round_number
				)
				known /= 1 + 1.0 * history.outside
				value = max(value, known * min(0.95, self.availability.retention) ** age)
		return value

	def _assess_independently(
		self, date: DateContext, observed: Profile, stated_ideal: Profile
	) -> Decision:
		"""Record evidence, then decide whether to repeat, propose, or block."""
		if date.round == self.request_round and date.partner in self.last_requests:
			self.request_hits += 1
		aspiration = self._aspiration(date.rounds_left, date.round, date.partner)
		history = self.history.setdefault(date.partner, PartnerHistory())
		history.invited = bool(
			history.last_own_proposal_round and date.round - history.last_own_proposal_round <= 2
		)
		history.profile_changes += int(
			history.last_profile is not None and history.last_profile != observed
		)
		history.ideal_changes += int(
			history.last_ideal is not None and history.last_ideal != stated_ideal
		)
		history.last_profile, history.last_ideal = observed, stated_ideal
		for reports, report in ((history.profiles, observed), (history.ideals, stated_ideal)):
			if report not in reports:
				reports.append(report)
				del reports[:-4]
		history.dates += 1
		estimate = self.beliefs.date_estimate(history, date.partner_ad)
		history.estimate = estimate
		history.outside = self._outside_options(date.partner)
		if history.dates == 1:
			self.opportunities[date.partner] = estimate.reported_happiness
			if len(self.opportunities) > 128:
				del self.opportunities[next(iter(self.opportunities))]

		# Merely omitting a candidate cannot stop random fallback dates. Block
		# a strongly unfavorable, uncontradicted report while search time remains.
		# This is a model-based rejection, not proof of their true incompatibility.
		if (
			self.bonus > 0
			and date.rounds_left > 2
			and self.availability.alternatives(date.round) > 1
			and estimate.reported_happiness <= 0
			and estimate.positive < 0.2
			and history.profile_changes == 0
			and history.ideal_changes == 0
		):
			history.asked_again = False
			return Decision(incompatible=True)

		fraction_left = (date.rounds_left - 1) / max(1, self.rounds - 1)
		available = self.availability.alternatives(date.round)
		alternatives = min(1.0, available * self.fresh_positive / max(4, sqrt(self.people)))
		reservation = PARAMS['RES_BASE'] + PARAMS['RES_SPAN'] * fraction_left * alternatives
		ready = estimate.positive >= reservation and estimate.reported_happiness > aspiration
		if history.last_proposal_round and date.round - history.last_proposal_round <= 2:
			# Willingness can justify ending the search, but does not establish
			# compatibility or remove the probability threshold.
			ready = estimate.positive >= reservation and estimate.reported_happiness > 0
		if self.bonus > 150 * self.dims:
			ready = True  # Every possible marriage is strictly positive.
		if date.rounds_left == 1:
			# Remove the search threshold, but require favorable dating evidence.
			# A tiny positive tail from the broad prior alone is not calibrated
			# evidence for marrying someone whose reports imply a negative score.
			ready = estimate.reported_happiness > 0
		propose = bool(self.bonus > 0 and date.date_number >= 2 and ready)
		# Compare conditional on a repeat being arranged. If they decline,
		# we can search next turn anyway; asking does not forfeit that option.
		continuation = estimate.positive * self._commitment_chance(
			history, date.date_number, date.rounds_left - 1, date.round + 1
		)
		search = self._search_value(date.rounds_left - 1, date.round + 1, date.partner)
		worth_retaining = available == 0 or continuation >= search
		again = bool(
			self.bonus > 0
			and date.rounds_left > 1
			and estimate.positive >= 0.5
			and estimate.reported_happiness > 0
			# While most of the game remains, a repeat date must be with an excellent
			# match; otherwise the round is better spent meeting someone new.
			and (ready or alternatives < PARAMS['AGAIN_ALT'])
			and worth_retaining
		)
		if not again:
			history.cooldown_until = date.round + int(PARAMS['COOLDOWN'])
		history.asked_again = again
		if propose:
			history.last_own_proposal_round = date.round
		# Ambiguous reports and stalled courtships receive reversible cooldowns.
		return Decision(again=again, propose=propose)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		"""Coordinate under a tested public gate, otherwise search independently."""
		if self.coordinating is None:
			if self.statistical_coordination:
				# Include our own unchanged advertisement so peers see identical inputs.
				ads = [self.advertise(), *(candidate.ad for candidate in candidates)]
				self.coordinating = (
					len(ads) == self.people
					and all(ad.own_dim < self.monotonic for ad in ads)
					and _cooperation_score(ads) >= _COOPERATION_CUTOFF
				)
			else:
				self.coordinating = (
					self.coordination_eligible
					and bool(candidates)
					and all(candidate.ad.own_dim == 1 for candidate in candidates)
				)
		if (
			self.statistical_coordination
			and turn.round < min(2 * self.people, self.rounds // 2)
			and len(candidates) < self.people - 1
		):
			# Reserved repeats, blocks or marriages contradict shared exploration.
			# Candidate loss need not establish which event occurred.
			self.coordinating = False
		if not self.coordinating:
			return self._choose_independently(candidates, turn)

		self.availability.observe(len(candidates), turn.round)
		self.available_ids = {candidate.index for candidate in candidates}
		self.exploring = (
			turn.round < min(2 * self.people, self.rounds // 2) and turn.rounds_left > 6
		)
		if self.exploring:
			fresh = [candidate for candidate in candidates if candidate.index not in self.history]
			fresh.sort(
				key=lambda candidate: self.beliefs.ad_estimate(candidate.ad).mean_happiness,
				reverse=True,
			)
			width = min(6, self.list_limit if self.list_limit is not None else len(fresh))
			selected = [candidate.index for candidate in fresh[:width]]
		else:
			if not self.target_chosen:
				self.target = self._matching_target()
				self.target_chosen = True
			selected = (
				[self.target]
				if self.target in self.available_ids
				else self._choose_independently(candidates, turn)
			)
		self.last_requests = set(selected)
		return selected

	def _matching_target(self) -> int | None:
		"""Construct positive matches from personal dating evidence.

		Small markets optimize cardinality and happiness exactly. The larger-market
		solver maximizes cardinality but only locally refines happiness.

		The fixed exploration deadline helps peers build comparable graphs. Each
		person solves their own graph; no state is shared between instances.
		The graph is an estimate: stable disclosures are not proof of honesty.
		"""
		people = {
			index: (history.last_profile, history.last_ideal)
			for index, history in self.history.items()
			if index in self.available_ids
			and history.last_profile is not None
			and history.profile_changes == 0
			and history.ideal_changes == 0
		}
		people[self.index] = (self.profile, self.ideal)
		indices = sorted(people)
		count = len(indices)
		# One extra positive pair outweighs any possible happiness difference.
		reward = (count + 1) * max(1, self.bonus)
		weights = {}
		for first, a in enumerate(indices):
			profile_a, ideal_a = people[a]
			for second in range(first + 1, count):
				profile_b, ideal_b = people[indices[second]]
				cost_a = decrement(ideal_a, profile_b, self.monotonic)
				cost_b = decrement(ideal_b, profile_a, self.monotonic)
				happy_a = self.bonus - cost_a - cost_b / 2
				happy_b = self.bonus - cost_b - cost_a / 2
				if happy_a > 0 and happy_b > 0:
					weights[first, second] = reward + happy_a + happy_b

		if self.statistical_coordination:
			mate = _match_large_market(count, weights)
			partner = mate[indices.index(self.index)]
			return indices[partner] if partner >= 0 else None

		choices: dict[int, int | None] = {}

		@lru_cache(None)
		def solve(mask: int) -> float:
			if not mask:
				return 0.0
			bit = mask & -mask
			first = bit.bit_length() - 1
			rest = mask ^ bit
			best, choice = solve(rest), None
			remaining = rest
			while remaining:
				second_bit = remaining & -remaining
				second = second_bit.bit_length() - 1
				remaining ^= second_bit
				weight = weights.get((first, second))
				if weight is None:
					continue
				value = weight + solve(rest ^ second_bit)
				if value > best:
					best, choice = value, second
			choices[mask] = choice
			return best

		mask = (1 << count) - 1
		solve(mask)
		target = None
		while mask:
			bit = mask & -mask
			first = bit.bit_length() - 1
			second = choices[mask]
			mask ^= bit
			if second is not None:
				if indices[first] == self.index:
					target = indices[second]
					break
				if indices[second] == self.index:
					target = indices[first]
					break
				mask ^= 1 << second
		solve.cache_clear()
		return target

	def _assess_date(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		decision = self._assess_independently(date, observed, stated_ideal)
		if not self.coordinating:
			return decision
		history = self.history[date.partner]
		if self.exploring:
			# Blocking would hide vertices from our graph and spoil agreement.
			history.asked_again = False
			return Decision()
		estimate = history.estimate
		if (
			date.partner == self.target
			and estimate.reported_happiness > 0
			and estimate.partner_reported_happiness > 0
		):
			history.asked_again = True
			return Decision(again=date.rounds_left > 1, propose=date.date_number >= 2)
		return decision
