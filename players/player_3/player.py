"""Group 3 marriage-market player.

Toggle class attributes to ablate strategies (see evaluate.py). ``Player3`` runs
the full stack; ``Player3V1`` matches the original deliverable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
from models.profile import MAX_VALUE, Profile, decrement


@dataclass
class _PartnerState:
	observations: list[Profile] = field(default_factory=list)
	stated_ideals: list[Profile] = field(default_factory=list)
	trust: float = 1.0
	proposals_to: int = 0
	dates_without_reciprocate: int = 0


class _Player3Core(BasePlayer):
	"""Shared helpers and feature flags."""

	MIN_DATES = 2
	EXPLORE_FRACTION = 0.65
	RESERVATION_PERCENTILE = 0.70
	MIN_SAMPLES = 5

	# --- strategy toggles (S1–S8) ---
	USE_TRUST = True
	USE_MARKET_BAR = True
	USE_CONFIDENCE = True
	USE_ROUNDS_LEFT_COMMIT = True
	ENDGAME_ROUNDS = 3
	USE_COORDINATION = True
	MAX_ONE_SIDED_PROPOSALS = 2
	WILLING_EST_FLOOR = 0.05
	USE_STEER_LIST = True
	COMMIT_FRESH_CAP = 3
	USE_INCOMPATIBLE = True
	INCOMPATIBLE_EST = -5.0
	AD_SEEK_MODE = 'costly'  # 'v1' | 'costly' | 'low_mono'
	OWN_AD_INFLATE = 0
	LIE_MODE = 'honest'  # 'honest' | 'first_mono' | 'lean_partner'
	MONO_LIE_SHIFT = 1
	CONFIDENCE_SCALE = 25.0
	MONO_AD_SKEPTICISM = 0.35

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self._commit_round = max(1, int(self.rounds * self.EXPLORE_FRACTION))
		self._est: dict[int, float] = {}
		self._est_lb: dict[int, float] = {}
		self._seen: list[float] = []
		self._proposed_to_me: set[int] = set()
		self._mutual_again: dict[int, int] = {}
		self._partners: dict[int, _PartnerState] = {}
		self._dim_sum: list[float] = [0.0] * self.dims
		self._dim_count = 0
		self._last_projection: Profile | None = None
		self._ad: Advertisement | None = None
		self._last_singles = ctx.people

	def _partner(self, index: int) -> _PartnerState:
		if index not in self._partners:
			self._partners[index] = _PartnerState()
		return self._partners[index]

	def _dim_means(self) -> Profile:
		if self._dim_count == 0:
			return tuple(5 for _ in range(self.dims))
		return tuple(min(MAX_VALUE, max(0, int(round(s / self._dim_count)))) for s in self._dim_sum)

	def _observe_dims(self, observed: Profile) -> None:
		for i, v in enumerate(observed):
			self._dim_sum[i] += v
		self._dim_count += 1

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

	def _gap(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		if dim < self.monotonic and gap <= 0:
			return 0
		return gap * gap

	def _ad_trust_penalty(self, ad: Advertisement) -> float:
		if ad.own_dim < self.monotonic and ad.own_value >= 9:
			return 1.0 + self.MONO_AD_SKEPTICISM
		if ad.seek_dim < self.monotonic and ad.seek_value >= 9:
			return 1.0 + self.MONO_AD_SKEPTICISM * 0.5
		return 1.0

	def _ad_cost(self, ad: Advertisement) -> float:
		mine = self._gap(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		theirs = self._gap(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		cost = mine + theirs / 2
		if self.USE_STEER_LIST:
			cost *= self._ad_trust_penalty(ad)
			means = self._dim_means()
			for dim in range(self.dims):
				if dim in (ad.own_dim, ad.seek_dim):
					continue
				cost += 0.15 * self._gap(dim, self.ideal[dim], means[dim])
		return cost

	def _mirrored_ideal(self, stated_ideal: Profile, our_projection: Profile) -> bool:
		return all(a == b for a, b in zip(stated_ideal, our_projection, strict=True))

	def _adjust_profiles(
		self,
		partner: int,
		partner_ad: Advertisement,
		observed: Profile,
		stated_ideal: Profile,
		our_projection: Profile,
	) -> tuple[Profile, Profile, float]:
		if not self.USE_TRUST:
			return observed, stated_ideal, 1.0

		state = self._partner(partner)
		trust = state.trust

		if state.observations and observed != state.observations[-1]:
			trust *= 0.65
		if not state.observations and abs(observed[partner_ad.own_dim] - partner_ad.own_value) > 1:
			trust *= 0.55

		if self._mirrored_ideal(stated_ideal, our_projection):
			trust *= 0.4
			means = self._dim_means()
			stated_ideal = means

		trust = max(0.15, min(1.0, trust))
		state.trust = trust

		means = self._dim_means()
		adj_obs = tuple(
			int(round(trust * o + (1.0 - trust) * means[i])) for i, o in enumerate(observed)
		)
		adj_ideal = tuple(
			int(round(trust * s + (1.0 - trust) * means[i])) for i, s in enumerate(stated_ideal)
		)
		return adj_obs, adj_ideal, trust

	def _estimate(
		self,
		observed: Profile,
		stated_ideal: Profile,
		trust: float = 1.0,
	) -> float:
		mine = decrement(self.ideal, observed, self.monotonic)
		theirs = decrement(stated_ideal, self.profile, self.monotonic)
		return self.bonus - mine - theirs / 2

	def _confidence_lb(self, est: float, trust: float) -> float:
		if not self.USE_CONFIDENCE:
			return est
		width = (1.0 - trust) * self.CONFIDENCE_SCALE
		return est - width

	def _exploring(self, round_num: int, rounds_left: int) -> bool:
		if not self.USE_ROUNDS_LEFT_COMMIT:
			return round_num <= self._commit_round
		min_explore = max(3, int(self.rounds * 0.35))
		return rounds_left > min_explore

	def _reservation_v1(self, rounds_left: int) -> float:
		frac = max(0.0, min(1.0, (rounds_left - 1) / max(1, self.rounds - 1)))
		if len(self._seen) >= self.MIN_SAMPLES:
			base = self._percentile(self._seen, self.RESERVATION_PERCENTILE)
		else:
			base = 0.5 * self.bonus
		return max(0.0, base) * frac

	def _reservation(self, rounds_left: int, singles: int) -> float:
		if not self.USE_MARKET_BAR:
			return self._reservation_v1(rounds_left)

		frac = max(0.0, min(1.0, (rounds_left - 1) / max(1, self.rounds - 1)))
		if len(self._seen) >= self.MIN_SAMPLES:
			base = self._percentile(self._seen, self.RESERVATION_PERCENTILE)
		else:
			base = 0.5 * self.bonus

		market = singles / max(2, self.people)
		scarcity = 1.0 - market
		base *= 0.6 + 0.4 * scarcity
		return max(0.0, base) * frac

	def _seek_ad(self) -> tuple[int, int]:
		if self.AD_SEEK_MODE == 'v1':
			seek = max(range(self.dims), key=lambda i: abs(self.ideal[i] - 5))
			return seek, self.ideal[seek]
		if self.AD_SEEK_MODE == 'low_mono' and self.monotonic:
			seek = min(range(self.monotonic), key=lambda i: self.ideal[i])
			return seek, self.ideal[seek]
		# costly: dimension where unmet ideal vs population mean hurts most
		means = self._dim_means()
		seek = max(
			range(self.dims),
			key=lambda i: self._gap(i, self.ideal[i], means[i]),
		)
		return seek, self.ideal[seek]

	def _own_ad(self) -> tuple[int, int]:
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		value = min(MAX_VALUE, self.profile[own] + self.OWN_AD_INFLATE)
		return own, value

	def advertise(self) -> Advertisement:
		own_dim, own_value = self._own_ad()
		seek_dim, seek_value = self._seek_ad()
		self._ad = Advertisement(
			own_dim=own_dim,
			own_value=own_value,
			seek_dim=seek_dim,
			seek_value=seek_value,
		)
		return self._ad

	def project(self, date: DateContext) -> Profile:
		if self.LIE_MODE == 'honest':
			out = self.profile
		elif self.LIE_MODE == 'first_mono' and date.date_number == 1:
			out = tuple(
				min(MAX_VALUE, v + self.MONO_LIE_SHIFT) if i < self.monotonic else v
				for i, v in enumerate(self.profile)
			)
		else:
			out = self.profile
		self._last_projection = out
		return out

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		if self.LIE_MODE == 'lean_partner':
			blend = tuple(
				int(round(0.65 * self.ideal[i] + 0.35 * observed[i])) for i in range(self.dims)
			)
			return blend
		return self.ideal

	def _candidate_score(self, c: Candidate) -> float:
		est = self._est.get(c.index, 0.0)
		if est <= 0:
			return est
		score = est
		if self.USE_COORDINATION and self.USE_STEER_LIST:
			if c.index in self._proposed_to_me:
				score *= 1.4
			score *= 1.0 + 0.15 * self._mutual_again.get(c.index, 0)
		return score

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		if not candidates:
			return []
		self._last_singles = turn.singles
		exploring = self._exploring(turn.round, turn.rounds_left)

		known = sorted(
			(c for c in candidates if self._est.get(c.index, 0.0) > 0),
			key=lambda c: self._candidate_score(c),
			reverse=True,
		)
		fresh = sorted(
			(c for c in candidates if c.index not in self._est),
			key=lambda c: (self._ad_cost(c.ad), c.index),
		)
		order = [c.index for c in known]
		if exploring:
			order += [c.index for c in fresh]
		else:
			cap = self.COMMIT_FRESH_CAP if self.USE_STEER_LIST else 3
			order += [c.index for c in fresh[:cap]]

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
		our_projection = self._last_projection or self.profile
		state = self._partner(date.partner)
		state.observations.append(observed)
		state.stated_ideals.append(stated_ideal)
		self._observe_dims(observed)

		adj_obs, adj_ideal, trust = self._adjust_profiles(
			date.partner,
			date.partner_ad,
			observed,
			stated_ideal,
			our_projection,
		)
		est = self._estimate(adj_obs, adj_ideal, trust)
		lb = self._confidence_lb(est, trust)
		self._est[date.partner] = est
		self._est_lb[date.partner] = lb
		if self.USE_TRUST:
			self._seen.append(est)
		else:
			naive = self._estimate(observed, stated_ideal)
			self._est[date.partner] = naive
			lb = naive
			self._seen.append(naive)

		if lb <= 0 and est <= 0:
			incompat = self.USE_INCOMPATIBLE and lb <= self.INCOMPATIBLE_EST
			return Decision(again=False, propose=False, incompatible=incompat)

		bar = self._reservation(date.rounds_left, self._last_singles)

		willing = (
			date.partner in self._proposed_to_me or self._mutual_again.get(date.partner, 0) >= 2
		)
		if willing and lb < self.WILLING_EST_FLOOR:
			willing = False

		endgame = self.USE_MARKET_BAR and date.rounds_left <= self.ENDGAME_ROUNDS
		propose = date.date_number >= self.MIN_DATES and (
			(lb > bar or (willing and lb > 0)) or (endgame and lb > 0)
		)

		if self.USE_COORDINATION and date.partner in self._proposed_to_me and lb > 0:
			propose = propose or date.date_number >= self.MIN_DATES

		exploring = self._exploring(date.round, date.rounds_left)
		pursue = lb > max(bar, 0.5 * self.bonus) if exploring else lb > 0.0

		if self.USE_COORDINATION:
			ps = self._partner(date.partner)
			if propose:
				ps.proposals_to += 1
			if (
				ps.proposals_to >= self.MAX_ONE_SIDED_PROPOSALS
				and date.partner not in self._proposed_to_me
			):
				propose = False
				if not exploring:
					pursue = False

		incompat = False
		if self.USE_INCOMPATIBLE:
			if lb <= self.INCOMPATIBLE_EST or state.trust < 0.25 and lb < 0:
				incompat = True
			ps = self._partner(date.partner)
			if (
				ps.dates_without_reciprocate >= 3
				and date.partner not in self._proposed_to_me
				and not exploring
			):
				incompat = True

		if incompat:
			return Decision(again=False, propose=False, incompatible=True)

		return Decision(again=pursue or propose, propose=propose)

	def on_outcome(self, outcome: DateOutcome) -> None:
		if outcome.partner_proposed:
			self._proposed_to_me.add(outcome.partner)
			ps = self._partner(outcome.partner)
			ps.proposals_to = 0
		if outcome.again:
			self._mutual_again[outcome.partner] = self._mutual_again.get(outcome.partner, 0) + 1
			ps = self._partner(outcome.partner)
			ps.dates_without_reciprocate = 0
		elif not outcome.married:
			ps = self._partner(outcome.partner)
			ps.dates_without_reciprocate += 1


class Player3V1(_Player3Core):
	"""Original Group 3 deliverable (ablation baseline)."""

	USE_TRUST = False
	USE_MARKET_BAR = False
	USE_CONFIDENCE = False
	USE_ROUNDS_LEFT_COMMIT = False
	USE_COORDINATION = False
	USE_STEER_LIST = False
	USE_INCOMPATIBLE = False
	AD_SEEK_MODE = 'v1'
	OWN_AD_INFLATE = 0
	LIE_MODE = 'honest'

	def project(self, date: DateContext) -> Profile:
		return self.profile

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		return self.ideal

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		if not candidates:
			return []
		self._last_singles = turn.singles
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

		bar = self._reservation_v1(date.rounds_left)
		willing = (
			date.partner in self._proposed_to_me or self._mutual_again.get(date.partner, 0) >= 2
		)
		propose = date.date_number >= self.MIN_DATES and est > 0 and (est > bar or willing)

		exploring = date.round <= self._commit_round
		pursue = est > max(bar, 0.5 * self.bonus) if exploring else est > 0.0
		return Decision(again=pursue or propose, propose=propose)


class Player3(_Player3Core):
	"""Submitted player — flags tuned via ``tools/tune_defaults.py`` (18 seeds, 5 arenas)."""

	USE_TRUST = True
	USE_MARKET_BAR = True
	USE_CONFIDENCE = False
	USE_ROUNDS_LEFT_COMMIT = True
	USE_COORDINATION = True
	USE_STEER_LIST = False
	USE_INCOMPATIBLE = False
	LIE_MODE = 'honest'
	AD_SEEK_MODE = 'costly'
	OWN_AD_INFLATE = 0


class Player3AllOn(_Player3Core):
	"""Pre-tuning stack (all S1–S5 toggles on); kept for ablation comparison."""

	USE_TRUST = True
	USE_MARKET_BAR = True
	USE_CONFIDENCE = True
	USE_ROUNDS_LEFT_COMMIT = True
	USE_COORDINATION = True
	USE_STEER_LIST = True
	USE_INCOMPATIBLE = True
	LIE_MODE = 'honest'
	AD_SEEK_MODE = 'costly'


# Local ablation variants (evaluate harness only; not separate player_N dirs).
class Player3NoTrust(Player3):
	USE_TRUST = False


class Player3NoMarketBar(Player3):
	USE_MARKET_BAR = False


class Player3NoCoordination(Player3):
	USE_COORDINATION = False


class Player3WithSteer(Player3):
	USE_STEER_LIST = True


class Player3WithConfidence(Player3):
	USE_CONFIDENCE = True


class Player3WithIncompatible(Player3):
	USE_INCOMPATIBLE = True


class Player3LieFirstMono(Player3):
	LIE_MODE = 'first_mono'
