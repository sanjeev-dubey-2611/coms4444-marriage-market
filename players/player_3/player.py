"""Group 3 marriage-market player.

Self-contained coordination and controlled-deception strategy. ``Player3V1``
preserves the original deliverable for comparison.
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


class Coordination:
	ADAPTIVE_REPEAT = False
	SCARCITY_RESERVATION = False
	FOCUS_OFFERS = False
	REPEAT_QUANTILE = 0.75
	REPEAT_MIN_PARTNERS = 3
	CANDIDATE_CAP = None
	STOP_AFTER_OFFERS = None
	BILATERAL_REPEAT = False

	def _reservation(self, rounds_left, singles):
		bar = super()._reservation(rounds_left, singles)
		if self.SCARCITY_RESERVATION:
			bar *= min(1.0, singles / max(2, self.people))
		return bar

	def _coordination_safe(self, partner):
		probabilities = getattr(self, '_positive', None)
		return probabilities is None or probabilities.get(partner, 0.0) >= 0.9

	def decide(self, date, observed, stated_ideal):
		decision = super().decide(date, observed, stated_ideal)
		if (
			self.ADAPTIVE_REPEAT
			and not decision.again
			and not decision.incompatible
			and date.rounds_left > 1
			and len(self._est) >= self.REPEAT_MIN_PARTNERS
			and self._coordination_safe(date.partner)
		):
			personal_floor = min(
				0.5 * self.bonus, self._percentile(list(self._est.values()), self.REPEAT_QUANTILE)
			)
			bar = max(0.0, personal_floor, self._reservation(date.rounds_left, self._last_singles))
			if self._est[date.partner] > bar and (
				self._partner(date.partner).proposals_to < self.MAX_ONE_SIDED_PROPOSALS
				or date.partner in self._proposed_to_me
			):
				decision = Decision(again=True, propose=decision.propose)
		if (
			self.BILATERAL_REPEAT
			and not decision.again
			and date.rounds_left > 1
			and self._partner(date.partner).trust >= 0.75
			and self._coordination_safe(date.partner)
		):
			theirs = (
				self.bonus
				- decrement(stated_ideal, self.profile, self.monotonic)
				- decrement(self.ideal, observed, self.monotonic) / 2
			)
			if min(self._est[date.partner], theirs) > max(
				0.35 * self.bonus, self._reservation(date.rounds_left, self._last_singles)
			):
				decision = Decision(again=True, propose=decision.propose)
		if (
			self.STOP_AFTER_OFFERS is not None
			and self._partner(date.partner).proposals_to >= self.STOP_AFTER_OFFERS
			and date.partner not in self._proposed_to_me
		):
			# Still offer on this date: their proposal can arrive simultaneously.
			# Stop reserving every following round for an unresponsive partner.
			decision = Decision(again=False, propose=decision.propose)
		return decision

	def choose_dates(self, candidates, turn):
		chosen = super().choose_dates(candidates, turn)
		if self.FOCUS_OFFERS:
			offers = [
				c
				for c in candidates
				if c.index in self._proposed_to_me
				and self._est.get(c.index, 0) > 0
				and self._coordination_safe(c.index)
			]
			if offers:
				# Matching consumes a set, so sorting an otherwise unchanged list cannot prioritize an offer.
				best = max(offers, key=lambda c: (self._est[c.index], -c.index))
				return [best.index]
		if self.STOP_AFTER_OFFERS is not None:
			chosen = [
				i
				for i in chosen
				if i in self._proposed_to_me
				or i not in self._partners
				or self._partners[i].proposals_to < self.STOP_AFTER_OFFERS
			]
		return chosen[: self.CANDIDATE_CAP]


DECEPTION_DEFAULTS = dict(
	ad_inflate=0,
	ad_ceiling=10,
	ad_min_true=0,
	ad_min_bonus_per_dim=0,
	ad_own='original',
	ad_seek_shift=0,
	ad_seek='original',
	consistent_ad=True,
	project_mode='honest',
	project_step=2,
	stable_profile=True,
	preference_blend=0.0,
	preference_step=10,
	stable_ideal=True,
	avoid_mirror=True,
	adaptive='none',
	compatibility_floor=0.1,
	late_fraction=0.6,
	guard='none',
	guard_margin=0.0,
	guard_shift=2,
	guard_min_ad=0,
	guard_radius=1,
	score_debias=0,
)


class Deception:
	"""Separate public messages from our real private utility calculation."""

	SETTINGS = DECEPTION_DEFAULTS

	def __init__(self, snapshot, ctx):
		super().__init__(snapshot, ctx)
		self._sent_profiles = {}
		self._sent_ideals = {}

	def advertise(self):
		ad = super().advertise()
		s = self.SETTINGS
		own, value = ad.own_dim, ad.own_value
		if s['ad_own'] == 'nonmono' and self.monotonic < self.dims:
			own = min(range(self.monotonic, self.dims), key=lambda i: abs(self.profile[i] - 5))
			value = self.profile[own]
		elif s['ad_own'] == 'balanced':
			own = min(
				range(self.dims),
				key=lambda i: sum(self._gap(i, w, self.profile[i]) for w in range(11)),
			)
			value = self.profile[own]
		elif s['ad_own'] == 'weak_mono' and self.monotonic:
			own = min(range(self.monotonic), key=lambda i: self.profile[i])
			value = self.profile[own]
		if (
			own < self.monotonic
			and value >= s['ad_min_true']
			and self.bonus / self.dims >= s['ad_min_bonus_per_dim']
		):
			value = max(value, min(s['ad_ceiling'], value + s['ad_inflate']))
		seek, want = ad.seek_dim, ad.seek_value
		if s['ad_seek'] == 'low_mono' and self.monotonic:
			seek = min(range(self.monotonic), key=lambda i: self.ideal[i])
			want = self.ideal[seek]
		shift = s['ad_seek_shift']
		if seek < self.monotonic:
			want = max(0, want - shift)
		else:
			want += max(-shift, min(shift, 5 - want))
		self._ad = Advertisement(own, value, seek, want)
		return self._ad

	def _lie_allowed(self, date, observed=None):
		s = self.SETTINGS
		mode = s['adaptive']
		if mode == 'none':
			return True
		state = self._partners.get(date.partner)
		if mode == 'late' and date.round < s['late_fraction'] * self.rounds:
			return False
		if mode == 'reluctant' and (date.date_number < 2 or date.partner in self._proposed_to_me):
			return False
		if state and state.trust < 0.75:
			return False
		if observed is None:
			if state and state.observations:
				observed = state.observations[-1]
			else:
				values = list(self._dim_means())
				values[date.partner_ad.own_dim] = date.partner_ad.own_value
				observed = tuple(values)
		ideal = state.stated_ideals[0] if state and state.stated_ideals else self._dim_means()
		if self._estimate(observed, ideal) <= s['compatibility_floor'] * self.bonus:
			return False
		if mode == 'marginal':
			theirs = (
				self.bonus
				- decrement(ideal, self.profile, self.monotonic)
				- decrement(self.ideal, observed, self.monotonic) / 2
			)
			return theirs < 0.6 * self.bonus
		return True

	def project(self, date):
		s = self.SETTINGS
		if s['stable_profile'] and date.partner in self._sent_profiles:
			out = self._sent_profiles[date.partner]
		else:
			values = list(self.profile)
			mode, step = s['project_mode'], s['project_step']
			if self._lie_allowed(date):
				if mode in ('mono', 'first_mono') and (
					mode != 'first_mono' or date.date_number == 1
				):
					for i in range(self.monotonic):
						values[i] = min(10, values[i] + step)
				elif mode == 'partner':
					state = self._partners.get(date.partner)
					if state and state.stated_ideals:
						wanted = state.stated_ideals[-1]
					else:
						wanted = list(self.profile)
						wanted[date.partner_ad.seek_dim] = date.partner_ad.seek_value
					for i, want in enumerate(wanted):
						if i < self.monotonic:
							want = max(want, values[i])
						values[i] += max(-step, min(step, want - values[i]))
			if s['consistent_ad'] and self._ad:
				values[self._ad.own_dim] = self._ad.own_value
			out = tuple(values)
			self._sent_profiles[date.partner] = out
		self._last_projection = out
		return out

	def describe_ideal(self, date, observed):
		s = self.SETTINGS
		if s['stable_ideal'] and date.partner in self._sent_ideals:
			return self._sent_ideals[date.partner]
		values = list(self.ideal)
		blend = s['preference_blend']
		if blend and self._lie_allowed(date, observed):
			for i, have in enumerate(observed):
				target = min(have, values[i]) if i < self.monotonic else have
				shift = round((1 - blend) * values[i] + blend * target) - values[i]
				values[i] += max(-s['preference_step'], min(s['preference_step'], shift))
		if s['consistent_ad'] and self._ad:
			values[self._ad.seek_dim] = self._ad.seek_value
		# Only alter a deceptive statement here; a no-op policy must be exactly honest.
		if s['avoid_mirror'] and tuple(values) != self.ideal and tuple(values) == observed:
			editable = [
				i
				for i in range(self.dims)
				if not (s['consistent_ad'] and self._ad and i == self._ad.seek_dim)
			]
			if editable:
				i = min(editable, key=lambda j: abs(self.ideal[j] - values[j]))
				values[i] += -1 if values[i] > 0 else 1
		out = tuple(values)
		self._sent_ideals[date.partner] = out
		return out

	def _adjust_profiles(self, partner, partner_ad, observed, stated_ideal, our_projection):
		adj_obs, adj_ideal, trust = super()._adjust_profiles(
			partner, partner_ad, observed, stated_ideal, our_projection
		)
		shift = self.SETTINGS['score_debias']
		if (
			shift
			and partner_ad.own_dim < self.monotonic
			and partner_ad.own_value >= self.SETTINGS['guard_min_ad']
		):
			values = list(adj_obs)
			values[partner_ad.own_dim] = max(0, values[partner_ad.own_dim] - shift)
			adj_obs = tuple(values)
		return adj_obs, adj_ideal, trust

	def decide(self, date, observed, stated_ideal):
		previous_count = self._partner(date.partner).proposals_to
		decision = super().decide(date, observed, stated_ideal)
		s = self.SETTINGS
		if s['guard'] != 'none':
			state = self._partner(date.partner)
			ours = self._estimate(observed, stated_ideal)
			if s['guard'] == 'first':
				ours = min(ours, self._estimate(state.observations[0], state.stated_ideals[0]))
			elif s['guard'] == 'debiased':
				ad = date.partner_ad
				if ad.own_dim < self.monotonic and ad.own_value >= s['guard_min_ad']:
					values = list(observed)
					values[ad.own_dim] = max(0, values[ad.own_dim] - s['guard_shift'])
					ours = min(ours, self._estimate(tuple(values), stated_ideal))
			elif s['guard'] == 'bounded':
				r = s['guard_radius']
				mine = sum(
					max(
						self._gap(i, self.ideal[i], max(0, v - r)),
						self._gap(i, self.ideal[i], min(10, v + r)),
					)
					for i, v in enumerate(observed)
				)
				theirs = sum(
					max(
						self._gap(i, max(0, v - r), self.profile[i]),
						self._gap(i, min(10, v + r), self.profile[i]),
					)
					for i, v in enumerate(stated_ideal)
				)
				ours = min(ours, self.bonus - mine - theirs / 2)
			if ours <= s['guard_margin'] * self.bonus:
				state.proposals_to = previous_count
				return Decision()
		return decision


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


class Player3(Deception, Coordination, _Player3Core):
	"""Controlled deception validated against frozen coordination on 100 fresh seeds."""

	SETTINGS = {
		**DECEPTION_DEFAULTS,
		'ad_inflate': 2,
		'preference_blend': 0.65,
		'adaptive': 'late',
		'late_fraction': 0.35,
		'stable_ideal': False,
		'guard': 'raw',
	}

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
	MAX_ONE_SIDED_PROPOSALS = float('inf')
	STOP_AFTER_OFFERS = 4
	CANDIDATE_CAP = 8
	BILATERAL_REPEAT = True


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


"""Independent research additions. Appended to a preserved Group 3 source only.

Population evidence is an individual's public first-date reports, counted once.
No imports, opponent identification, hidden data, or external dependencies here.
"""

_BaselinePlayer3 = Player3


class ResearchPolicy:
	EXPECTED_ADS = False
	UNIFIED_LIST = False
	SHORTLIST = None
	QUALITY = False
	QUALITY_STRENGTH = 0.5
	AD_POLICY = 'baseline'
	FIRST_IDEAL = False
	FIX_TRUST = False
	RECIPROCITY = False
	POPULATION = True

	def __init__(self, snapshot, ctx):
		super().__init__(snapshot, ctx)
		self._population = [[1.0] * 11 for _ in range(self.dims)]
		self._population_seen = set()
		self._loss_cache = None
		self._round_now = 1

	def _population_loss(self, dim, fixed, ideal_random):
		row = self._population[dim]
		return sum(
			w * (self._gap(dim, x, fixed) if ideal_random else self._gap(dim, fixed, x))
			for x, w in enumerate(row)
		) / sum(row)

	def _prior_losses(self):
		if self._loss_cache is None:
			self._loss_cache = (
				[self._population_loss(i, self.ideal[i], False) for i in range(self.dims)],
				[self._population_loss(i, self.profile[i], True) for i in range(self.dims)],
			)
		return self._loss_cache

	def _ad_utilities(self, ad):
		mine, theirs = self._prior_losses()
		a = (
			sum(mine)
			- mine[ad.own_dim]
			+ self._gap(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		)
		b = (
			sum(theirs)
			- theirs[ad.seek_dim]
			+ self._gap(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		)
		return self.bonus - a - b / 2, self.bonus - b - a / 2

	def _ad_cost(self, ad):
		if not self.EXPECTED_ADS:
			return super()._ad_cost(ad)
		ours, theirs = self._ad_utilities(ad)
		return -ours + (max(0.0, -theirs) if self.RECIPROCITY else 0.0)

	def advertise(self):
		ad = super().advertise()
		if self.AD_POLICY == 'baseline' or ad.own_dim >= self.monotonic:
			return ad
		v = self.profile[ad.own_dim]
		if self.AD_POLICY == 'honest':
			shift = 0
		elif self.AD_POLICY == 'budget':
			# Bound the maximum extra squared shortfall hidden by the lie.
			budget = max(0.0, self.bonus) / (4 * self.dims)
			shift = max(
				(s for s in range(4) if (10 - v) ** 2 - max(0, 10 - v - s) ** 2 <= budget),
				default=0,
			)
		elif self.AD_POLICY == 'selective':
			# Weak advertised traits incur more harm for the same exaggeration.
			shift = 2 if v >= 7 else (1 if v >= 4 and self.bonus >= 10 * self.dims else 0)
		else:
			shift = 1
		self._ad = Advertisement(ad.own_dim, min(10, v + shift), ad.seek_dim, ad.seek_value)
		return self._ad

	def choose_dates(self, candidates, turn):
		self._round_now = turn.round
		self._last_singles = turn.singles
		if not self.UNIFIED_LIST:
			chosen = super().choose_dates(candidates, turn)
			if self.SHORTLIST is not None:
				cap = self.SHORTLIST if turn.rounds_left > 4 else self.CANDIDATE_CAP
				chosen = chosen[:cap]
			return chosen
		ranked = []
		for c in candidates:
			if c.index in self._est:
				score = self._est[c.index]
				if score <= 0:
					continue
				state = self._partner(c.index)
				if (
					state.proposals_to >= self.STOP_AFTER_OFFERS
					and c.index not in self._proposed_to_me
				):
					continue
				# An observed proposal makes this a one-date rather than two-date opportunity.
				if c.index in self._proposed_to_me:
					score += max(0.0, self.bonus - score) / 4
			else:
				score, theirs = self._ad_utilities(c.ad)
				if self.RECIPROCITY:
					score -= max(0.0, -theirs)
			ranked.append((score, -c.index, c.index))
		ranked.sort(reverse=True)
		cap = self.SHORTLIST or self.CANDIDATE_CAP
		if turn.rounds_left <= 4:
			cap = self.CANDIDATE_CAP
		if self.list_limit is not None:
			cap = min(cap, self.list_limit)
		return [v[2] for v in ranked[:cap]]

	def _quality_bar(self, date):
		if date.rounds_left <= 3 or self._last_singles <= 4:
			return 0.0
		mine, theirs = self._prior_losses()
		prior = self.bonus - sum(mine) - sum(theirs) / 2
		values = list(self._est.values())
		if values:
			reference = (prior * 4 + sum(values)) / (4 + len(values))
			variance = sum((x - reference) ** 2 for x in values) / len(values)
		else:
			reference, variance = prior, (self.bonus / 4) ** 2
		aspiration = max(0.0, reference + self.QUALITY_STRENGTH * math.sqrt(variance))
		breadth = min(1.0, self._last_singles / max(2, self.people))
		horizon = min(1.0, (date.rounds_left - 2) / max(4.0, self.rounds / 4))
		return aspiration * math.sqrt(breadth) * horizon

	def decide(self, date, observed, stated_ideal):
		self._round_now = date.round
		if self.POPULATION and date.partner not in self._population_seen:
			self._population_seen.add(date.partner)
			for i in range(self.dims):
				self._population[i][observed[i]] += 1
				self._population[i][stated_ideal[i]] += 1
			self._loss_cache = None
		state = self._partner(date.partner)
		previous_proposals = state.proposals_to
		if self.FIX_TRUST:
			# The old core appends the current observation before its trust check.
			# Assess history before that append, using only previously seen reports.
			if state.observations and observed != state.observations[-1]:
				state.trust = max(0.15, state.trust * 0.65)
			elif (
				not state.observations
				and abs(observed[date.partner_ad.own_dim] - date.partner_ad.own_value) > 1
			):
				state.trust = max(0.15, state.trust * 0.55)
		if self.FIRST_IDEAL and state.stated_ideals:
			# Preference relaxation does not change the underlying private scoring profile.
			first = state.stated_ideals[0]
			if decrement(first, self.profile, self.monotonic) > decrement(
				stated_ideal, self.profile, self.monotonic
			):
				stated_ideal = first
		decision = super().decide(date, observed, stated_ideal)
		if self.QUALITY and not decision.incompatible:
			est = min(self._est.get(date.partner, 0.0), self._estimate(observed, stated_ideal))
			if est <= self._quality_bar(date):
				state.proposals_to = previous_proposals
				return Decision()
		return decision


class Player3(ResearchPolicy, _BaselinePlayer3):
	pass


"""Second-stage hypotheses: coordinate quality rejection with future search."""


class Refinement:
	FILTER_REJECTED = False
	OFFER_DISCOUNT = False
	LIVE_RESERVATION = False
	AD_INFORMATION = False
	COOLDOWN = False

	def __init__(self, snapshot, ctx):
		super().__init__(snapshot, ctx)
		self._recent_rejection = {}
		self._seen_round = {}
		self._previous_singles = ctx.people
		self._previous_round = 0
		self._market_survival = 1.0

	def advertise(self):
		ad = super().advertise()
		if not self.AD_INFORMATION:
			return ad

		# Select disclosed dimensions by variance of their actual quadratic loss.
		def variance(dim, fixed, ideal_random):
			vals = [
				self._gap(dim, x, fixed) if ideal_random else self._gap(dim, fixed, x)
				for x in range(11)
			]
			mean = sum(vals) / 11
			return sum((x - mean) ** 2 for x in vals) / 11

		own = max(range(self.dims), key=lambda i: variance(i, self.profile[i], True))
		seek = max(range(self.dims), key=lambda i: variance(i, self.ideal[i], False))
		self._ad = Advertisement(own, self.profile[own], seek, self.ideal[seek])
		return self._ad

	def _quality_bar(self, date):
		bar = super()._quality_bar(date)
		if self.LIVE_RESERVATION:
			# Survival over the two rounds needed to sample and marry an alternative.
			bar *= self._market_survival**2
		if self.OFFER_DISCOUNT and date.partner in self._proposed_to_me:
			bar *= 0.8
		return bar

	def choose_dates(self, candidates, turn):
		gap = turn.round - self._previous_round
		if gap > 0 and self._previous_singles > 2:
			survival = (turn.singles / self._previous_singles) ** (1 / gap)
			self._market_survival = max(0.25, min(1.0, survival))
		self._previous_round, self._previous_singles = turn.round, turn.singles
		self._last_singles = turn.singles
		if self.FILTER_REJECTED and turn.rounds_left > 3 and turn.singles > 4:
			bar = self._quality_bar(
				DateContext(turn.round, turn.rounds_left, -1, self._ad, 1, False)
			)
			candidates = tuple(
				c for c in candidates if c.index not in self._est or self._est[c.index] > bar
			)
		if self.COOLDOWN and turn.rounds_left > 4:
			candidates = tuple(
				c for c in candidates if turn.round - self._recent_rejection.get(c.index, -100) > 2
			)
		return super().choose_dates(candidates, turn)

	def decide(self, date, observed, stated_ideal):
		self._seen_round[date.partner] = date.round
		decision = super().decide(date, observed, stated_ideal)
		if not decision.again:
			self._recent_rejection[date.partner] = date.round
		return decision


_StageOnePlayer3 = Player3


class Player3(Refinement, _StageOnePlayer3):
	pass

	QUALITY = True
	QUALITY_STRENGTH = 1.0
	EXPECTED_ADS = True
	UNIFIED_LIST = True
	FILTER_REJECTED = True


# Independently authored continuation of our frozen Group 3 policy.
# Group 9's public-code study motivated separating fit from willingness and
# considering remaining opportunities. No opponent implementation is used.

_FrozenQuFilter = Player3


class CoveragePolicy:
	R2_DEPLETION = False
	R2_OPPORTUNITY = False
	R2_WILLINGNESS = False
	R2_RETAIN = False
	R2_REOPEN = False
	R2_ATTAINABLE = False
	R2_ENDGAME = False
	R2_DISPERSION = False
	R2_LOCAL_FILTER = False

	def __init__(self, snapshot, ctx):
		super().__init__(snapshot, ctx)
		self._r2_first_scores = {}
		self._r2_last_date = {}
		self._r2_last_offer = {}
		self._r2_received = {}
		self._r2_eligible = {}
		self._r2_offers = {}
		self._r2_repeat_requests = 0
		self._r2_repeat_successes = 0
		self._r2_visible = ()
		self._r2_decision = Decision()
		self._r2_variance_cache = None

	def _r2_singles(self, round_number):
		elapsed = max(0, round_number - self._previous_round)
		return 2 + max(0, self._last_singles - 2) * self._market_survival**elapsed

	def _r2_prior_variance(self):
		# Independent quadratic-loss variance. A constant marriage bonus shifts
		# utility but cannot change its variance. The population is public reports.
		if self._r2_variance_cache is None:
			total = 0.0
			for i, row in enumerate(self._population):
				n = sum(row)
				for fixed, reverse, weight in (
					(self.ideal[i], False, 1),
					(self.profile[i], True, 0.25),
				):
					losses = [
						self._gap(i, x, fixed) if reverse else self._gap(i, fixed, x)
						for x in range(11)
					]
					mean = sum(w * x for w, x in zip(row, losses)) / n  # noqa: B905 - equal lengths by construction
					total += weight * sum(w * (x - mean) ** 2 for w, x in zip(row, losses)) / n  # noqa: B905 - equal lengths by construction
			self._r2_variance_cache = total
		return self._r2_variance_cache

	def _r2_reciprocity(self, partner, theirs=None):
		n = self._r2_eligible.get(partner, 0)
		offers = self._r2_offers.get(partner, 0)
		if partner in self._r2_received:
			# Repeated offers give stronger evidence than an isolated old one.
			age = max(0, self._round_now - self._r2_received[partner])
			return (offers + 2) / (n + 3) / (1 + age / max(2, self.rounds / 4))
		if theirs is None:
			state = self._partners.get(partner)
			if state and state.observations:
				theirs = (
					self.bonus
					- decrement(state.stated_ideals[-1], self.profile, self.monotonic)
					- decrement(self.ideal, state.observations[-1], self.monotonic) / 2
				)
		sigma = max(1.0, math.sqrt(self._r2_prior_variance()))
		fit = 0.5 if theirs is None else 0.5 * (1 + math.erf(theirs / (sigma * math.sqrt(2))))
		return fit / (1 + n - offers)

	def _r2_continuation(self, date):
		# Finite remaining two-date opportunities, with a zero payoff when a
		# repeat/commitment fails. This is a public-data model, not oracle access.
		left = max(0, date.rounds_left - 1)
		count = max(0.0, self._r2_singles(date.round) - 2)
		visible = [c for c in self._r2_visible if c.index != date.partner]
		unseen = sum(c.index not in self._est for c in visible)
		age = max(0, date.round - self._previous_round)
		attempts = min(left // 2, int(min(count, unseen) * self._market_survival**age))
		known = [
			max(0.0, self._est.get(c.index, 0)) * self._r2_reciprocity(c.index)
			for c in visible
			if c.index in self._est
		]
		value = max(known, default=0.0) if left else 0.0
		mine, theirs = self._prior_losses()
		mean = self.bonus - sum(mine) - sum(theirs) / 2
		sd = math.sqrt(self._r2_prior_variance())
		values = list(self._r2_first_scores.values()) + [mean - sd, mean, mean, mean + sd]
		repeat = (1 + self._r2_repeat_successes) / (2 + self._r2_repeat_requests)
		offers = sum(self._r2_offers.values())
		eligible = sum(self._r2_eligible.values())
		commit = (1 + offers) / (2 + eligible)
		for _ in range(attempts):
			value += repeat * commit * sum(max(0.0, x - value) for x in values) / len(values)
		return value

	def _quality_bar(self, date):
		bar = super()._quality_bar(date)
		if self.R2_DISPERSION and date.rounds_left > 3 and self._last_singles > 4:
			mine, theirs = self._prior_losses()
			prior = self.bonus - sum(mine) - sum(theirs) / 2
			values = list(self._est.values())
			ref = (4 * prior + sum(values)) / (4 + len(values))
			var = (4 * self._r2_prior_variance() + sum((v - ref) ** 2 for v in values)) / (
				4 + len(values)
			)
			horizon = min(1.0, (date.rounds_left - 2) / max(4, self.rounds / 4))
			bar = (
				max(0, ref + math.sqrt(var))
				* math.sqrt(min(1, self._last_singles / max(2, self.people)))
				* horizon
			)
		if self.R2_DEPLETION and self._last_singles > 4:
			forecast = self._r2_singles(date.round)
			bar *= math.sqrt(forecast / self._last_singles)
		if self.R2_OPPORTUNITY:
			bar = min(bar, self._r2_continuation(date))
		if self.R2_WILLINGNESS and date.partner in self._r2_received:
			probability = self._r2_reciprocity(date.partner)
			bar = min(bar, self._r2_continuation(date) / max(0.05, probability))
		return bar

	def choose_dates(self, candidates, turn):
		self._round_now = turn.round
		self._r2_visible = candidates
		endgame = self.R2_ENDGAME and turn.rounds_left <= 3
		if self.R2_ENDGAME and turn.rounds_left == 1:
			candidates = tuple(c for c in candidates if c.dates and self._est.get(c.index, 0) > 0)
		saved = {}
		if self.R2_REOPEN or endgame:
			for c in candidates:
				state = self._partners.get(c.index)
				if state and state.proposals_to >= self.STOP_AFTER_OFFERS:
					delay = turn.round - self._r2_last_offer.get(c.index, turn.round)
					if endgame or delay >= max(2, self.rounds // 10):
						saved[c.index] = state.proposals_to
						state.proposals_to = 0
		try:
			filtered = self.FILTER_REJECTED
			if self.R2_LOCAL_FILTER:
				gap = turn.round - self._previous_round
				if gap > 0 and self._previous_singles > 2:
					self._market_survival = max(
						0.25, min(1.0, (turn.singles / self._previous_singles) ** (1 / gap))
					)
				self._previous_round, self._previous_singles = turn.round, turn.singles
				self._last_singles = turn.singles
				if filtered and turn.rounds_left > 3 and turn.singles > 4:
					candidates = tuple(
						c
						for c in candidates
						if c.index not in self._est
						or self._est[c.index]
						> self._quality_bar(
							DateContext(
								turn.round, turn.rounds_left, c.index, c.ad, c.dates + 1, False
							)
						)
					)
				# Our partner-specific filter already ran; avoid the inherited
				# sentinel partner=-1 treating this opportunity as its own rival.
				self.FILTER_REJECTED = False
			try:
				chosen = super().choose_dates(candidates, turn)
			finally:
				self.FILTER_REJECTED = filtered
			if self.R2_ATTAINABLE:
				# Rank by positive return times a public willingness estimate.
				# Reproduce baseline eligibility, changing membership, not merely
				# order (the simulator treats the shortlist as a set).
				ranked = []
				bar = self._quality_bar(
					DateContext(turn.round, turn.rounds_left, -1, self._ad, 1, False)
				)
				for c in candidates:
					if c.index in self._est:
						h = self._est[c.index]
						if h <= 0 or (turn.rounds_left > 3 and turn.singles > 4 and h <= bar):
							continue
						if (
							self._partner(c.index).proposals_to >= self.STOP_AFTER_OFFERS
							and c.index not in self._proposed_to_me
						):
							continue
						probability = self._r2_reciprocity(c.index)
						if c.index in self._proposed_to_me:
							h += max(0.0, self.bonus - h) / 4
					else:
						h, theirs = self._ad_utilities(c.ad)
						probability = self._r2_reciprocity(c.index, theirs)
					ranked.append((h * probability, -c.index, c.index))
				ranked.sort(reverse=True)
				cap = (
					min(self.CANDIDATE_CAP, self.list_limit)
					if self.list_limit is not None
					else self.CANDIDATE_CAP
				)
				chosen = [r[2] for r in ranked[:cap]]
			return chosen
		finally:
			for i, value in saved.items():
				self._partners[i].proposals_to = value

	def decide(self, date, observed, stated_ideal):
		if date.partner not in self._population_seen:
			self._r2_variance_cache = None
		decision = super().decide(date, observed, stated_ideal)
		h = min(self._est.get(date.partner, 0), self._estimate(observed, stated_ideal))
		self._r2_first_scores.setdefault(date.partner, h)
		self._r2_last_date[date.partner] = date.round
		if (
			self.R2_RETAIN
			and not decision.again
			and not decision.incompatible
			and date.rounds_left > 1
			and h > 0
		):
			next_date = DateContext(
				date.round + 1,
				date.rounds_left - 1,
				date.partner,
				date.partner_ad,
				date.date_number + 1,
				True,
			)
			if (
				h > self._quality_bar(next_date)
				and self._partner(date.partner).proposals_to < self.STOP_AFTER_OFFERS
			):
				decision = Decision(again=True, propose=decision.propose)
		if decision.propose:
			self._r2_last_offer[date.partner] = date.round
		self._r2_decision = decision
		return decision

	def on_outcome(self, outcome):
		if outcome.date_number >= 2:
			self._r2_eligible[outcome.partner] = self._r2_eligible.get(outcome.partner, 0) + 1
			self._r2_offers[outcome.partner] = self._r2_offers.get(outcome.partner, 0) + int(
				outcome.partner_proposed
			)
		if outcome.partner_proposed:
			self._r2_received[outcome.partner] = outcome.round
		if self._r2_decision.again and not outcome.married and outcome.round < self.rounds:
			self._r2_repeat_requests += 1
			self._r2_repeat_successes += int(outcome.again)
		super().on_outcome(outcome)


class Player3(CoveragePolicy, _FrozenQuFilter):
	pass


"""Independent date-loss calibration and exact-tie shortlist experiments.

Only our Group 3 estimates and legal observations are used. The point-loss
versus expected-loss distinction follows directly from the game's convex
loss; this is not an opponent decoder or an opponent's implementation.
"""

_CalibrationBase = Player3


class CalibrationPolicy:
	CALIBRATED_LOSS = False
	SHORTLIST_TIES = False

	def __init__(self, snapshot, ctx):
		super().__init__(snapshot, ctx)
		self._calibration_pending = None
		self._tie_opportunities = 0
		self._tie_changes = 0

	def _adjust_profiles(self, partner, partner_ad, observed, stated_ideal, our_projection):
		adjusted = super()._adjust_profiles(
			partner, partner_ad, observed, stated_ideal, our_projection
		)
		if not self.CALIBRATED_LOSS or adjusted[2] >= 1:
			return adjusted
		trust = adjusted[2]
		mine, theirs = self._prior_losses()
		# A mirrored ideal was already treated as uninformative by the core.
		# Its expected loss is therefore the entire existing prior, rather
		# than loss at the prior's mean or at the mirrored declaration.
		ideal_trust = 0.0 if self._mirrored_ideal(stated_ideal, our_projection) else trust
		expected_loss = sum(
			trust * self._gap(i, self.ideal[i], observed[i])
			+ (1 - trust) * mine[i]
			+ 0.5
			* (
				ideal_trust * self._gap(i, stated_ideal[i], self.profile[i])
				+ (1 - ideal_trust) * theirs[i]
			)
			for i in range(self.dims)
		)
		point_score = super()._estimate(*adjusted)
		score = min(point_score, self.bonus - expected_loss)
		# The core immediately evaluates exactly these adjusted profiles.
		# Consume this once so subsequent raw guards retain their old meaning.
		self._calibration_pending = (adjusted, score)
		return adjusted

	def _estimate(self, observed, stated_ideal, trust=1.0):
		pending = self._calibration_pending
		if pending is not None and (observed, stated_ideal, trust) == pending[0]:
			self._calibration_pending = None
			return pending[1]
		return super()._estimate(observed, stated_ideal, trust)

	def decide(self, date, observed, stated_ideal):
		self._calibration_pending = None
		try:
			return super().decide(date, observed, stated_ideal)
		finally:
			self._calibration_pending = None

	def choose_dates(self, candidates, turn):
		chosen = list(super().choose_dates(candidates, turn))
		if not self.SHORTLIST_TIES or not chosen:
			return chosen
		cap = self.SHORTLIST or self.CANDIDATE_CAP
		if turn.rounds_left <= 4:
			cap = self.CANDIDATE_CAP
		if self.list_limit is not None:
			cap = min(cap, self.list_limit)
		if len(chosen) < cap:
			return chosen
		# The final baseline entry defines the actual cutoff. Known partners
		# are anchors: only equal-scoring fresh entries can be exchanged.
		fresh = {c.index: c for c in candidates if c.index not in self._est}
		if chosen[-1] not in fresh:
			return chosen
		cutoff = self._ad_utilities(fresh[chosen[-1]].ad)[0]
		tied = sorted(i for i, c in fresh.items() if self._ad_utilities(c.ad)[0] == cutoff)
		positions = [j for j, i in enumerate(chosen) if i in tied]
		if len(tied) <= len(positions):
			return chosen
		self._tie_opportunities += 1
		replacements = self.rng.sample(tied, len(positions))
		if {chosen[j] for j in positions} != set(replacements):
			self._tie_changes += 1
		for j, value in zip(positions, replacements):  # noqa: B905 - equal lengths by construction
			chosen[j] = value
		return chosen


class Player3(CalibrationPolicy, _CalibrationBase):
	pass


class Player3(Player3):
	R2_DISPERSION = True
	R2_ENDGAME = True
