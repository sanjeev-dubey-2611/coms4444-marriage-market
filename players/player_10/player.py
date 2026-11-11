"""Group 10's greedy strategy with an optimistic projected profile."""

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


class Player10(BasePlayer):
	"""Greedy dating decisions, with maximum values projected on monotonic traits."""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.estimates: dict[int, float] = {}
		self.proposed_to_me: set[int] = set()

		self.samples: list = [self.profile, self.ideal]
		self.dated: list[int] = []

		self.monotonic_bump = int(self.bonus / 75)

	def advertise(self) -> Advertisement:
		# Show off the monotonic dimension we score best on, and ask for the
		# dimension where our ideal is furthest from the middle of the scale.
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		seek = max(range(self.dims), key=lambda i: abs(self.ideal[i] - 5))
		return Advertisement(
			own_dim=own, own_value=self.profile[own], seek_dim=seek, seek_value=self.ideal[seek]
		)

	def is_collect_data(self, ctx: RoundContext | DateContext):
		"""We collect data if we are less than half-way through and we haven't seen everyone.
		So we stop collecting data if we are more than half-way through or we have seen everyone.
		"""
		return ctx.rounds_left > ctx.round and len(self.dated) < self.people

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self.remaining = [c.index for c in candidates]
		if self.is_collect_data(turn):
			# When collecting data, choose everyone we haven't chosen before
			chosen = [c.index for c in candidates if c not in self.dated]
			return chosen[: self.list_limit]
		else:
			chosen = [c.index for c in candidates if self.estimates.get(c.index, 0.0) > 0]
			fresh = sorted(
				(c for c in candidates if c.index not in self.estimates),
				key=lambda c: (self._ad_cost(c.ad), c.index),
			)
			chosen += [c.index for c in fresh[: max(1, len(fresh) // 2)]]
			return chosen[: self.list_limit]

	def _gap(self, dim: int, want: int, have: int) -> int:
		gap = want - have
		if dim < self.monotonic and gap <= 0:
			return 0
		return gap * gap

	def _ad_cost(self, ad: Advertisement) -> float:
		"""Estimate cost from the two advertised dimensions."""
		mine = self._gap(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
		theirs = self._gap(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
		return mine + theirs / 2

	def project(self, date: DateContext) -> Profile:
		projected = list(self.profile)
		for i in range(self.monotonic):
			projected[i] = min(10, projected[i] + self.monotonic_bump)
		return tuple(projected)

	# def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
	# 	ideal = list(self.ideal)
	# 	for i in range(self.monotonic):
	# 		ideal[i] = max(0, ideal[i] - 2)
	# 	return tuple(ideal)

	def calculate_estimate(self, their_profile, their_ideal):
		mine = decrement(self.ideal, their_profile, self.monotonic)
		theirs = decrement(their_ideal, self.profile, self.monotonic)
		return self.bonus - mine - theirs / 2

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		estimate = self.calculate_estimate(observed, stated_ideal)
		self.estimates[date.partner] = estimate

		# if self.is_collect_data(date):
		# 	if (
		# 		date.date_number == 1
		# 	):  # There is a chance we get paired again with them without our consent, so only add if first date
		# 		self.dated.append(date.partner)
		# 		self.samples.extend([observed, stated_ideal])
		# 	# Don't do anything, but if it seems our estimate will be negative, then we can set them as incompatible
		# 	return Decision(incompatible=estimate <= 0)
		# elif date.partner not in self.dated:
		# 	# We can still collect data if we see someone new while searching for matches
		# 	self.dated.append(date.partner)
		# 	self.samples.extend([observed, stated_ideal])
		# else:
		# 	i = self.dated.index(date.partner)
		# 	_prev_observed = self.samples[i * 2]
		# 	_prev_ideal = self.samples[i * 2 + 1]
		# @TODO: We can maybe do something if they've changed their profile

		if estimate <= 0:
			return Decision(incompatible=True)

		# Find average person remaining and calculate their estimate. We want to propose if it is better than mean.
		# samples = np.array(self.samples)
		# dated = np.array(self.dated)
		# remaining_samples = samples[np.where(np.isin(dated, self.remaining))]
		# remaining_mean = remaining_samples.mean(axis=0)
		# mean_estimate = self.calculate_estimate(remaining_mean, remaining_mean)

		propose_threshold = float(date.rounds_left) / float(self.rounds) * self.bonus

		# @TODO: we can potentially do some threshold decay based on number of turns left and/or single people left
		# propose_threshold = mean_estimate
		# propose_threshold = 0
		# if date.partner in self.proposed_to_me:
		# 	propose_threshold = mean_estimate / 2
		propose = bool(date.date_number >= 2 and estimate > propose_threshold)
		return Decision(again=estimate > propose_threshold, propose=propose)

	def on_outcome(self, outcome: DateOutcome) -> None:
		if outcome.partner_proposed:
			self.proposed_to_me.add(outcome.partner)
