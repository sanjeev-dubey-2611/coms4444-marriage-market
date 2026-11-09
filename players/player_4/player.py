"""Group 4's player, following players/player_4/strategy_1104.md.

Changes since that strategy:
- The threshold is a falling quantile of first-date scores we have seen,
  not a fixed fraction of the bonus, so it adapts to B and the distribution.
- Monotonic dimensions are projected truthfully; only regular extremes are masked.
- The lie check ignores the ad's seek half, and a liar's penalty is inflated
  rather than their score scaled down.
"""

from bisect import insort
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

# Masking bounds for regular dimensions, used by project() and advertise().
REGULAR_PROJECTION_LOW = 2
REGULAR_PROJECTION_HIGH = 8

# Threshold quantile of first-date scores at the two ends of the run.
EARLY_THRESHOLD_QUANTILE = 0.9
LATE_THRESHOLD_QUANTILE = 0.0
# Share of the run at each end where the quantile stays flat.
FLAT_THRESHOLD_SHARE = 0.1
# Until we have this many first-date scores, fall back to a fraction of the bonus.
MIN_SCORES_FOR_QUANTILE = 5
FALLBACK_THRESHOLD_FRACTION = 0.8
# Runs shorter than this scale the fallback down, since a short run may end
# before we ever have enough scores for the quantile.
FULL_FALLBACK_ROUNDS = 20

UNTRUTHFUL_PENALTY_MULTIPLIER = 1.2
FAILED_PROPOSALS_BEFORE_COOLDOWN = 2
# How far below the threshold (as a fraction of the bonus) we still accept a proposal.
ACCEPT_PROPOSAL_MARGIN_FRACTION = 0.1


@dataclass
class DateRecord:
	"""One date with a partner, from our side."""

	round: int
	date_number: int
	expected_score: float
	we_proposed: bool = False
	they_proposed: bool = False


@dataclass
class PlayerProfile:
	"""Everything we know about one other player."""

	index: int
	advertisement: Advertisement | None = None
	projected_profile: Profile | None = None  # from our latest date with them
	stated_ideal: Profile | None = None  # from our latest date with them
	is_untruthful: bool = False  # sticky once set
	expected_score: float | None = None  # None until we have dated them
	dating_history: list[DateRecord] = field(default_factory=list)
	failed_proposals_in_a_row: int = 0
	cooldown_last_round: int = 0  # excluded from choose_dates up to and including this round
	accept_next_proposal_chance: bool = False  # they proposed and were close to our threshold

	@property
	def has_dated(self) -> bool:
		return self.expected_score is not None

	def is_in_cooldown(self, round_number: int) -> bool:
		return round_number <= self.cooldown_last_round


class Player4(BasePlayer):
	"""Ad-ranked first dates, a falling score threshold, and mild masking of extremes."""

	COOLDOWN_ROUNDS = 5

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.player_profiles: dict[int, PlayerProfile] = {}
		# Expected score from every first date, kept sorted for the threshold quantile.
		# Dated players below the threshold stay in player_profiles and come back
		# into choose_dates once it drops.
		self.first_date_scores: list[float] = []

	# ----- helpers -----

	def _get_profile(self, index: int) -> PlayerProfile:
		if index not in self.player_profiles:
			self.player_profiles[index] = PlayerProfile(index=index)
		return self.player_profiles[index]

	def _is_monotonic(self, dim: int) -> bool:
		return dim < self.monotonic

	def _mask_value(self, dim: int, value: int) -> int:
		"""Hide extremes: pull regular values into [2, 8].

		Monotonic values stay true. Inflating them makes a partner marry us
		expecting more than we have, and we pay half of their real penalty.
		"""
		if self._is_monotonic(dim):
			return value
		return min(max(value, REGULAR_PROJECTION_LOW), REGULAR_PROJECTION_HIGH)

	def _masked_profile(self) -> Profile:
		return tuple(self._mask_value(dim, value) for dim, value in enumerate(self.profile))

	@staticmethod
	def _centeredness(value: int) -> int:
		# 5 -> 10, 7 -> 6, 0 or 10 -> 0
		return MAX_VALUE - abs(value - MAX_VALUE // 2) * 2

	def _dim_penalty(self, dim: int, desired: int, actual: int) -> int:
		"""One-dimension decrement, same rule as models.profile.decrement."""
		if self._is_monotonic(dim) and actual >= desired:
			return 0
		return (desired - actual) ** 2

	def _ramp_progress(self, round_number: int) -> float:
		"""0 for the first 10% of rounds, 1 for the last 10%, linear in between."""
		progress = (round_number - 1) / self.rounds  # 0 on round 1, just under 1 on the last
		ramp_start = FLAT_THRESHOLD_SHARE
		ramp_end = 1 - FLAT_THRESHOLD_SHARE
		# Clamp progress through the ramp to [0, 1].
		return min(max((progress - ramp_start) / (ramp_end - ramp_start), 0.0), 1.0)

	def _threshold_score(self, round_number: int) -> float:
		"""Minimum expected score to ask again / propose this round.

		A quantile of the first-date scores seen so far, falling from the 90th
		percentile to the minimum along the ramp, and never below 0. Scores scale
		with B and the population, so the threshold does too.
		"""
		ramp_progress = self._ramp_progress(round_number)
		scores = self.first_date_scores
		if len(scores) < MIN_SCORES_FOR_QUANTILE:
			# Too few dates to trust a quantile: falling fraction of the bonus,
			# smaller in short runs (5 rounds -> a quarter of the usual fraction).
			fraction = FALLBACK_THRESHOLD_FRACTION * min(1.0, self.rounds / FULL_FALLBACK_ROUNDS)
			return fraction * (1 - ramp_progress) * self.bonus
		quantile = (
			EARLY_THRESHOLD_QUANTILE
			+ (LATE_THRESHOLD_QUANTILE - EARLY_THRESHOLD_QUANTILE) * ramp_progress
		)
		# Index of that quantile in the sorted scores, clamped to the last one.
		position = min(int(quantile * len(scores)), len(scores) - 1)
		return max(0.0, scores[position])

	def _ad_penalty(self, advertisement: Advertisement) -> float:
		"""Lower is better: expected penalty implied by their ad alone.

		Mirrors happiness: our full penalty plus half of theirs.
		"""
		my_penalty = self._dim_penalty(
			advertisement.own_dim, self.ideal[advertisement.own_dim], advertisement.own_value
		)
		their_penalty = self._dim_penalty(
			advertisement.seek_dim, advertisement.seek_value, self.profile[advertisement.seek_dim]
		)
		return my_penalty + their_penalty / 2

	def _expected_score(self, partner: PlayerProfile) -> float:
		"""Happiness if we married them, taking their projection and stated ideal as true."""
		my_penalty = decrement(self.ideal, partner.projected_profile, self.monotonic)
		their_penalty = decrement(partner.stated_ideal, self.profile, self.monotonic)
		total_penalty = my_penalty + their_penalty / 2
		if partner.is_untruthful:
			# Inflate the penalty, not shrink the score: the discount then means the
			# same at any B and never pushes a negative score up towards 0.
			total_penalty *= UNTRUTHFUL_PENALTY_MULTIPLIER
		return self.bonus - total_penalty

	def _check_truthfulness(
		self, partner: PlayerProfile, observed: Profile, stated_ideal: Profile
	) -> None:
		"""Flag them if this date contradicts their ad or their previous date.

		Only the ad's own half is checked. Some honest players post a made-up
		seek value (always 0, say) and still describe their true ideal.
		"""
		ad = partner.advertisement
		contradicts_ad = ad is not None and observed[ad.own_dim] != ad.own_value
		changed_since_last_date = partner.projected_profile is not None and (
			observed != partner.projected_profile or stated_ideal != partner.stated_ideal
		)
		if contradicts_ad or changed_since_last_date:
			partner.is_untruthful = True

	# ----- public actions -----

	def advertise(self) -> Advertisement:
		"""Advertise our most widely appealing trait and least demanding seek."""

		def own_appeal(dim: int) -> int:
			# Monotonic: higher is better. Regular: closer to 5 is better.
			value = self.profile[dim]
			return value if self._is_monotonic(dim) else self._centeredness(value)

		def seek_leniency(dim: int) -> int:
			# Monotonic: a low demand is easy to meet. Regular: closer to 5 is easier.
			value = self.ideal[dim]
			return MAX_VALUE - value if self._is_monotonic(dim) else self._centeredness(value)

		own_dim = max(range(self.dims), key=own_appeal)
		seek_dim = max(range(self.dims), key=seek_leniency)
		return Advertisement(
			own_dim=own_dim,
			# Masked, so the ad matches what project() will show.
			own_value=self._mask_value(own_dim, self.profile[own_dim]),
			seek_dim=seek_dim,
			seek_value=self.ideal[seek_dim],
		)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		threshold = self._threshold_score(turn.round)
		promising_dated: list[PlayerProfile] = []
		never_dated: list[PlayerProfile] = []

		for candidate in candidates:  # candidates are all still single
			partner = self._get_profile(candidate.index)
			partner.advertisement = candidate.ad
			if partner.is_in_cooldown(turn.round):
				continue
			if not partner.has_dated:
				never_dated.append(partner)
			elif partner.expected_score >= threshold or partner.accept_next_proposal_chance:
				# Above threshold, or a back-burner player the falling threshold now lets in.
				promising_dated.append(partner)

		promising_dated.sort(key=lambda partner: -partner.expected_score)  # best first
		self.rng.shuffle(never_dated)  # random tie-break for equal ad penalties
		never_dated.sort(key=lambda partner: self._ad_penalty(partner.advertisement))

		ranked = [partner.index for partner in promising_dated + never_dated]
		return ranked[: self.list_limit]  # slicing by None keeps everything

	def project(self, date: DateContext) -> Profile:
		return self._masked_profile()

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		return self.ideal

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		partner = self._get_profile(date.partner)
		partner.advertisement = date.partner_ad
		self._check_truthfulness(partner, observed, stated_ideal)
		partner.projected_profile = observed
		partner.stated_ideal = stated_ideal
		partner.expected_score = self._expected_score(partner)

		date_record = DateRecord(
			round=date.round, date_number=date.date_number, expected_score=partner.expected_score
		)
		partner.dating_history.append(date_record)
		if date.date_number == 1:
			insort(self.first_date_scores, partner.expected_score)

		# Marrying them would not beat staying single (0): the only place we mark
		# incompatible.
		if partner.expected_score <= 0:
			return Decision(incompatible=True)

		clears_threshold = partner.expected_score >= self._threshold_score(date.round)
		can_propose = date.date_number >= 2
		# During cooldown we still propose if they qualify, but never ask for more dates.
		may_ask_again = not partner.is_in_cooldown(date.round)

		if clears_threshold or (can_propose and partner.accept_next_proposal_chance):
			if can_propose:
				date_record.we_proposed = True
				return Decision(again=may_ask_again, propose=True)
			return Decision(again=True)

		# Good enough to keep, not good enough yet: wait for the threshold to drop.
		return Decision()

	def on_outcome(self, outcome: DateOutcome) -> None:
		partner = self._get_profile(outcome.partner)
		if outcome.married or not partner.dating_history:
			return
		date_record = partner.dating_history[-1]
		date_record.they_proposed = outcome.partner_proposed

		if date_record.we_proposed:
			partner.failed_proposals_in_a_row += 1
			partner.accept_next_proposal_chance = False  # used up
			if partner.failed_proposals_in_a_row >= FAILED_PROPOSALS_BEFORE_COOLDOWN:
				partner.cooldown_last_round = outcome.round + self.COOLDOWN_ROUNDS
				partner.failed_proposals_in_a_row = 0  # fresh count after cooldown

		# They proposed but fell a little short: accept a small loss for a quick exit.
		if outcome.partner_proposed and not date_record.we_proposed:
			shortfall = self._threshold_score(outcome.round) - date_record.expected_score
			if (
				date_record.expected_score > 0
				and shortfall <= ACCEPT_PROPOSAL_MARGIN_FRACTION * self.bonus
			):
				partner.accept_next_proposal_chance = True
