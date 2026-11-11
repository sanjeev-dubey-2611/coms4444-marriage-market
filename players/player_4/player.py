"""Group 4's player.

Dates the best-looking ads first, keeps a bar that starts high and falls
over the run, and never proposes below its floor.
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

REGULAR_PROJECTION_LOW = 2
REGULAR_PROJECTION_HIGH = 8


EARLY_THRESHOLD_QUANTILE = 0.9
LATE_THRESHOLD_QUANTILE = 0.0

FLAT_THRESHOLD_SHARE = 0.1

MIN_SCORES_FOR_QUANTILE = 5
FALLBACK_THRESHOLD_FRACTION = 0.8


FULL_FALLBACK_ROUNDS = 20

UNTRUTHFUL_PENALTY_MULTIPLIER = 1.2


FAILED_PROPOSALS_BEFORE_COOLDOWN = 2

ACCEPT_PROPOSAL_MARGIN_FRACTION = 0.1


SAFETY_MARGIN_FRACTION = 0.10


SAFETY_MARGIN_DECAY = 0.8


SCORE_FLOOR_FRACTION = 0.05


SCORE_FLOOR_PER_DIM = 1.0

ENDGAME_ROUNDS = 3

MIRROR_PRIOR_FRACTION = 0.25


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
	projected_profile: Profile | None = None
	stated_ideal: Profile | None = None
	is_untruthful: bool = False
	expected_score: float | None = None

	last_informative_mine: float | None = None
	last_informative_theirs: float | None = None
	dating_history: list[DateRecord] = field(default_factory=list)
	failed_proposals_in_a_row: int = 0
	cooldown_last_round: int = 0
	accept_next_proposal_chance: bool = False

	@property
	def has_dated(self) -> bool:
		return self.expected_score is not None

	def is_in_cooldown(self, round_number: int) -> bool:
		return round_number <= self.cooldown_last_round


class Player4(BasePlayer):
	"""Pre-final Player 4: patient bar, safety margin, and a hard floor."""

	COOLDOWN_ROUNDS = 5

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.player_profiles: dict[int, PlayerProfile] = {}

		self.first_date_scores: list[float] = []

		self._mine_seen: list[float] = []
		self._theirs_seen: list[float] = []

		self._last_projected: dict[int, Profile] = {}

	def _get_profile(self, index: int) -> PlayerProfile:
		if index not in self.player_profiles:
			self.player_profiles[index] = PlayerProfile(index=index)
		return self.player_profiles[index]

	def _is_monotonic(self, dim: int) -> bool:
		return dim < self.monotonic

	def _mask_value(self, dim: int, value: int) -> int:
		"""Hide extremes on regular dims; monotonic dims stay true."""
		if self._is_monotonic(dim):
			return value
		return min(max(value, REGULAR_PROJECTION_LOW), REGULAR_PROJECTION_HIGH)

	def _masked_profile(self) -> Profile:
		return tuple(self._mask_value(dim, value) for dim, value in enumerate(self.profile))

	@staticmethod
	def _centeredness(value: int) -> int:

		return MAX_VALUE - abs(value - MAX_VALUE // 2) * 2

	def _dim_penalty(self, dim: int, desired: int, actual: int) -> int:
		"""One-dimension decrement, same rule as models.profile.decrement."""
		if self._is_monotonic(dim) and actual >= desired:
			return 0
		return (desired - actual) ** 2

	def _ramp_progress(self, round_number: int) -> float:
		"""0 for the first 10% of rounds, 1 for the last 10%, linear in between."""
		progress = (round_number - 1) / self.rounds
		ramp_start = FLAT_THRESHOLD_SHARE
		ramp_end = 1 - FLAT_THRESHOLD_SHARE

		return min(max((progress - ramp_start) / (ramp_end - ramp_start), 0.0), 1.0)

	def _threshold_score(self, round_number: int) -> float:
		"""Minimum expected score to ask again / propose this round."""
		ramp_progress = self._ramp_progress(round_number)
		scores = self.first_date_scores
		if len(scores) < MIN_SCORES_FOR_QUANTILE:
			fraction = FALLBACK_THRESHOLD_FRACTION * min(1.0, self.rounds / FULL_FALLBACK_ROUNDS)
			value = fraction * (1 - ramp_progress) * self.bonus
		else:
			quantile = (
				EARLY_THRESHOLD_QUANTILE
				+ (LATE_THRESHOLD_QUANTILE - EARLY_THRESHOLD_QUANTILE) * ramp_progress
			)

			position = min(int(quantile * len(scores)), len(scores) - 1)
			value = scores[position]
		return value

	def _mirrored_value(self, seen: list[float]) -> float:
		"""Running mean of that half, or a prior if we have no data yet."""
		if seen:
			return sum(seen) / len(seen)
		return MIRROR_PRIOR_FRACTION * self.bonus

	def _ad_penalty(self, advertisement: Advertisement) -> float:
		"""Lower is better: expected penalty implied by their ad alone."""
		my_penalty = self._dim_penalty(
			advertisement.own_dim, self.ideal[advertisement.own_dim], advertisement.own_value
		)
		their_penalty = self._dim_penalty(
			advertisement.seek_dim, advertisement.seek_value, self.profile[advertisement.seek_dim]
		)
		return my_penalty + their_penalty / 2

	def _expected_score(self, partner: PlayerProfile) -> float:
		"""Happiness if we married them, with mirrored halves replaced."""
		observed = partner.projected_profile
		stated_ideal = partner.stated_ideal

		mine_from_own = False
		if tuple(observed) == tuple(self.ideal):
			if partner.last_informative_mine is not None:
				mine = partner.last_informative_mine
				mine_from_own = True
			else:
				mine = self._mirrored_value(self._mine_seen)
			mine_mirrored = True
		else:
			mine = decrement(self.ideal, observed, self.monotonic)
			mine_mirrored = False

		theirs_from_own = False
		shown = self._last_projected.get(partner.index)
		if tuple(stated_ideal) == tuple(self.profile) or (
			shown is not None and tuple(stated_ideal) == tuple(shown)
		):
			if partner.last_informative_theirs is not None:
				theirs = partner.last_informative_theirs
				theirs_from_own = True
			else:
				theirs = self._mirrored_value(self._theirs_seen)
			theirs_mirrored = True
		else:
			theirs = decrement(stated_ideal, self.profile, self.monotonic)
			theirs_mirrored = False
		if not mine_mirrored:
			self._mine_seen.append(float(mine))
			partner.last_informative_mine = float(mine)
		if not theirs_mirrored:
			self._theirs_seen.append(float(theirs))
			partner.last_informative_theirs = float(theirs)
		total_penalty = mine + theirs / 2
		if partner.is_untruthful and not (mine_from_own and theirs_from_own):
			total_penalty *= UNTRUTHFUL_PENALTY_MULTIPLIER
		return self.bonus - total_penalty

	def _check_truthfulness(
		self, partner: PlayerProfile, observed: Profile, stated_ideal: Profile
	) -> None:
		"""Flag them if this date contradicts their ad or their previous date."""
		ad = partner.advertisement
		contradicts_ad = ad is not None and observed[ad.own_dim] != ad.own_value
		changed_since_last_date = partner.projected_profile is not None and (
			observed != partner.projected_profile or stated_ideal != partner.stated_ideal
		)
		if contradicts_ad or changed_since_last_date:
			partner.is_untruthful = True

	def advertise(self) -> Advertisement:
		"""Advertise our most widely appealing trait and least demanding seek."""

		def own_appeal(dim: int) -> int:

			value = self.profile[dim]
			return value if self._is_monotonic(dim) else self._centeredness(value)

		def seek_leniency(dim: int) -> int:

			value = self.ideal[dim]
			return MAX_VALUE - value if self._is_monotonic(dim) else self._centeredness(value)

		own_dim = max(range(self.dims), key=own_appeal)
		seek_dim = max(range(self.dims), key=seek_leniency)
		return Advertisement(
			own_dim=own_dim,
			own_value=self._mask_value(own_dim, self.profile[own_dim]),
			seek_dim=seek_dim,
			seek_value=self.ideal[seek_dim],
		)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		threshold = self._threshold_score(turn.round)
		promising_dated: list[PlayerProfile] = []
		second_chance: list[PlayerProfile] = []
		never_dated: list[PlayerProfile] = []

		for candidate in candidates:
			partner = self._get_profile(candidate.index)
			partner.advertisement = candidate.ad
			if partner.is_in_cooldown(turn.round):
				continue
			if not partner.has_dated:
				never_dated.append(partner)
			elif partner.expected_score >= threshold or partner.accept_next_proposal_chance:
				promising_dated.append(partner)
			elif partner.expected_score <= 0 and candidate.dates < 2:
				second_chance.append(partner)

		promising_dated.sort(key=lambda partner: -partner.expected_score)
		second_chance.sort(key=lambda partner: -partner.expected_score)
		self.rng.shuffle(never_dated)
		never_dated.sort(key=lambda partner: self._ad_penalty(partner.advertisement))

		if turn.rounds_left <= ENDGAME_ROUNDS:
			ranked = [partner.index for partner in promising_dated]
		else:
			ranked = [partner.index for partner in promising_dated + second_chance + never_dated]
		return ranked[: self.list_limit]

	def project(self, date: DateContext) -> Profile:
		"""Show our masked profile only."""
		result = self._masked_profile()
		self._last_projected[date.partner] = result
		return result

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

		if partner.expected_score <= 0:
			if date.date_number >= 2:
				return Decision(incompatible=True)
			return Decision()

		threshold = self._threshold_score(date.round)

		safety = (
			SAFETY_MARGIN_FRACTION
			* self.bonus
			* (1 - SAFETY_MARGIN_DECAY * self._ramp_progress(date.round))
		)
		floor = max(SCORE_FLOOR_FRACTION * self.bonus, SCORE_FLOOR_PER_DIM * self.dims)
		clears_threshold = partner.expected_score >= threshold
		clears_with_safety = partner.expected_score >= threshold + safety
		can_propose = date.date_number >= 2

		may_ask_again = not partner.is_in_cooldown(date.round)

		if partner.accept_next_proposal_chance:
			shortfall = threshold - partner.expected_score
			if shortfall > ACCEPT_PROPOSAL_MARGIN_FRACTION * self.bonus:
				partner.accept_next_proposal_chance = False

		if (
			can_propose
			and partner.expected_score >= floor
			and (clears_with_safety or partner.accept_next_proposal_chance)
		):
			date_record.we_proposed = True
			return Decision(again=may_ask_again, propose=True)
		if clears_threshold:
			return Decision(again=True)

		return Decision()

	def on_outcome(self, outcome: DateOutcome) -> None:
		partner = self._get_profile(outcome.partner)
		if outcome.married or not partner.dating_history:
			return
		date_record = partner.dating_history[-1]
		date_record.they_proposed = outcome.partner_proposed

		if date_record.we_proposed:
			partner.failed_proposals_in_a_row += 1
			partner.accept_next_proposal_chance = False
			if partner.failed_proposals_in_a_row >= FAILED_PROPOSALS_BEFORE_COOLDOWN:
				partner.cooldown_last_round = outcome.round + self.COOLDOWN_ROUNDS
				partner.failed_proposals_in_a_row = 0

		if outcome.partner_proposed and not date_record.we_proposed:
			shortfall = self._threshold_score(outcome.round) - date_record.expected_score
			if (
				date_record.expected_score
				>= max(SCORE_FLOOR_FRACTION * self.bonus, SCORE_FLOOR_PER_DIM * self.dims)
				and shortfall <= ACCEPT_PROPOSAL_MARGIN_FRACTION * self.bonus
			):
				partner.accept_next_proposal_chance = True
