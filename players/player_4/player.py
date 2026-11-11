"""Team Player 4 (Group 4), final revised player for COMS 4444 Project 3.

A careful matchmaker. It dates the best-looking ads first, keeps a bar for
proposing that starts high and falls as the rounds run out, and never
proposes on a score at or below its floor.

Design:
- First dates are ranked by ad penalty (our full penalty plus half of
  theirs, matching the payoff).
- The propose bar is a falling quantile of first-date scores, from the 90th
  percentile early to the minimum late, flat for the first and last 10
  percent of rounds. It is not clamped at 0, so it tracks the actual score
  distribution even when scores are mostly negative.
- A safety margin above the bar is required to propose: 0.10 * bonus early,
  decaying to 0.02 * bonus late, so the player is picky while there is time
  to be picky and less picky as rounds run out.
- A hard floor on the estimated score for proposing: the larger of 0.05 *
  bonus and 1.0 point per dimension. The floor does not decay. Estimation
  error from a flattering projection does not shrink with the bonus; it
  scales with the number of dimensions, so a purely fractional floor is far
  too small at low bonus. The floor applies to the propose path and to the
  fast-lane flag.
- Mirror check: if a partner shows our ideal back, or states our own
  profile or our own projection as their ideal, that half of the estimate
  carries no information. It is replaced by that partner's own last
  informative value for that half if one exists, else by the running mean
  of that half across partners dated. Informative halves feed the running
  means; mirrored halves do not.
- Lie check: a partner who contradicts their ad or changes their shown
  traits between dates is flagged, and their penalty is multiplied by 1.2.
  The multiplier is skipped when both halves of the estimate come from
  that partner's own past informative data, because that data is
  trustworthy (true traits do not change).
- Second look: a score of 0 or below on date 1 gets a second look, not a
  burn; only a second bad date marks a partner incompatible.
- Fast lane: if a partner proposed to us and was close to the bar, we
  accept their next proposal when the score clears the floor.
- Cooldown: after 2 failed proposals in a row, a partner is excluded from
  our list for 5 rounds (we still propose if we meet them by chance and
  they qualify).
- Endgame: in the last 3 rounds, only known good partners are listed.
- Presentation: a masked profile (regular dims clamped to [2, 8],
  monotonic dims truthful), an ad that matches what is shown, and the true
  ideal stated honestly. No inflated projection.

Two candidate fixes were considered and deliberately NOT made:
- Using the minimum score across dates was dropped: it would fight the
  second look (a bad first date would keep burning a partner on date 2)
  and it penalises partners whose projections legitimately improve.
- Capping the threshold so it never rises was dropped: a rising bar after
  meeting great partners is correct reservation logic, not a flaw.
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
# Kept at 104's [2, 8], not narrowed to [1, 9] as in 105.
REGULAR_PROJECTION_LOW = 2
REGULAR_PROJECTION_HIGH = 8

# Threshold quantile of first-date scores at the two ends of the run.
# Kept at 104's 90th percentile, not softened to 80th as in 105.
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
# Kept at 104's 5 rounds after 2 failed proposals, not shortened to 3
# after 1 as in 105.
FAILED_PROPOSALS_BEFORE_COOLDOWN = 2
# How far below the threshold (as a fraction of the bonus) we still accept a proposal.
ACCEPT_PROPOSAL_MARGIN_FRACTION = 0.1
# Safety margin above the threshold required to propose. Raised from 0.05
# to 0.10 * bonus for 107: the 10-scenario sweep showed the estimate itself
# is wrong at 5 dims and at low bonus (a stable flattering projection), so
# a bigger margin absorbs more estimation error. The margin decays late in
# the run (see SAFETY_MARGIN_DECAY) so the player does not strand a match.
SAFETY_MARGIN_FRACTION = 0.10
# Fraction of the safety margin lost by the end of the run. The margin goes
# from 0.10 * bonus early to 0.02 * bonus late: picky while there is time
# to be picky, less picky as rounds run out. The score floor does not decay.
SAFETY_MARGIN_DECAY = 0.8
# Hard floor on the estimated score for proposing (revision 10): never
# propose on a score at or below this fraction of the bonus, even via the
# fast lane. A barely positive estimate can hide a negative true score
# (the partner's projection can flatter them), so the floor is positive,
# not 0. The threshold path already clears this floor whenever the
# threshold is non-negative; the fast lane is the path that needed it.
SCORE_FLOOR_FRACTION = 0.05
# Absolute floor per dimension. The estimation error from a
# flattering projection does NOT shrink with the bonus: it scales with the
# number of dimensions, roughly a few points per dim. At bonus 20 the
# fractional floor is only 1.0, far below the typical error (4-12 points
# at 5 dims), so the player proposed on estimates of 1-10 that were truly
# negative. The floor is now the larger of the fractional floor and one
# point per dimension, so at bonus 20 and 5 dims it is 5.0, not 1.0.
SCORE_FLOOR_PER_DIM = 1.0
# Last rounds in which only known good partners are listed (revision 6).
ENDGAME_ROUNDS = 3
# Prior penalty for a mirrored half before we have a running mean (revision 2).
MIRROR_PRIOR_FRACTION = 0.25
# Shrinkage for a mirrored half once we have a running mean (107 fix 1).
# A mirrored half carries no information, so the honest guess is the
# population mean of that half, but the mean can be very pessimistic at
# high dims (large penalties), which blocked proposing entirely at d=10.
# We shrink it halfway toward trusting the mirror (0). With no data yet
# the prior above is used unshrunk.
MIRROR_SHRINKAGE = 1.0


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
	# Last informative half for this partner, from a date when that half
	# was not mirrored (110). Used instead of the population mean when
	# the partner mirrors on a later date.
	last_informative_mine: float | None = None
	last_informative_theirs: float | None = None
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
	"""Final revised Player 4: patient bar, safety margin, and a hard floor."""

	COOLDOWN_ROUNDS = 5

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.player_profiles: dict[int, PlayerProfile] = {}
		# Expected score from every first date, kept sorted for the threshold quantile.
		# Dated players below the threshold stay in player_profiles and come back
		# into choose_dates once it drops.
		self.first_date_scores: list[float] = []
		# Informative halves of the penalty, for the mirror fallback (revision 2).
		self._mine_seen: list[float] = []
		self._theirs_seen: list[float] = []
		# What we actually projected to each partner on their latest date.
		self._last_projected: dict[int, Profile] = {}

	def _get_profile(self, index: int) -> PlayerProfile:
		if index not in self.player_profiles:
			self.player_profiles[index] = PlayerProfile(index=index)
		return self.player_profiles[index]

	def _is_monotonic(self, dim: int) -> bool:
		return dim < self.monotonic

	def _mask_value(self, dim: int, value: int) -> int:
		"""Hide extremes: pull regular values into [1, 9].

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
		percentile to the minimum along the ramp. Scores scale with B, so the
		threshold does too. No scarcity factor: 104 has none, and the hybrid
		drops 105's. For 107 the quantile is NOT clamped at 0: at high dims
		the first-date scores are mostly below 0, and clamping the bar at 0
		meant nothing ever cleared it, so the player married nobody at d=10.
		The bar now tracks the actual score distribution. The propose floor
		(0.05 * bonus) is separate and still blocks any score at or below it.
		"""
		ramp_progress = self._ramp_progress(round_number)
		scores = self.first_date_scores
		if len(scores) < MIN_SCORES_FOR_QUANTILE:
			# Too few dates to trust a quantile: falling fraction of the bonus,
			# smaller in short runs (5 rounds -> a quarter of the usual fraction).
			fraction = FALLBACK_THRESHOLD_FRACTION * min(1.0, self.rounds / FULL_FALLBACK_ROUNDS)
			value = fraction * (1 - ramp_progress) * self.bonus
		else:
			quantile = (
				EARLY_THRESHOLD_QUANTILE
				+ (LATE_THRESHOLD_QUANTILE - EARLY_THRESHOLD_QUANTILE) * ramp_progress
			)
			# Index of that quantile in the sorted scores, clamped to the last one.
			position = min(int(quantile * len(scores)), len(scores) - 1)
			value = scores[position]
		return value

	def _mirrored_value(self, seen: list[float]) -> float:
		"""Fallback penalty for a mirrored half of the estimate.

		A mirrored half carries no information about their true traits, so
		the honest guess is the running mean of that half across partners
		dated. 107 shrank that mean halfway toward trusting the mirror (0),
		but the high-dims dig showed the shrinkage inflates estimates (0.0
		estimated for a true -44.0) and does not fix the d=10 zero, so 109
		uses the full mean (MIRROR_SHRINKAGE = 1.0). With no data yet, use
		the prior (0.25 * bonus) unshrunk.
		"""
		if seen:
			return MIRROR_SHRINKAGE * (sum(seen) / len(seen))
		return MIRROR_PRIOR_FRACTION * self.bonus

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
		"""Happiness if we married them, with mirrored halves replaced.

		A partner who shows our ideal back, or states our own profile or our
		own projection as their ideal, is mirroring: that half of the estimate
		carries no information about their true traits. It is replaced by
		this partner's own last informative value for that half (from a date
		when they showed their real traits), falling back to the running
		mean of that half across partners dated (see _mirrored_value) only
		if the partner never showed an informative value. Informative halves
		are recorded for the running means and as the partner's last
		informative value; mirrored halves are not.
		"""
		observed = partner.projected_profile
		stated_ideal = partner.stated_ideal
		# Mine: their shown profile against our ideal.
		mine_from_own = False
		if tuple(observed) == tuple(self.ideal):
			# Mirrored: prefer this partner's own last informative mine
			# (from a date when they showed their real profile) over the
			# population mean. A good match that starts mirroring keeps its
			# own good value instead of being punished with the average.
			if partner.last_informative_mine is not None:
				mine = partner.last_informative_mine
				mine_from_own = True
			else:
				mine = self._mirrored_value(self._mine_seen)
			mine_mirrored = True
		else:
			mine = decrement(self.ideal, observed, self.monotonic)
			mine_mirrored = False
		# Theirs: their stated ideal against our profile.
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
			# Inflate the penalty, not shrink the score: the discount then means the
			# same at any B and never pushes a negative score up towards 0.
			# Skipped when both halves come from this partner's own past
			# informative data: that data is trustworthy (their true traits do
			# not change), so the multiplier would double-penalise a good
			# match that merely changed what it shows.
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
		second_chance: list[PlayerProfile] = []
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
			elif partner.expected_score <= 0 and candidate.dates < 2:
				# One bad first date: a second look, not a burn (revision 4).
				second_chance.append(partner)
			# Else below threshold but positive: not listed, wait for the threshold to drop.

		promising_dated.sort(key=lambda partner: -partner.expected_score)  # best first
		second_chance.sort(key=lambda partner: -partner.expected_score)  # least-bad first
		self.rng.shuffle(never_dated)  # random tie-break for equal ad penalties
		never_dated.sort(key=lambda partner: self._ad_penalty(partner.advertisement))

		if turn.rounds_left <= ENDGAME_ROUNDS:
			# Revision 6: a fresh date this late cannot lead to a proposal.
			ranked = [partner.index for partner in promising_dated]
		else:
			ranked = [partner.index for partner in promising_dated + second_chance + never_dated]
		return ranked[: self.list_limit]  # slicing by None keeps everything

	def project(self, date: DateContext) -> Profile:
		"""Show our masked profile only.

		No +2 on the partner's monotonic seek dim (removed in revision 11):
		it made a partner look better than they are, inflating our estimate
		of them and marrying us into a negative true score.
		"""
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

		# Marrying them would not beat staying single (0). A first bad date
		# gets a second look (revision 4); only a second bad date burns them.
		if partner.expected_score <= 0:
			if date.date_number >= 2:
				return Decision(incompatible=True)
			return Decision()

		threshold = self._threshold_score(date.round)
		# 107 fix 3: the safety margin decays late in the run, from
		# 0.10 * bonus early to 0.02 * bonus late, so the player gets less
		# picky as rounds run out instead of stranding a match. The floor
		# does not decay, so the no-negative-or-zero rule still holds late.
		safety = (
			SAFETY_MARGIN_FRACTION
			* self.bonus
			* (1 - SAFETY_MARGIN_DECAY * self._ramp_progress(date.round))
		)
		floor = max(SCORE_FLOOR_FRACTION * self.bonus, SCORE_FLOOR_PER_DIM * self.dims)
		clears_threshold = partner.expected_score >= threshold
		clears_with_safety = partner.expected_score >= threshold + safety
		can_propose = date.date_number >= 2
		# During cooldown we still propose if they qualify, but never ask for more dates.
		may_ask_again = not partner.is_in_cooldown(date.round)

		# Revision 9: re-check a stale fast-lane flag before using it.
		if partner.accept_next_proposal_chance:
			shortfall = threshold - partner.expected_score
			if shortfall > ACCEPT_PROPOSAL_MARGIN_FRACTION * self.bonus:
				partner.accept_next_proposal_chance = False

		# Revision 10: never propose on a score at or below the floor, even
		# via the fast lane. A barely positive estimate can hide a negative
		# true score.
		if can_propose and partner.expected_score >= floor and (
			clears_with_safety or partner.accept_next_proposal_chance
		):
			date_record.we_proposed = True
			return Decision(again=may_ask_again, propose=True)
		if clears_threshold:
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
		# Revision 10: only set the fast lane if the score clears the floor,
		# so a flag is never set for a score we would refuse to propose on.
		if outcome.partner_proposed and not date_record.we_proposed:
			shortfall = self._threshold_score(outcome.round) - date_record.expected_score
			if (
				date_record.expected_score
				>= max(SCORE_FLOOR_FRACTION * self.bonus, SCORE_FLOOR_PER_DIM * self.dims)
				and shortfall <= ACCEPT_PROPOSAL_MARGIN_FRACTION * self.bonus
			):
				partner.accept_next_proposal_chance = True
