"""Experimental Player 6 variants. Each changes exactly one thing."""

from models.player import Candidate, DateContext, DateOutcome, Decision, RoundContext
from models.profile import Profile
from players.player_6.player import Player6


class AdRank(Player6):
	"""Idea 1a: during exploration, list the best-looking people by ad. No randomness."""

	def _explore_key(self, candidate: Candidate) -> float:
		return self._candidate_penalty(candidate)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self._record_ad_estimates(candidates)
		if turn.round > self.exploration_rounds:
			return super().choose_dates(candidates, turn)
		# Still meet new people first; rank by estimate within each date count.
		choices = sorted(candidates, key=lambda c: (c.dates, self._explore_key(c), c.index))
		return [c.index for c in choices[: self.list_limit]]


class AdRankNoisy(AdRank):
	"""Idea 1b: same, but each estimate is multiplied by a random factor in [0.5, 1.5]."""

	def _explore_key(self, candidate: Candidate) -> float:
		return self._candidate_penalty(candidate) * self.rng.uniform(0.5, 1.5)


class StrongAgain(Player6):
	"""Idea 2: a match with penalty <= 0.5 B skips exploration (again, and propose from date 2)."""

	strong_ratio = 0.5

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		if date.round <= self.exploration_rounds:
			self._remember(date.partner, observed, stated_ideal)
			if self._penalty(observed, stated_ideal) <= self.strong_ratio * self.bonus:
				return Decision(again=True, propose=date.date_number >= 2)
			return Decision()
		return super().decide(date, observed, stated_ideal)


class PreferKnown(Player6):
	"""Idea 3: after exploration, favour acceptable people already dated (more so late)
	and people who proposed to us, so they always make the capped list."""

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.proposed_to_me: set[int] = set()

	def on_outcome(self, outcome: DateOutcome) -> None:
		if outcome.partner_proposed:
			self.proposed_to_me.add(outcome.partner)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		self._record_ad_estimates(candidates)
		if turn.round <= self.exploration_rounds:
			return super().choose_dates(candidates, turn)
		lateness = 1 - turn.rounds_left / self.rounds

		def key(c: Candidate) -> float:
			score = self._candidate_penalty(c)
			if c.index in self.observations:
				score -= 0.25 * self.bonus * lateness
			if c.index in self.proposed_to_me:
				score -= 0.25 * self.bonus
			return score

		choices = [
			c
			for c in sorted(candidates, key=lambda c: (key(c), c.index))
			if c.index not in self.observations
			or self._acceptable(self._penalty(*self.observations[c.index]))
		]
		return [c.index for c in choices[: self.list_limit]]


VARIANTS = {
	'base': Player6,
	'ad_rank': AdRank,
	'ad_rank_noisy': AdRankNoisy,
	'strong_again': StrongAgain,
	'prefer_known': PreferKnown,
}
