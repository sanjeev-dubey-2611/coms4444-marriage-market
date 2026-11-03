import math

from models.player import (
	Advertisement,
	Candidate,
	DateContext,
	Decision,
	GameContext,
	PlayerSnapshot,
	RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import Profile, decrement

OBSERVE_FRACTION = 0.37


class Player2(BasePlayer):
	"""Honest player that applies the 37% (secretary) rule.

	Advertises, projects and describes itself honestly, and assumes everyone
	else does the same. Its ad shows its best monotonic characteristic (highest
	value) and the monotonic characteristic it wants most.

	For the first 37% of rounds it only observes: it dates new people, scores
	each by the handout's happiness formula, and remembers the best score. After
	that it goes back to anyone (including people met during observation) whose
	score is at least that best and positive, asks them again, and proposes from
	the second date on.

  Note: We may want to shorten this period of observing or make it dynamic as our best matches might get married before we decide to start proposing.
  How can we tell when someone is good for us?
  Can we see if/when others get married? Maybe also getting proposed to can influence our decision.
  Maybe consider their type and see how well we match each other's types (since we also take 1/2 of their penalty)
  Maybe a heap or something to sort who we decide to date again, could keep more than just best score as, if they get married we are looking at no one.
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)
		self.cutoff = math.ceil(OBSERVE_FRACTION * self.rounds)
		self.scores: dict[int, float] = {}
		self.best_observed = 0.0

	def advertise(self) -> Advertisement:
		showable = range(self.monotonic) if self.monotonic else range(self.dims)
		own = max(showable, key=lambda i: self.profile[i])
		seek = max(showable, key=lambda i: self.ideal[i])
		return Advertisement(
			own_dim=own, own_value=self.profile[own], seek_dim=seek, seek_value=self.ideal[seek]
		)

	def _good_enough(self, score: float) -> bool:
		return score > 0 and score >= self.best_observed

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		if turn.round <= self.cutoff:
			chosen = [c.index for c in candidates if c.index not in self.scores]
			self.rng.shuffle(chosen)
			return chosen[: self.list_limit]
		known = sorted(
			(c for c in candidates if self._good_enough(self.scores.get(c.index, 0.0))),
			key=lambda c: (-self.scores[c.index], c.index),
		)
		fresh = [c.index for c in candidates if c.index not in self.scores]
		self.rng.shuffle(fresh)
		return ([c.index for c in known] + fresh)[: self.list_limit]

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		mine = decrement(self.ideal, observed, self.monotonic)
		theirs = decrement(stated_ideal, self.profile, self.monotonic)
		score = self.bonus - mine - theirs / 2
		self.scores[date.partner] = score

		if score <= 0:
			return Decision(incompatible=True)

		if date.round <= self.cutoff:
			self.best_observed = max(self.best_observed, score)
			return Decision()

		if not self._good_enough(score):
			return Decision()
		return Decision(again=True, propose=date.date_number >= 2)
