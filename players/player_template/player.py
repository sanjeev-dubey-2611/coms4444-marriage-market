"""Starting point for a group's player.

Copy this whole directory to ``players/player_<k>/`` using your group number,
then rename the class to ``Player<k>``. Group 4 would end up with
``players/player_4/player.py`` containing ``class Player4``. The registry looks
for exactly that; nothing else needs editing.

Keep the ``__init__.py``. Discovery uses ``pkgutil.iter_modules``, which only
reports directories that have one, so a group directory without it is silently
invisible to the simulator - no error, just a player that never turns up.

Optional: edit ``distribution.json`` in the same directory. That file is your
group's one-dimension trait distribution - exactly 11 nonnegative
probabilities for values 0 through 10, summing to 1. Use JSON:
``{"P": [...]}`` or a bare list of 11 numbers. Staff may assign your row (or
a preset) to individual profile dimensions when they run the simulator; you
do not choose the dimension and you do not submit a full d x 11 table. See
the README section "Submitting a distribution".

This directory is not itself discovered - the registry only matches
``player_<digits>`` - so the template can never appear in a run as a competitor.
"""

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
from models.profile import Profile


class PlayerTemplate(BasePlayer):
	"""Rename me to Player<k>, where <k> is your group number.

	One instance of this class is one person. Your code may be used for several
	people in the same simulation - perhaps all of them - with one instance per
	person. Do not share state between instances (class attributes, module
	globals, files): that is exactly the kind of bypass the handout forbids.
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
		super().__init__(snapshot, ctx)

		# super() has already set these from ctx and snapshot:
		#
		#   self.index      which person you are (0-based); others refer to you by it
		#   self.id         your UUID, stable for the whole simulation
		#   self.profile    your true profile T_p, a tuple of d ints in 0-10
		#   self.ideal      your desired profile D_p, a tuple of d ints in 0-10
		#   self.people     N, how many people are in the simulation
		#   self.dims       d, the number of dimensions
		#   self.monotonic  m, the first m dimensions are monotonic
		#   self.rounds     t, how many rounds of dating there are
		#   self.bonus      B, the marriage bonus
		#   self.rng        private seeded random generator; use instead of random
		#   self.list_limit maximum returned names, or None if disabled
		#
		# The engine constructs you once, before round 1. Anything you want to
		# carry between rounds lives on self, so initialise it here.
		self.history: dict[int, list[Profile]] = {}

	def advertise(self) -> Advertisement:
		"""Called once, before round 1. Fixed for the whole simulation.

		Post one of your own characteristics and one characteristic you are
		looking for. Neither has to be true.
		"""
		return Advertisement(
			own_dim=0, own_value=self.profile[0], seek_dim=0, seek_value=self.ideal[0]
		)

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		"""Name the people you would like to date this round.

		Called once per round while you are single, unless you are already on a
		repeat date because you and your last date both asked for one.

		WHAT YOU CAN SEE

			candidate.index         who they are, stable for the whole run
			candidate.ad            their advertisement, possibly a lie
			candidate.dates         how many times you two have dated so far
			turn.round              this round, 1-based
			turn.rounds_left        rounds remaining, including this one
			turn.singles            how many people are still single

		HOW PAIRING WORKS

			1. Pairs who both asked for another date are paired first.
			2. Pairs who listed each other are paired at random.
			3. Pairs where only one listed the other are paired at random.
			4. Everyone left is paired at random.

		Nobody is ever paired with someone either side marked incompatible, so
		in rare cases you sit a round out.
		"""
		return [c.index for c in candidates[: self.list_limit]]

	def project(self, date: DateContext) -> Profile:
		"""The profile your behaviour projects on this date.

		Your date senses exactly what you return here. It does not have to be
		your true profile, but your final happiness is scored on true profiles
		only - and you also lose half of whatever your partner loses.
		"""
		return self.profile

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		"""At the end of the date, describe the ideal partner you are seeking.

		``observed`` is what your date projected. Honest or not, it is up to you.
		"""
		return self.ideal

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		"""After the date: ask again, propose, or mark them incompatible.

		``observed`` is what they projected; ``stated_ideal`` is the ideal
		partner they described. ``date.date_number`` counts this date.

		RULES

			propose       only legal when date.date_number >= 2
			again         if you both ask, you are paired first next round
			incompatible  you will never be paired with them again

		If you both propose on the same date you are married and leave the pool
		for good. incompatible cannot be combined with again or propose.

		IF YOU GET IT WRONG

		An exception, a malformed return value, or taking longer than the
		--timeout budget is recorded as a fault and the engine substitutes a
		safe default: your true profile for project, your true ideal for
		describe_ideal, nobody for choose_dates, and Decision() (no again, no
		proposal, no incompatibility) for decide. Your failure never affects the
		other groups.
		"""
		self.history.setdefault(date.partner, []).append(observed)
		return Decision(again=False, propose=False, incompatible=False)

	def on_outcome(self, outcome: DateOutcome) -> None:
		"""Told what happened after both of you decided.

		outcome.partner_proposed  they proposed to you this round
		outcome.married           you are now married (to them)
		outcome.again             you both asked; you meet again next round
		"""
		return None
