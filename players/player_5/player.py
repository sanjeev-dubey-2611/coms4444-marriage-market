"""Starting point for a group's player.

Copy this whole directory to ``players/player_<k>/`` using your group number,
then rename the class to ``Player<k>``. Group 4 would end up with
``players/player_4/player.py`` containing ``class Player4``. The registry looks
for exactly that; nothing else needs editing.

Keep the ``__init__.py``. Discovery uses ``pkgutil.iter_modules``, which only
reports directories that have one, so a group directory without it is silently
invisible to the simulator - no error, just a player that never turns up.

This directory is not itself discovered - the registry only matches
``player_<digits>`` - so the template can never appear in a run as a competitor.
"""
import math
import statistics
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
from models.profile import Profile, happiness

class DatedPartner:
	def __init__(self):
		self.profile, self.ideal, self.advertisement = None, None, None
		self.profile_history, self.ideal_history, self.advertisement_history = [], [], []
		self.date_requested_count = 0
		self.date_accepted_count = 0
		self.marriage_requested_count = 0

	def update_information(self, profile: Profile, ideal: Profile, advertisement: Advertisement):
		self.profile_history.append(profile)
		self.ideal_history.append(ideal)
		self.advertisement_history.append(advertisement)
		self.date_accepted_count = self.date_accepted_count + 1
		self.profile = self.profile_history[-1]
		self.ideal = self.ideal_history[-1]
		self.advertisement = self.advertisement_history[-1]

	def update_request_counts(self, date_again, propose_again):
		self.date_requested_count = self.date_requested_count + int(date_again)
		self.marriage_requested_count = self.marriage_requested_count + int(propose_again)

	def __str__(self):
		aggregated_details = "\nprofile: {}, ideal:{}, advertisement:{}".format(
			self.profile, self.ideal, self.advertisement
		)
		history_details = "profile_history:{}, ideal_history:{}, advertisement_history:{}".format(
			self.profile_history, self.ideal_history, self.advertisement_history
		)
		interaction_details = "date_accepted_count:{}, date_requested_count:{}, marriage_requested_count:{}".format(
			self.date_accepted_count, self.date_requested_count, self.marriage_requested_count
		)
		return "\n".join([aggregated_details, history_details, interaction_details])


class Player5(BasePlayer):
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
		# carry between rounds lives on self, so initialize it here.
		self.dated_partner_history: dict[int, DatedPartner] = {}
		self.exploration_time_fraction = 0.2
		self.exploration_people_fraction = 0.2
		self.data_sanity_count = 1
		self.exploration_mode = True
		self.random_match_rate = math.ceil(self.rounds / self.people)
		self.max_date_requests_per_person = 3 * self.random_match_rate
		self.max_marriage_requests_per_person = 3 * self.random_match_rate
		self.initial_k_std_away = 0.5

	def advertise(self) -> Advertisement:
		"""Called once, before round 1. Fixed for the whole simulation.

		Post one of your own characteristics and one characteristic you are
		looking for. Neither has to be true.
		"""
		# If there are no monotonic dimensions, use dimension 0 for both traits.
		own_dim = max(range(self.monotonic), key=lambda i: self.profile[i]) if self.monotonic else 0
		seek_dim = min(range(self.monotonic), key=lambda i: self.ideal[i]) if self.monotonic else 0
		return Advertisement(
			own_dim=own_dim,
			own_value=self.profile[own_dim],
			seek_dim=seek_dim,
			seek_value=0,
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

		print()
		print("self.rounds", self.rounds, "self.people", self.people)
		print("Turn:", turn)
		time_criteria = turn.round / self.rounds < self.exploration_time_fraction
		people_criteria = (self.people - turn.singles) / self.people < self.exploration_people_fraction
		self.exploration_mode = time_criteria and people_criteria
		self.mark_partners_out_of_system(candidates)

		happiness_scores = dict(sorted(self.get_dated_partners_vs_happiness().items(), key=lambda item: item[1], reverse=True))
		avg_happiness, std_happiness = 0, 0
		if len(happiness_scores) > self.data_sanity_count:
			avg_happiness, std_happiness = statistics.mean(happiness_scores.values()), statistics.stdev(happiness_scores.values())
			avg_happiness, std_happiness = round(avg_happiness, 2), round(std_happiness, 2)
		good_partners = [p_idx for p_idx, p_score in happiness_scores.items() if p_score >= avg_happiness and p_score > 0]
		unexplored_partners = [c.index for c in sorted(candidates, key=lambda x: x.dates, reverse=False)]

		if self.exploration_mode:
			print("[exploration mode]", "time_criteria:", time_criteria, "people_criteria:", people_criteria)
			chosen = unexplored_partners
			chosen = [x for x in chosen if x != self.index]
		else:
			print("[exploitation mode]", "time_criteria:", time_criteria, "people_criteria:", people_criteria)
			chosen = good_partners
			chosen = [x for x in chosen if x != self.index]

		print("chosen:", chosen[: self.list_limit])
		return chosen[: self.list_limit]

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
		print("current date details:", "id:", date.partner, "ad:", date.partner_ad, "date_number:", date.date_number)
		if date.partner not in self.dated_partner_history:
			self.dated_partner_history[date.partner] = DatedPartner()
		self.dated_partner_history[date.partner].update_information(observed, stated_ideal, date.partner_ad)
		happiness_scores = self.get_dated_partners_vs_happiness()
		print("happiness_scores:", happiness_scores)
		current_happiness = happiness_scores[date.partner]
		avg_happiness, std_happiness = 0, 0
		if len(happiness_scores) > self.data_sanity_count:
			avg_happiness, std_happiness = statistics.mean(happiness_scores.values()), statistics.stdev(happiness_scores.values())
			avg_happiness, std_happiness = round(avg_happiness, 2), round(std_happiness, 2)
		progress = (date.round - 1) / max(1, self.rounds - 1)
		k_std_away = self.initial_k_std_away * (1 - progress)
		print(
			"current_happiness:", current_happiness,
			"date_again:", avg_happiness,
			"k_std_away:", k_std_away,
			"propose_again:", avg_happiness + k_std_away * std_happiness,
			"never_again:", avg_happiness - 1.5 * std_happiness,
			"Date History:", self.dated_partner_history[date.partner],
			"max_marriage_requests_per_person", self.max_marriage_requests_per_person,
			"max_date_requests_per_person", self.max_date_requests_per_person,
		)
		date_again = (
			current_happiness >= avg_happiness and
			current_happiness > 0 and
			self.dated_partner_history[date.partner].date_requested_count < self.max_date_requests_per_person
		)
		propose_again = (
			date.date_number >= 2 and
			current_happiness >= (avg_happiness + k_std_away * std_happiness) and
			current_happiness > 0 and
			self.dated_partner_history[date.partner].marriage_requested_count < self.max_marriage_requests_per_person
		)
		never_again = (
			current_happiness < (avg_happiness - 1.5 * std_happiness) or
			self.dated_partner_history[date.partner].date_requested_count >= self.max_date_requests_per_person or
			self.dated_partner_history[date.partner].marriage_requested_count >= self.max_marriage_requests_per_person
		)
		if self.exploration_mode:
			date_again, propose_again, never_again = False, False, False
		if never_again:
			date_again, propose_again = False, False
		self.dated_partner_history[date.partner].update_request_counts(date_again, propose_again)
		print("date_again:", date_again, "propose_again:", propose_again, "never_again:", never_again, "\n\n")
		return Decision(
			again=date_again,
			propose=propose_again,
			incompatible=never_again
		)

	def get_dated_partners_vs_happiness(self):
		happiness_scores = dict()
		for partner_id, partner_history in self.dated_partner_history.items():
			partner_scores = list()
			for partner_profile, partner_ideal in zip(partner_history.profile_history, partner_history.ideal_history):
				score = happiness(self.bonus, self.profile, self.ideal, partner_profile, partner_ideal, self.monotonic)
				partner_scores.append(score)
			happiness_scores[partner_id] = min(partner_scores)
		return happiness_scores

	def mark_partners_out_of_system(self, candidates):
		available_partners = [x.index for x in candidates]
		self.dated_partner_history = {
			partner_id: partner
			for partner_id, partner in self.dated_partner_history.items()
			if partner_id in available_partners
		}
		return
