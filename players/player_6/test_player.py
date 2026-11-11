"""Tests for Player 6's secretary-style, truthful dating strategy."""

import math
import uuid

from models.player import (
	Advertisement,
	Candidate,
	DateContext,
	GameContext,
	PlayerSnapshot,
	RoundContext,
)
from players.player_6.player import Player6


def make_player(rounds=20, bonus=100, limit=None):
	return Player6(
		PlayerSnapshot(uuid.UUID(int=6), 0, (5, 2), (5, 2)),
		GameContext(10, 2, 1, rounds, bonus, limit),
	)


def date(round_number, number=1, partner=1):
	return DateContext(
		round_number,
		21 - round_number,
		partner,
		Advertisement(0, 5, 0, 5),
		number,
		False,
	)


def test_exploration_and_truthful_disclosures():
	p = make_player()

	assert p.explore_rounds == math.ceil(20 * Player6.EXPLORE_FRACTION)
	assert p.project(date(5)) == p.profile
	assert p.project(date(7)) == p.profile
	assert p.describe_ideal(date(7), (5, 8)) == p.ideal

	# Score = 100 - 49 - 0.5 * 49 = 26.5: positive but not exceptional while exploring.
	assert not p.decide(date(1, 2), (5, 9), (5, 9)).propose

	# A perfect match is exceptional enough to pursue during exploration.
	assert p.decide(date(2), p.ideal, p.profile).again
	assert not p.decide(date(2), p.ideal, p.profile).propose

	assert p.decide(date(7, 2), p.ideal, p.profile).propose


def test_repeat_observations_replace_distribution_sample():
	p = make_player()

	p.decide(date(1), (5, 8), (5, 8))
	p.decide(date(2, 2), (7, 6), (7, 6))

	assert p.traits[1][8] == 0
	assert p.traits[1][6] == 1
	assert all(sum(row) == 1 for row in p.traits + p.desires)


def test_tuned_early_exception_accepts_positive_match_below_old_cutoff():
	p = make_player()
	# Score = 100 - 25 - 0.5 * 25 = 62.5: clears the early-exception cutoff.
	decision = p.decide(date(1), (5, 7), (5, 7))
	assert decision.again
	assert not decision.propose


def test_nonpositive_matches_never_proposed_to():
	p = make_player()

	assert not p.decide(date(7, 2), (0, 10), (10, 10)).propose

	assert not make_player(bonus=0).decide(date(7, 2), (5, 2), (5, 2)).propose


def test_short_game_and_interest_limits():
	p = make_player(rounds=2, limit=1)

	assert p.explore_rounds == 0
	assert p.decide(date(1), p.ideal, p.profile).again

	candidates = tuple(Candidate(i, Advertisement(0, 5, 0, 5), 0) for i in range(1, 5))

	assert len(p.choose_dates(candidates, RoundContext(1, 2, 10))) == 1

	q = make_player(limit=0)
	assert q.choose_dates(candidates, RoundContext(1, 20, 10)) == []
	assert not q.partners


def test_denied_list_is_reversible_without_permanent_incompatibility():
	p = make_player(rounds=20)

	# Score = 100 - 64 - 0.5 * 49 = 11.5: positive, but below the post-exploration
	# reservation threshold early on, and above it once the threshold decays.
	p.decide(date(1, partner=3), (5, 10), (5, 9))
	p.choose_dates((), RoundContext(p.explore_rounds + 1, 14, 10))

	assert 3 in p.denied

	p.choose_dates((), RoundContext(20, 1, 2))

	assert 3 not in p.denied


def test_lie_flag_controls_ad_exaggeration():
	p = make_player()
	p.LIE = True
	assert p.advertise().own_value == 7

	p = make_player()
	p.LIE = False
	assert p.advertise().own_value == 5


def test_assume_lies_docks_monotonic_claims_only():
	# Player: m=1, profile (5, 2), ideal (5, 2).
	honest = make_player()
	sceptic = make_player()
	sceptic.ASSUME_LIES = True

	# Monotonic claim 6 is believed as 4: gap 1 on dim 0. Dim 1 untouched.
	assert honest._penalty((6, 2), (5, 2)) == 0
	assert sceptic._penalty((6, 2), (5, 2)) == 1
	# Non-monotonic dim is never docked; docking floors at 0.
	assert sceptic._penalty((7, 9), (5, 2)) == honest._penalty((7, 9), (5, 2))
	assert sceptic._believed(0, 1) == 0

	# Ads are claims too.
	candidate = Candidate(1, Advertisement(0, 6, 1, 2), 0)
	assert sceptic._estimate_unknown(candidate) == honest._estimate_unknown(candidate) - 1
