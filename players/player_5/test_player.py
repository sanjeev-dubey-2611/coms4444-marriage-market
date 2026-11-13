"""Behavior and evidence-accounting checks for Group 5's market awareness."""

import uuid

import pytest

from core.engine import Engine
from models.player import (
	Advertisement,
	Candidate,
	DateContext,
	GameContext,
	PlayerSnapshot,
	RoundContext,
)
from players.player_5.player import Courtship, Player5, unknown_ideal_cost
from players.player_6.player import Player6
from players.random_player import RandomPlayer


def make_player(profile=(5, 2), ideal=(5, 2), monotonic=1, people=100, bonus=100, limit=None):
	return Player5(
		PlayerSnapshot(uuid.UUID(int=5), 0, profile, ideal),
		GameContext(people, len(profile), monotonic, 100, bonus, limit),
	)


def date(partner=1, number=1, rounds_left=50):
	return DateContext(
		101 - rounds_left, rounds_left, partner, Advertisement(0, 5, 0, 5), number, False
	)


def test_without_evidence_costs_match_uniform_assumptions():
	p = make_player()
	assert p.prior_ideal_loss == pytest.approx(unknown_ideal_cost(p.profile, p.monotonic))
	assert p.unknown_trait_cost == pytest.approx((5.0, 19.0))
	assert p.unknown_seek_cost == pytest.approx((5.0, 19.0))
	for dim in range(p.dims):
		assert p._pmf(dim) == pytest.approx([1 / 11] * 11)


def test_ads_are_counted_once_then_replaced_by_one_full_date():
	p = make_player()
	candidates = (Candidate(1, Advertisement(0, 9, 1, 8), 0),)
	p.choose_dates(candidates, RoundContext(1, 100, 100))
	first = p.unknown_trait_cost, p.unknown_seek_cost
	for round_number in range(2, 10):
		p.choose_dates(candidates, RoundContext(round_number, 101 - round_number, 100))
	assert (p.unknown_trait_cost, p.unknown_seek_cost) == first
	p.decide(date(), (4, 7), (6, 8))
	assert all(sum(row) == 0 for row in p.ad_traits + p.ad_desires)
	assert all(sum(row) == 1 for row in p.traits + p.desires)
	after_date = p.unknown_trait_cost, p.unknown_seek_cost
	p.decide(date(number=2), (4, 7), (6, 8))
	p.choose_dates(candidates, RoundContext(52, 49, 100))
	assert (p.unknown_trait_cost, p.unknown_seek_cost) == after_date
	assert all(sum(row) == 1 for row in p.traits + p.desires)


def test_date_without_prior_ad_still_counts_once():
	p = make_player()
	p.decide(date(), (5, 8), (5, 8))
	p.decide(date(number=2), (5, 8), (5, 8))
	assert all(sum(row) == 1 for row in p.traits + p.desires)
	assert all(sum(row) == 0 for row in p.ad_traits + p.ad_desires)


def test_learning_changes_which_advertised_trait_is_valuable():
	p = make_player(profile=(0, 0), ideal=(0, 0), monotonic=0, bonus=500)
	a = Advertisement(0, 2, 0, 0)
	b = Advertisement(1, 3, 0, 0)
	assert p._ad_value(a) > p._ad_value(b)
	for person in range(1, 9):
		p.decide(date(partner=person), (0, 10), (5, 5))
	# Seeing poor values on dim 1 makes a good dim-1 advertisement more useful.
	assert p._ad_value(b) > p._ad_value(a)
	assert p.unknown_trait_cost[0] < 35 < p.unknown_trait_cost[1]


def test_large_biased_ad_pool_cannot_overwhelm_prior():
	p = make_player()
	candidates = tuple(Candidate(i, Advertisement(0, 10, 1, 10), 0) for i in range(1, 1000))
	p.choose_dates(candidates[:8], RoundContext(1, 100, 1000))
	small = p._pmf(0), p._pmf(1, desire=True)
	p.choose_dates(candidates, RoundContext(2, 99, 1000))
	assert p._pmf(0) == pytest.approx(small[0])
	assert p._pmf(1, desire=True) == pytest.approx(small[1])
	assert p._pmf(0)[10] < 0.25
	assert sum(p._pmf(0)) == pytest.approx(1)


def test_mirrored_ideals_do_not_train_the_lie_safeguard():
	p = make_player(profile=(4, 4, 4), ideal=(5, 5, 5))
	uniform_loss = p.prior_ideal_loss
	p.choose_dates((Candidate(1, Advertisement(0, 5, 1, 4), 0),), RoundContext(1, 100, 100))
	p.decide(date(), (5, 5, 5), p.profile)
	assert p.prior_ideal_loss == pytest.approx(uniform_loss)
	assert all(sum(row) == 0 for row in p.desires + p.ad_desires)
	assert all(sum(row) == 1 for row in p.traits)
	assert p.history[1].score < p.bonus


def test_changed_claims_do_not_erase_worst_mismatch_or_add_samples():
	p = make_player()
	p.decide(date(), (0, 10), (5, 2))
	worst = p.history[1].own_loss
	evidence = p.unknown_trait_cost
	p.decide(date(number=2), p.ideal, (5, 2))
	assert p.history[1].own_loss == worst
	assert p.unknown_trait_cost == evidence
	assert all(sum(row) == 1 for row in p.traits)


def test_depleted_market_pursues_known_match_even_above_attempt_cap():
	p = make_player()
	p.first_scores.extend([100] * 8)
	p.history[1] = Courtship((5, 2), 0, score=80, partner_score=80)
	candidates = tuple(Candidate(i, Advertisement(0, 5, 1, 2), int(i == 1)) for i in range(1, 10))
	full = p.choose_dates(candidates, RoundContext(40, 61, 100))
	full_search = p._search_value(60, 8)
	depleted = p.choose_dates(candidates, RoundContext(40, 61, 10))
	assert 1 not in full
	assert depleted == [1]
	assert 0 < p._search_value(60, 8) < full_search


def test_fewer_visible_candidates_do_not_imply_fewer_singles():
	p = make_player()
	p.first_scores.extend([100] * 8)
	candidates = tuple(Candidate(i, Advertisement(0, 5, 1, 2), 0) for i in range(1, 100))
	p.choose_dates(candidates, RoundContext(1, 100, 100))
	full = p._search_value(100, 99)
	p.choose_dates(candidates[:9], RoundContext(2, 99, 100))
	assert p._search_value(99, 9) == full
	assert p._search_value(1, 9) == 0
	assert p._search_value(99, 0) == 0


@pytest.mark.parametrize('number', [1, 2])
def test_urgency_never_accepts_nonpositive_scores(number):
	p = make_player(profile=(0, 0), ideal=(10, 10), monotonic=0, bonus=10)
	p.choose_dates((), RoundContext(100, 1, 2))
	decision = p.decide(date(number=number, rounds_left=1), (0, 0), (10, 10))
	assert not decision.again
	assert not decision.propose


def test_first_date_cannot_propose_even_when_only_two_singles_remain():
	p = make_player()
	p.choose_dates((), RoundContext(99, 2, 2))
	assert not p.decide(date(rounds_left=2), p.ideal, p.profile).propose
	assert p.decide(date(number=2, rounds_left=1), p.ideal, p.profile).propose


@pytest.mark.parametrize('limit', [None, 0, 1, 4])
def test_interest_list_respects_limits(limit):
	p = make_player(limit=limit)
	candidates = tuple(Candidate(i, Advertisement(0, 5, 1, 2), 0) for i in range(1, 20))
	chosen = p.choose_dates(candidates, RoundContext(1, 100, 20))
	assert set(chosen) <= {c.index for c in candidates}
	assert len(chosen) == len(set(chosen))
	if limit is not None:
		assert len(chosen) <= limit


@pytest.mark.parametrize('dims,monotonic,rounds', [(1, 0, 1), (1, 1, 2), (5, 2, 50), (24, 24, 20)])
def test_seed7_games_are_reproducible_and_fault_free(dims, monotonic, rounds):
	def run():
		return Engine(
			[Player5] * 10 + [Player6] * 6 + [RandomPlayer] * 4,
			dims=dims,
			monotonic=monotonic,
			rounds=rounds,
			seed=7,
			timeout=1,
		).run()

	a = run()
	assert not a['faults']
	assert run() == a
