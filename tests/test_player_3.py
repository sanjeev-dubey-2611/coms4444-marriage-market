from models.player import GameContext, PlayerSnapshot
from players.player_3.player import Player3, Player3V1, Player3WithConfidence


def _ctx(**kwargs) -> GameContext:
	defaults = dict(people=20, dims=5, monotonic=2, rounds=100, bonus=100.0, list_limit=40)
	defaults.update(kwargs)
	return GameContext(**defaults)


def _snap(profile=None, ideal=None) -> PlayerSnapshot:
	import uuid

	p = profile or (5, 5, 5, 5, 5)
	i = ideal or (8, 8, 2, 2, 2)
	return PlayerSnapshot(id=uuid.uuid4(), index=0, profile=p, ideal=i)


def test_mirror_detection_lowers_trust():
	player = Player3(_snap(), _ctx())
	ad = player.advertise()
	our = player.project(
		type(
			'DC',
			(),
			{
				'round': 1,
				'rounds_left': 99,
				'partner': 1,
				'partner_ad': ad,
				'date_number': 1,
				'repeat': False,
			},
		)()
	)
	stated = our
	_adj_obs, _adj_ideal, trust = player._adjust_profiles(1, ad, our, stated, our)
	assert trust < 1.0


def test_v1_reservation_decays_to_zero():
	player = Player3V1(_snap(), _ctx())
	player._seen = [10.0, 20.0, 30.0, 40.0, 50.0]
	assert player._reservation_v1(50) > 0
	assert player._reservation_v1(1) == 0.0


def test_confidence_lb_below_point_estimate():
	player = Player3WithConfidence(_snap(), _ctx())
	assert player._confidence_lb(50.0, 0.5) < 50.0
