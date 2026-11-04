from core import registry
from core.registry import discover
from models.player import Player
from players.greedy_player import GreedyPlayer
from players.random_player import RandomPlayer


def test_builtins_are_discovered():
	available, _ = discover()
	assert available['r'] is RandomPlayer
	assert available['g'] is GreedyPlayer


def test_every_discovered_player_subclasses_player():
	available, _ = discover()
	assert all(issubclass(cls, Player) for cls in available.values())


def test_template_is_never_a_competitor():
	available, _ = discover()
	assert 'template' not in available
	assert not any(cls.__name__ == 'PlayerTemplate' for cls in available.values())


def test_missing_init_is_reported(tmp_path, monkeypatch):
	(tmp_path / 'player_7').mkdir()
	(tmp_path / 'player_7' / 'player.py').write_text('')
	(tmp_path / 'player_template').mkdir()
	monkeypatch.setattr(registry, '_PLAYERS_DIR', tmp_path)

	_, errors = discover()
	messages = {e.code: e.message for e in errors}
	assert '7' in messages
	assert 'no __init__.py' in messages['7']
	assert 'template' not in messages


def test_unimportable_group_is_reported_and_skipped(tmp_path, monkeypatch):
	(tmp_path / 'player_99').mkdir()
	(tmp_path / 'player_99' / '__init__.py').write_text('')
	monkeypatch.setattr(registry, '_PLAYERS_DIR', tmp_path)

	available, errors = discover()
	assert '99' not in available
	assert any(e.code == '99' for e in errors)
	assert 'g' in available
