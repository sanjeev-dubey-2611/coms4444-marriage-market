import json

from core.distributions import (
	discover_distributions,
	distribution_from_settings,
	distribution_from_sweep_value,
)
from models.profile import normal_row, uniform_row


def test_discover_distributions_loads_group_rows(tmp_path):
	group = tmp_path / 'player_3'
	group.mkdir()
	(group / 'distribution.json').write_text(json.dumps(list(normal_row())))
	(tmp_path / 'player_template').mkdir()
	(tmp_path / 'player_template' / 'distribution.json').write_text(json.dumps(list(uniform_row())))

	loaded, errors = discover_distributions(tmp_path)
	assert errors == []
	assert loaded == {'3': normal_row()}


def test_discover_distributions_reports_invalid_rows(tmp_path):
	group = tmp_path / 'player_4'
	group.mkdir()
	(group / 'distribution.json').write_text(json.dumps([0.5, 0.5]))

	loaded, errors = discover_distributions(tmp_path)
	assert loaded == {}
	assert len(errors) == 1
	assert errors[0].code == '4'


def test_distribution_from_settings_composes_dim_map(tmp_path, monkeypatch):
	group = tmp_path / 'player_3'
	group.mkdir()
	only_zero = [1.0] + [0.0] * 10
	(group / 'distribution.json').write_text(json.dumps(only_zero))
	monkeypatch.setattr('core.distributions._PLAYERS_DIR', tmp_path)

	table, name = distribution_from_settings(
		distribution='normal',
		dims=2,
		monotonic=0,
		dim_dist='0:3',
	)
	assert name == 'compose:0:3,1:normal'
	assert table[0] == tuple(only_zero)
	assert table[1] == normal_row()


def test_distribution_from_settings_reads_dim_dist_file(tmp_path, monkeypatch):
	group = tmp_path / 'player_7'
	group.mkdir()
	(group / 'distribution.json').write_text(json.dumps(list(uniform_row())))
	monkeypatch.setattr('core.distributions._PLAYERS_DIR', tmp_path)

	map_path = tmp_path / 'dims.txt'
	map_path.write_text('0: 7\n1: normal\n')
	table, name = distribution_from_settings(
		distribution='mixed',
		dims=2,
		monotonic=1,
		dim_dist_file=str(map_path),
	)
	assert name.startswith('compose:')
	assert table[0] == uniform_row()
	assert table[1] == normal_row()


def test_distribution_from_sweep_value_accepts_dim_map_object():
	table, name = distribution_from_sweep_value(
		{'0': 'uniform', '1': 'normal'},
		dims=2,
		monotonic=0,
		catalog={},
	)
	assert table == (uniform_row(), normal_row())
	assert name == 'compose:0:uniform,1:normal'
