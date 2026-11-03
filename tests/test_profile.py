import json
import random

import pytest

from models.profile import (
	PRESETS,
	VALUES,
	build_distribution,
	compose_distribution,
	decrement,
	dims_from_dim_map,
	happiness,
	load_dim_map,
	load_distribution,
	load_distribution_row,
	normal_row,
	parse_dim_map,
	sample_profile,
	uniform_row,
	validate_distribution,
	validate_row,
)


def test_monotonic_dimensions_only_penalise_a_shortfall():
	# Wanted 7 and got 9 on a monotonic dimension: more is fine.
	assert decrement((7,), (9,), monotonic=1) == 0
	assert decrement((7,), (7,), monotonic=1) == 0
	assert decrement((7,), (4,), monotonic=1) == 9


def test_other_dimensions_penalise_both_directions():
	assert decrement((7,), (9,), monotonic=0) == 4
	assert decrement((7,), (4,), monotonic=0) == 9


def test_only_the_first_m_dimensions_are_monotonic():
	desired = (5, 5, 5)
	partner = (8, 8, 8)
	assert decrement(desired, partner, monotonic=0) == 27
	assert decrement(desired, partner, monotonic=1) == 18
	assert decrement(desired, partner, monotonic=3) == 0


def test_happiness_subtracts_half_the_partners_decrement():
	# Mine: wanted (8, 5), partner is (6, 7) with m=1 -> 4 + 4 = 8.
	# Theirs: wanted (4, 2), I am (3, 3) -> 1 + 1 = 2, halved to 1.
	assert happiness(100, (3, 3), (8, 5), (6, 7), (4, 2), monotonic=1) == 100 - 8 - 1


def test_happiness_can_be_negative():
	assert happiness(0, (0,), (10,), (0,), (10,), monotonic=0) == -150


@pytest.mark.parametrize('name', PRESETS)
def test_presets_are_valid_distributions(name):
	table = build_distribution(name, dims=6, monotonic=2)
	assert validate_distribution(table, 6) == table


def test_mixed_uses_normal_rows_for_monotonic_dimensions():
	table = build_distribution('mixed', dims=4, monotonic=2)
	assert table[0] == normal_row(5.0, 2.0)
	assert table[1] == normal_row(5.0, 2.0)
	assert table[2] == uniform_row()


def test_unknown_preset_is_rejected():
	with pytest.raises(ValueError, match='unknown distribution'):
		build_distribution('bimodal', dims=2, monotonic=0)


def test_validate_rejects_rows_that_do_not_sum_to_one():
	with pytest.raises(ValueError, match='sums to'):
		validate_distribution([[0.5] * 11], 1)


def test_validate_rejects_wrong_shapes():
	with pytest.raises(ValueError, match='one row per dimension'):
		validate_distribution([uniform_row()], 2)
	with pytest.raises(ValueError, match='11 entries'):
		validate_distribution([[0.5, 0.5]], 1)


def test_validate_rejects_negative_probabilities():
	row = [0.0] * 11
	row[0], row[1], row[2] = 1.5, -1.0, 0.5
	with pytest.raises(ValueError, match='negative'):
		validate_distribution([row], 1)


def test_load_distribution_accepts_a_bare_list_or_a_P_key(tmp_path):
	rows = [list(uniform_row()), list(normal_row())]
	bare = tmp_path / 'bare.json'
	bare.write_text(json.dumps(rows))
	keyed = tmp_path / 'keyed.json'
	keyed.write_text(json.dumps({'P': rows}))
	assert load_distribution(bare, 2) == load_distribution(keyed, 2)


def test_load_distribution_row_accepts_bare_keyed_or_one_row_table(tmp_path):
	row = list(normal_row())
	bare = tmp_path / 'bare.json'
	bare.write_text(json.dumps(row))
	keyed = tmp_path / 'keyed.json'
	keyed.write_text(json.dumps({'P': row}))
	table = tmp_path / 'table.json'
	table.write_text(json.dumps({'P': [row]}))
	assert load_distribution_row(bare) == load_distribution_row(keyed) == load_distribution_row(table)


def test_load_distribution_row_rejects_multi_row_tables(tmp_path):
	path = tmp_path / 'multi.json'
	path.write_text(json.dumps({'P': [list(uniform_row()), list(normal_row())]}))
	with pytest.raises(ValueError, match='single dimension row'):
		load_distribution_row(path)


def test_parse_dim_map_accepts_commas_and_lines():
	assert parse_dim_map('0:3,1:normal') == {0: '3', 1: 'normal'}
	assert parse_dim_map('0: 3\n# comment\n1:normal\n2:7') == {0: '3', 1: 'normal', 2: '7'}
	# Commas inside comments must not become fake entries.
	assert parse_dim_map('# a, b, c\n0: 3') == {0: '3'}


def test_dims_from_dim_map_requires_contiguous_dimensions():
	assert dims_from_dim_map({0: '3', 1: 'normal', 2: '7'}) == 3
	with pytest.raises(ValueError, match='missing dimension'):
		dims_from_dim_map({0: '3', 2: '7'})


def test_load_dim_map_reads_json_or_text(tmp_path):
	json_path = tmp_path / 'map.json'
	json_path.write_text(json.dumps({'0': '3', '1': 'normal'}))
	text_path = tmp_path / 'map.txt'
	text_path.write_text('0: 3\n1: normal\n')
	assert load_dim_map(json_path) == load_dim_map(text_path) == {0: '3', 1: 'normal'}


def test_compose_distribution_mixes_presets_and_catalog_rows():
	custom = validate_row([1.0] + [0.0] * 10)
	table, label = compose_distribution(
		{0: '3', 2: 'uniform'},
		dims=3,
		monotonic=1,
		catalog={'3': custom},
		default='normal',
	)
	assert table[0] == custom
	assert table[1] == normal_row()
	assert table[2] == uniform_row()
	assert label == 'compose:0:3,1:normal,2:uniform'


def test_compose_rejects_unknown_sources_and_out_of_range_dims():
	with pytest.raises(ValueError, match='unknown distribution source'):
		compose_distribution({0: '99'}, dims=1, monotonic=0, catalog={})
	with pytest.raises(ValueError, match='only allows'):
		compose_distribution({5: 'uniform'}, dims=2, monotonic=0, catalog={})


def test_samples_stay_on_the_scale_and_follow_the_distribution():
	only_seven = [0.0] * 11
	only_seven[7] = 1.0
	table = (uniform_row(), tuple(only_seven))
	rng = random.Random(1)
	for _ in range(200):
		profile = sample_profile(rng, table)
		assert len(profile) == 2
		assert profile[0] in VALUES
		assert profile[1] == 7
