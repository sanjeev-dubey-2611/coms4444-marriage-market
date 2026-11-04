import copy
import json
import subprocess
import sys

import pytest

from players.player_3.tools import evaluate


def test_cli_selects_variant_and_seed_window_and_records_provenance(tmp_path):
	report_path = tmp_path / 'baseline.json'
	result = subprocess.run(
		[
			sys.executable,
			str(evaluate.ROOT / 'players/player_3/tools/evaluate.py'),
			'--quick',
			'--variants',
			'default',
			'--seed-start',
			'101',
			'--seeds',
			'3',
			'--workers',
			'1',
			'--out',
			str(report_path),
		],
		cwd=evaluate.ROOT,
		capture_output=True,
		text=True,
		timeout=60,
	)
	assert result.returncode == 0, result.stderr
	report = json.loads(report_path.read_text())
	assert {row['variant'] for row in report['raw']} == {'default'}
	assert {row['seed'] for row in report['raw']} == {101, 102, 103}
	self_play = [row for row in report['raw'] if row['label'].startswith('self_play|')]
	assert len(self_play) == 3
	assert self_play[0]['config'] == {
		'players': [['3v', 20]],
		'rounds': 100,
		'bonus': 100.0,
		'dims': 5,
		'monotonic': 2,
		'distribution': 'mixed',
		'timeout': 0.0,
		'list_multiple': 4.0,
	}
	assert report['jobs'] == len(report['raw'])
	assert len(report['metadata']['git_commit']) == 40
	assert report['metadata']['python_version']
	assert len(report['metadata']['source_sha256']['players/player_3/player.py']) == 64
	assert len(report['metadata']['source_sha256']['players/player_7/__init__.py']) == 64
	assert all(row['player_faults'] == 0 for row in report['raw'])


def test_default_quick_variants_remain_available():
	jobs = evaluate.build_jobs(seeds=[1], quick=True, ablation=False)
	assert {job['variant'] for job in jobs} == {'v1', 'default', 'all_on'}


@pytest.mark.parametrize(
	'args',
	[
		['--seeds', '0'],
		['--seeds', '-1'],
		['--workers', '0'],
		['--variants', 'not_a_variant'],
		['--variants', 'default', 'default'],
	],
)
def test_cli_rejects_invalid_run_selection(args, tmp_path):
	result = subprocess.run(
		[
			sys.executable,
			str(evaluate.ROOT / 'players/player_3/tools/evaluate.py'),
			*args,
			'--out',
			str(tmp_path / 'invalid.json'),
		],
		cwd=evaluate.ROOT,
		capture_output=True,
		text=True,
		timeout=15,
	)
	assert result.returncode == 2
	assert not (tmp_path / 'invalid.json').exists()


def _row(seed, happiness, *, variant='default', positive_rate=0.5):
	return {
		'label': 'self_play|B100|t100|mixed',
		'seed': seed,
		'variant': variant,
		'n': 20,
		'mean_happiness': happiness,
		'positive_rate': positive_rate,
		'married_rate': 0.5,
		'faults': 0,
		'player_faults': 0,
		'config': {
			'players': [['3v', 20]],
			'rounds': 100,
			'bonus': 100.0,
			'dims': 5,
			'monotonic': 2,
			'distribution': 'mixed',
			'timeout': 0.0,
			'list_multiple': 4.0,
		},
	}


def _report(rows):
	return {
		'raw': rows,
		'metadata': {
			'python_version': '3.13.9',
			'platform': 'test-platform',
			'source_sha256': {
				'core/engine.py': 'a' * 64,
				'players/player_7/player.py': 'b' * 64,
				'players/player_3/player.py': 'c' * 64,
			},
		},
	}


def _compare(before, after, **kwargs):
	from players.player_3.tools.compare_reports import compare_reports

	return compare_reports(_report(before), _report(after), **kwargs)


@pytest.mark.parametrize(
	'change',
	[
		'engine',
		'opponent',
		'missing_source',
		'new_opponent_helper',
		'no_metadata',
		'no_hashes',
		'python',
		'platform',
	],
)
def test_comparison_rejects_changes_to_fixed_experiment_sources(change):
	from players.player_3.tools.compare_reports import compare_reports

	before = _report([_row(1, 10)])
	after = copy.deepcopy(before)
	metadata = after['metadata']
	if change == 'engine':
		metadata['source_sha256']['core/engine.py'] = 'd' * 64
	elif change == 'opponent':
		metadata['source_sha256']['players/player_7/player.py'] = 'd' * 64
	elif change == 'missing_source':
		del metadata['source_sha256']['core/engine.py']
	elif change == 'new_opponent_helper':
		metadata['source_sha256']['players/player_7/helper.py'] = 'd' * 64
	elif change == 'no_metadata':
		del after['metadata']
	elif change == 'no_hashes':
		metadata['source_sha256'] = {}
	elif change == 'python':
		metadata['python_version'] = '3.14.0'
	elif change == 'platform':
		metadata['platform'] = 'another-platform'
	with pytest.raises(ValueError):
		compare_reports(before, after)


def test_comparison_allows_player3_changes_on_a_new_commit():
	from players.player_3.tools.compare_reports import compare_reports

	before = _report([_row(1, 10)])
	after = copy.deepcopy(before)
	after['metadata']['git_commit'] = 'new-commit'
	after['metadata']['source_sha256']['players/player_3/player.py'] = 'd' * 64
	after['metadata']['source_sha256']['players/player_3/distribution.py'] = 'e' * 64
	assert compare_reports(before, after)['pairs'] == 1


def test_comparison_pairs_by_seed_and_can_compare_different_variant_names():
	before = [_row(1, 10), _row(2, 20)]
	after = [
		_row(2, 18, variant='unique_mean', positive_rate=0.6),
		_row(1, 22, variant='unique_mean', positive_rate=0.7),
		_row(99, -1000, variant='unselected'),
	]
	report = _compare(before, after, after_variant='unique_mean')
	assert report['pairs'] == 2
	metrics = report['by_cell'][0]['metrics']
	assert metrics['mean_happiness']['before'] == 15
	assert metrics['mean_happiness']['after'] == 20
	assert metrics['mean_happiness']['delta'] == 5
	assert metrics['positive_rate']['delta'] == pytest.approx(0.15)
	assert report['paired_deltas'][0]['seed'] == 1
	assert report['paired_deltas'][0]['mean_happiness'] == 12
	assert report['paired_deltas'][1]['mean_happiness'] == -2


@pytest.mark.parametrize(
	'fault',
	[
		'missing_seed',
		'duplicate_seed',
		'wrong_config',
		'wrong_roster_order',
		'wrong_population',
		'missing_config',
		'nonfinite_metric',
		'no_selected_variant',
	],
)
def test_comparison_rejects_unpaired_or_incompatible_reports(fault):
	before = [_row(1, 10), _row(2, 20)]
	after = copy.deepcopy(before)
	if fault == 'missing_seed':
		after.pop()
	elif fault == 'duplicate_seed':
		after.append(copy.deepcopy(after[0]))
	elif fault == 'wrong_config':
		after[0]['config']['dims'] = 6
	elif fault == 'wrong_roster_order':
		before[0]['config']['players'] = [['3v', 10], ['g', 10]]
		after[0]['config']['players'] = [['g', 10], ['3v', 10]]
	elif fault == 'wrong_population':
		after[0]['n'] = 10
	elif fault == 'missing_config':
		del after[0]['config']
	elif fault == 'nonfinite_metric':
		after[0]['mean_happiness'] = float('nan')
	elif fault == 'no_selected_variant':
		after = []
	with pytest.raises(ValueError):
		_compare(before, after)


def test_comparison_cli_prints_json_and_reports_invalid_pairs(tmp_path):
	before_path = tmp_path / 'before.json'
	after_path = tmp_path / 'after.json'
	before_path.write_text(json.dumps(_report([_row(1, 10)])))
	after_path.write_text(json.dumps(_report([_row(1, 15)])))
	command = [
		sys.executable,
		str(evaluate.ROOT / 'players/player_3/tools/compare_reports.py'),
		str(before_path),
		str(after_path),
	]
	result = subprocess.run(command, capture_output=True, text=True, timeout=15)
	assert result.returncode == 0, result.stderr
	assert json.loads(result.stdout)['by_cell'][0]['metrics']['mean_happiness']['delta'] == 5
	after_path.write_text(json.dumps(_report([_row(2, 15)])))
	result = subprocess.run(command, capture_output=True, text=True, timeout=15)
	assert result.returncode == 2
