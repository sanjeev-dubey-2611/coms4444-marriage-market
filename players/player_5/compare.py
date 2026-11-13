"""Replay selected board configurations against the pre-change Group 5 player.

Run from the repository root:
    uv run python -m players.player_5.compare --seed 7 --out /tmp/group5-seed7.json

This is one-factor coverage, not the full Cartesian product of the board.
Every variant gets identical initial profiles, seat order, and engine seed.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from statistics import fmean

from core.engine import Engine
from core.registry import discover
from models.profile import load_distribution_row
from players.player_5.player import Player5

ROOT = Path(__file__).resolve().parents[2]
BASELINE = '9e4c5d8c646013b50554e6ad39e0c2025d2b3f4f'


@lru_cache
def variants(ref):
	# Load the exact old source without editing the checked-out player.
	source = subprocess.run(
		['git', 'show', f'{ref}:players/player_5/player.py'],
		cwd=ROOT,
		check=True,
		capture_output=True,
		text=True,
	).stdout
	module = types.ModuleType('_group5_baseline')
	sys.modules[module.__name__] = module
	exec(compile(source, f'{ref}:players/player_5/player.py', 'exec'), module.__dict__)
	baseline = module.Player5

	class MarketOnly(baseline):
		def __init__(self, snapshot, ctx):
			super().__init__(snapshot, ctx)
			self.current_singles = self.people

		def choose_dates(self, candidates, turn):
			self.current_singles = turn.singles
			return super().choose_dates(candidates, turn)

		_search_value = Player5._search_value

	class LearningOnly(Player5):
		_search_value = baseline._search_value

	return {
		'baseline': baseline,
		'market_only': MarketOnly,
		'learning_only': LearningOnly,
		'combined': Player5,
	}


def cases():
	base = dict(
		people=110,
		rounds=50,
		bonus=100,
		dims=5,
		monotonic=2,
		distribution='mixed',
		roster='mix-random',
	)
	result = {'reference': base}
	for bonus in (10, 50, 150, 500):
		result[f'bonus-{bonus}'] = dict(base, bonus=bonus)
	for rounds in (5, 150, 500):
		result[f'rounds-{rounds}'] = dict(base, rounds=rounds)
	for dims in (1, 2, 5, 8, 10, 24):
		for monotonic in sorted({0, dims // 2, dims}):
			if (dims, monotonic) != (5, 2):
				result[f'dims-{dims}-mono-{monotonic}'] = dict(base, dims=dims, monotonic=monotonic)
	for distribution in ('uniform', 'normal', 'group9', 'group10', 'group10-reversed', 'group2'):
		result[f'dist-{distribution}'] = dict(base, distribution=distribution)
	for people in (20, 22, 330, 1100):
		result[f'people-{people}'] = dict(
			base, people=people, roster='mix' if people == 20 else 'mix-random'
		)
	for roster in ('solo', 'solo-random', 'random90', 'mix'):
		result[f'roster-{roster}'] = dict(base, roster=roster)
	for opponent in (1, 2, 3, 4, 6, 7, 8, 9, 10):
		result[f'half-group{opponent}'] = dict(base, roster=f'half-{opponent}')
	return result


def seats_for(case):
	n, roster = case['people'], case['roster']
	if roster == 'solo':
		return ['5'] * n
	if roster == 'solo-random':
		return ['5'] * (n - n // 11) + ['r'] * (n // 11)
	if roster == 'random90':
		return ['5'] * (n // 10) + ['r'] * (n - n // 10)
	if roster.startswith('half-'):
		return ['5'] * (n // 2) + [roster.removeprefix('half-')] * (n // 2)
	codes = [str(i) for i in range(1, 11)] + (['r'] if roster == 'mix-random' else [])
	assert n % len(codes) == 0
	return codes * (n // len(codes))


def run_case(name, case, seed, ref):
	registry, errors = discover()
	if errors:
		raise RuntimeError(errors)
	seats = seats_for(case)
	distribution = case['distribution']
	if distribution.startswith('group'):
		group = distribution.removeprefix('group').split('-')[0]
		row = load_distribution_row(ROOT / f'players/player_{group}/distribution.json')
		if distribution.endswith('-reversed'):
			row = tuple(reversed(row))
		distribution = tuple(row for _ in range(case['dims']))
	metrics, initial_hash = {}, None
	for variant, player in variants(ref).items():
		engine = Engine(
			[player if code == '5' else registry[code] for code in seats],
			rounds=case['rounds'],
			bonus=case['bonus'],
			dims=case['dims'],
			monotonic=case['monotonic'],
			distribution=distribution,
			seed=seed,
			timeout=1.0,
			keep_records=False,
			list_multiple=4.0,
		)
		fingerprint = hashlib.sha256(
			json.dumps([engine.profiles, engine.ideals, seats]).encode()
		).hexdigest()
		if initial_hash is None:
			initial_hash = fingerprint
		assert fingerprint == initial_hash, 'Variants must start from the same market'
		results = engine.run()
		ours = [p for p, code in zip(results['players'], seats, strict=True) if code == '5']
		faults = results['faults']
		our_names = {p['name'] for p in ours}
		our_faults = [fault for fault in faults if fault.split('[', 1)[0] in our_names]
		metrics[variant] = {
			'positive_rate': fmean(p['positive'] for p in ours),
			'mean_happiness': fmean(p['happiness'] for p in ours),
			'married_rate': fmean(p['married'] for p in ours),
			'negative_rate': fmean(p['happiness'] < 0 for p in ours),
			'group5_faults': len(our_faults),
			'all_faults': len(faults),
			'fault_examples': faults[:10],
		}
	return {
		'name': name,
		'config': case,
		'seat_counts': {code: seats.count(code) for code in dict.fromkeys(seats)},
		'initial_market_sha256': initial_hash,
		'metrics': metrics,
	}


def summarize(results):
	summary = {}
	for variant in ('baseline', 'market_only', 'learning_only', 'combined'):
		deltas = [
			r['metrics'][variant]['positive_rate'] - r['metrics']['baseline']['positive_rate']
			for r in results
		]
		summary[variant] = {
			metric: fmean(r['metrics'][variant][metric] for r in results)
			for metric in ('positive_rate', 'mean_happiness', 'married_rate', 'negative_rate')
		}
		summary[variant].update(
			positive_rate_delta=fmean(deltas),
			wins=sum(d > 1e-12 for d in deltas),
			ties=sum(abs(d) <= 1e-12 for d in deltas),
			losses=sum(d < -1e-12 for d in deltas),
			group5_faults=sum(r['metrics'][variant]['group5_faults'] for r in results),
			all_faults=sum(r['metrics'][variant]['all_faults'] for r in results),
		)
	return summary


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--seed', type=int, default=7)
	parser.add_argument('--baseline', default=BASELINE)
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--case', action='append', choices=sorted(cases()))
	parser.add_argument('--out', type=Path, required=True)
	args = parser.parse_args()
	selected = {name: case for name, case in cases().items() if not args.case or name in args.case}
	results = []
	with ProcessPoolExecutor(max_workers=args.workers) as pool:
		jobs = {
			pool.submit(run_case, name, case, args.seed, args.baseline): name
			for name, case in selected.items()
		}
		for job in as_completed(jobs):
			result = job.result()
			results.append(result)
			print(f'{len(results)}/{len(jobs)} {result["name"]}', flush=True)
	results.sort(key=lambda r: r['name'])
	report = {
		'seed': args.seed,
		'baseline_commit': args.baseline,
		'player_sha256': hashlib.sha256(
			Path(__file__).with_name('player.py').read_bytes()
		).hexdigest(),
		'scope': 'One-factor board coverage; equally weighted markets; not a full Cartesian sweep or a multi-seed validation.',
		'list_multiple': 4,
		'timeout': 1,
		'summary': summarize(results),
		'cases': results,
	}
	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(json.dumps(report, indent=2) + '\n')
	print(json.dumps(report['summary'], indent=2), flush=True)
	if any(row['all_faults'] for row in report['summary'].values()):
		raise SystemExit('Faults recorded; inspect the report before interpreting results.')


if __name__ == '__main__':
	main()
