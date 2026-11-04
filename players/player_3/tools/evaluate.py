"""Per-group evaluation for Group 3 strategy variants.

Usage:
    uv run python players/player_3/tools/evaluate.py --quick
    uv run python players/player_3/tools/evaluate.py --seeds 30
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from core.engine import Engine  # noqa: E402
from core.registry import discover  # noqa: E402
from models.player import Player  # noqa: E402
from players.player_3.player import (  # noqa: E402
	Player3,
	Player3AllOn,
	Player3LieFirstMono,
	Player3NoCoordination,
	Player3NoMarketBar,
	Player3NoTrust,
	Player3V1,
	Player3WithConfidence,
	Player3WithIncompatible,
	Player3WithSteer,
)

VARIANTS: dict[str, type[Player]] = {
	'v1': Player3V1,
	'default': Player3,
	'all_on': Player3AllOn,
	'no_trust': Player3NoTrust,
	'no_market_bar': Player3NoMarketBar,
	'no_coordination': Player3NoCoordination,
	'with_confidence': Player3WithConfidence,
	'with_steer': Player3WithSteer,
	'with_incompatible': Player3WithIncompatible,
	'lie_first_mono': Player3LieFirstMono,
}


def _bootstrap_ci(values: list[float], n: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
	if len(values) < 2:
		v = values[0] if values else 0.0
		return v, v
	rng = random.Random(0)
	lo = int(n * alpha / 2)
	hi = n - lo - 1
	samples = []
	for _ in range(n):
		draw = [values[rng.randrange(len(values))] for _ in range(len(values))]
		samples.append(statistics.fmean(draw))
	samples.sort()
	return samples[lo], samples[hi]


def _metrics_for_name(players: list[dict], name: str) -> dict[str, float]:
	subset = [p for p in players if p['name'] == name]
	if not subset:
		return {}
	scores = [float(p['happiness']) for p in subset]
	pos = sum(1 for s in scores if s > 0) / len(scores)
	married = sum(1 for p in subset if p['married']) / len(subset)
	mrounds = [
		float(p['married_on']) for p in subset if p['married'] and p['married_on'] is not None
	]
	decs = [float(p['own_decrement']) for p in subset if p['own_decrement'] is not None]
	return {
		'n': len(subset),
		'mean_happiness': statistics.fmean(scores),
		'positive_rate': pos,
		'married_rate': married,
		'mean_married_round': statistics.fmean(mrounds) if mrounds else float('nan'),
		'mean_own_decrement': statistics.fmean(decs) if decs else float('nan'),
	}


def _run_job(job: dict) -> dict:
	variant = VARIANTS[job['variant']]
	available, _ = discover()
	roster: list[type[Player]] = []
	for code, count in job['players']:
		if code == '3v':
			roster.extend([variant] * count)
		else:
			cls = available[code]
			roster.extend([cls] * count)

	random.seed(job['seed'])
	results = Engine(
		players=roster,
		rounds=job['rounds'],
		bonus=job['bonus'],
		dims=job['dims'],
		monotonic=job['monotonic'],
		seed=job['seed'],
		distribution=job['distribution'],
		timeout=job['timeout'],
		keep_records=False,
		list_multiple=job['list_multiple'],
	).run()

	target_name = variant.__name__
	m = _metrics_for_name(results['players'], target_name)
	return {
		'label': job['label'],
		'variant': job['variant'],
		'seed': job['seed'],
		**m,
		'faults': len(results['faults']),
	}


def _arena_rosters(quick: bool) -> list[tuple[str, list[tuple[str, int]]]]:
	n = 20 if quick else 20
	half = n // 2
	rosters: list[tuple[str, list[tuple[str, int]]]] = [
		('self_play', [('3v', n)]),
		('vs_greedy', [('3v', half), ('g', half)]),
		('vs_random', [('3v', half), ('r', half)]),
	]
	avail, _ = discover()
	for k in range(1, 11):
		code = str(k)
		if code == '3' or code not in avail:
			continue
		rosters.append((f'vs_{code}', [('3v', half), (code, half)]))
	if not quick:
		mix: list[tuple[str, int]] = [('3v', 2)]
		avail, _ = discover()
		for code in sorted(avail, key=lambda c: (len(c), c)):
			if code in ('3', '3v', 'r'):
				continue
			mix.append((code, 2))
		while sum(c for _, c in mix) < n:
			mix.append(('g', 2))
		rosters.append(('full_mix', mix[: n // 2]))
	return rosters


def build_jobs(
	*,
	seeds: list[int],
	quick: bool,
	ablation: bool,
) -> list[dict]:
	bonuses = [100] if quick or ablation else [50, 100, 200]
	rounds_list = [100] if quick or ablation else [25, 100]
	dists = ['mixed'] if quick or ablation else ['uniform', 'normal', 'mixed']
	if ablation:
		rosters = [
			('self_play', [('3v', 20)]),
			('vs_10', [('3v', 10), ('10', 10)]),
			('vs_greedy', [('3v', 10), ('g', 10)]),
		]
		variants = list(VARIANTS)
	elif quick:
		rosters = _arena_rosters(True)
		variants = ['v1', 'default', 'all_on']
	else:
		rosters = _arena_rosters(False)
		variants = list(VARIANTS)
	jobs: list[dict] = []
	for arena, players in rosters:
		for bonus in bonuses:
			for rounds in rounds_list:
				for dist in dists:
					for variant in variants:
						for seed in seeds:
							jobs.append(
								{
									'label': f'{arena}|B{bonus}|t{rounds}|{dist}',
									'variant': variant,
									'players': players,
									'rounds': rounds,
									'bonus': float(bonus),
									'dims': 5,
									'monotonic': 2,
									'distribution': dist,
									'seed': seed,
									'timeout': 0.0,
									'list_multiple': 4.0,
								}
							)
	return jobs


def aggregate(rows: list[dict]) -> list[dict]:
	groups: dict[tuple[str, str], list[dict]] = {}
	for row in rows:
		key = (row['label'], row['variant'])
		groups.setdefault(key, []).append(row)

	out = []
	for (label, variant), items in sorted(groups.items()):
		entry: dict = {'label': label, 'variant': variant, 'seeds': len(items)}
		for field in ('mean_happiness', 'positive_rate', 'married_rate'):
			vals = [float(i[field]) for i in items]
			lo, hi = _bootstrap_ci(vals)
			entry[f'{field}_mean'] = statistics.fmean(vals)
			entry[f'{field}_ci_lo'] = lo
			entry[f'{field}_ci_hi'] = hi
		out.append(entry)
	return out


def main() -> None:
	parser = argparse.ArgumentParser(description='Evaluate Group 3 player variants.')
	parser.add_argument('--seeds', type=int, default=10, help='Number of seeds (1..N).')
	parser.add_argument('--quick', action='store_true', help='Smaller grid for smoke runs.')
	parser.add_argument('--ablation', action='store_true', help='All variants on key arenas.')
	parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
	parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'player_3_eval.json')
	args = parser.parse_args()

	seeds = list(range(1, args.seeds + 1))
	jobs = build_jobs(seeds=seeds, quick=args.quick, ablation=args.ablation)
	with mp.Pool(args.workers) as pool:
		rows = pool.map(_run_job, jobs)

	args.out.parent.mkdir(parents=True, exist_ok=True)
	report = {'jobs': len(jobs), 'by_cell': aggregate(rows), 'raw': rows}
	args.out.write_text(json.dumps(report, indent=2) + '\n')

	print(f'Wrote {args.out} ({len(jobs)} runs)')
	# Headline: full vs v1 on self_play mixed B100 t100
	for label_prefix in ('self_play|',):
		for row in report['by_cell']:
			if not row['label'].startswith(label_prefix):
				continue
			if row['variant'] in ('v1', 'default', 'all_on') and 'positive_rate_mean' in row:
				print(
					f'{row["label"]} {row["variant"]}: '
					f'pos={row["positive_rate_mean"]:.3f} '
					f'happy={row["mean_happiness_mean"]:.1f}'
				)


if __name__ == '__main__':
	main()
