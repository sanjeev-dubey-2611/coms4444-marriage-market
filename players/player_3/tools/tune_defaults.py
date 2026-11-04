"""Grid-search Player3 flag combinations and pick defaults.

uv run python players/player_3/tools/tune_defaults.py --seeds 20
uv run python players/player_3/tools/tune_defaults.py --seeds 30 --confirm
"""

from __future__ import annotations

import argparse
import itertools
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
from players.player_3.player import _Player3Core  # noqa: E402

FLAG_NAMES = (
	'USE_TRUST',
	'USE_MARKET_BAR',
	'USE_CONFIDENCE',
	'USE_ROUNDS_LEFT_COMMIT',
	'USE_COORDINATION',
	'USE_STEER_LIST',
	'USE_INCOMPATIBLE',
)

ARENAS: list[tuple[str, list[tuple[str, int]]]] = [
	('self_play', [('t', 20)]),
	('vs_10', [('t', 10), ('10', 10)]),
	('vs_greedy', [('t', 10), ('g', 10)]),
	('vs_9', [('t', 10), ('9', 10)]),
	('vs_6', [('t', 10), ('6', 10)]),
]


def _make_player(flags: dict[str, bool]) -> type[Player]:
	attrs = {**flags, 'LIE_MODE': 'honest', 'AD_SEEK_MODE': 'costly', 'OWN_AD_INFLATE': 0}
	label = 'P3_' + ''.join('1' if flags[k] else '0' for k in FLAG_NAMES)
	return type(label, (_Player3Core,), attrs)


def _run(job: dict) -> dict:
	cls = _make_player(job['flags'])
	available, _ = discover()
	roster: list[type[Player]] = []
	for code, count in job['players']:
		if code == 't':
			roster.extend([cls] * count)
		else:
			roster.extend([available[code]] * count)

	random.seed(job['seed'])
	results = Engine(
		players=roster,
		rounds=job['rounds'],
		bonus=job['bonus'],
		dims=5,
		monotonic=2,
		seed=job['seed'],
		distribution='mixed',
		timeout=0.0,
		keep_records=False,
	).run()

	name = cls.__name__
	subset = [p for p in results['players'] if p['name'] == name]
	scores = [float(p['happiness']) for p in subset]
	pos = sum(1 for s in scores if s > 0) / len(scores) if scores else 0.0
	return {
		'arena': job['arena'],
		'seed': job['seed'],
		'positive_rate': pos,
		'mean_happiness': statistics.fmean(scores) if scores else 0.0,
		'flags_key': job['flags_key'],
	}


def _flags_key(flags: dict[str, bool]) -> str:
	return ''.join('1' if flags[k] else '0' for k in FLAG_NAMES)


def build_jobs(seeds: list[int], flags_list: list[dict[str, bool]]) -> list[dict]:
	jobs = []
	for flags in flags_list:
		key = _flags_key(flags)
		for arena, players in ARENAS:
			for seed in seeds:
				jobs.append(
					{
						'arena': arena,
						'flags': flags,
						'flags_key': key,
						'players': players,
						'rounds': 100,
						'bonus': 100.0,
						'seed': seed,
					}
				)
	return jobs


def aggregate(rows: list[dict]) -> list[dict]:
	by_flag: dict[str, list[dict]] = {}
	for row in rows:
		by_flag.setdefault(row['flags_key'], []).append(row)

	summaries = []
	for key, items in by_flag.items():
		by_arena: dict[str, list[dict]] = {}
		for item in items:
			by_arena.setdefault(item['arena'], []).append(item)
		arena_pos = {
			a: statistics.fmean(x['positive_rate'] for x in xs) for a, xs in by_arena.items()
		}
		arena_hap = {
			a: statistics.fmean(x['mean_happiness'] for x in xs) for a, xs in by_arena.items()
		}
		mean_pos = statistics.fmean(arena_pos.values())
		min_pos = min(arena_pos.values())
		mean_hap = statistics.fmean(arena_hap.values())
		summaries.append(
			{
				'flags_key': key,
				'mean_positive_rate': mean_pos,
				'min_positive_rate': min_pos,
				'mean_happiness': mean_hap,
				'by_arena_pos': arena_pos,
				'by_arena_hap': arena_hap,
			}
		)
	# Primary: max min_pos (robust), then mean_pos, then mean_hap
	summaries.sort(
		key=lambda s: (s['min_positive_rate'], s['mean_positive_rate'], s['mean_happiness'])
	)
	return summaries


def flags_from_key(key: str) -> dict[str, bool]:
	return {name: key[i] == '1' for i, name in enumerate(FLAG_NAMES)}


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--seeds', type=int, default=20)
	parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() - 1))
	parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'player_3_tune.json')
	args = parser.parse_args()

	seeds = list(range(1, args.seeds + 1))
	all_flags = [
		dict(zip(FLAG_NAMES, combo, strict=True))
		for combo in itertools.product((False, True), repeat=len(FLAG_NAMES))
	]
	jobs = build_jobs(seeds, all_flags)
	with mp.Pool(args.workers) as pool:
		rows = pool.map(_run, jobs)

	ranked = aggregate(rows)
	best = ranked[-1]

	args.out.parent.mkdir(parents=True, exist_ok=True)
	report = {
		'seeds': len(seeds),
		'best': best,
		'best_flags': flags_from_key(best['flags_key']),
		'top10': ranked[-10:][::-1],
		'all': ranked,
	}
	args.out.write_text(json.dumps(report, indent=2) + '\n')

	print(f'Wrote {args.out}')
	print('Best flags:', report['best_flags'])
	print(
		f'min_pos={best["min_positive_rate"]:.3f} '
		f'mean_pos={best["mean_positive_rate"]:.3f} '
		f'mean_hap={best["mean_happiness"]:.1f}'
	)
	for arena, pos in sorted(best['by_arena_pos'].items()):
		print(f'  {arena}: pos={pos:.3f} hap={best["by_arena_hap"][arena]:.1f}')


if __name__ == '__main__':
	main()
