"""Measure one group's player over many seeds.

Takes the same flags as main.py, plus a few of its own:

    uv run tools/evaluate.py --player 1 2 --player 2 2 ... --player 10 2 --rounds 30 --bonus 150

    --seeds N     how many seeds to run, starting at --seed (default 30)
    --ours CODE   the group to highlight (default 1)
    --no-solo     skip the market where everyone runs our code
    --csv PATH    raw per-person rows (default results/evaluate.csv); a Markdown summary
                  ready to post is written beside it (results/evaluate_summary.md)
    --jobs N      parallel processes (default: CPU count)

Runs two markets per seed:
  mixed  the roster you passed
  solo   everyone runs our code, same N and same people (profiles depend only on seed and index)

Reports, per group: positive rate (handout's primary goal), mean happiness (secondary goal),
married rate, bad marriages (married but happiness <= 0), mean marriage round, faults.
"""

import argparse
import csv
import math
import os
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.distributions import distribution_from_settings  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.registry import discover  # noqa: E402
from models.cli import settings  # noqa: E402

FAULT_INDEX = re.compile(r'^\S+?\[(\d+)\]')


def parse_args(argv: list[str]):
	ours = argparse.ArgumentParser(add_help=False)
	ours.add_argument('--seeds', type=int, default=30)
	ours.add_argument('--ours', default='1')
	ours.add_argument('--no-solo', action='store_true')
	ours.add_argument('--csv', default=str(ROOT / 'results' / 'evaluate.csv'))
	ours.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
	extra, rest = ours.parse_known_args(argv)
	return extra, settings(rest)


def run_one(job: dict) -> list[dict]:
	"""One (market, seed). Runs in a worker process, so it takes player codes, not classes."""
	available, _ = discover()
	roster = [available[code] for code, count in job['players'] for _ in range(count)]
	s = job['settings']
	distribution, distribution_name = distribution_from_settings(
		distribution=s.distribution,
		dims=s.dims,
		monotonic=s.monotonic,
		dist_file=s.dist_file,
		dim_dist=s.dim_dist,
		dim_dist_file=s.dim_dist_file,
	)
	results = Engine(
		players=roster,
		rounds=s.rounds,
		bonus=s.bonus,
		dims=s.dims,
		monotonic=s.monotonic,
		seed=job['seed'],
		distribution=distribution,
		distribution_name=distribution_name,
		timeout=s.timeout,
		keep_records=False,
		list_multiple=s.list_multiple,
	).run()

	faults: dict[int, int] = {}
	for text in results['faults']:
		match = FAULT_INDEX.match(text)
		if match:
			index = int(match.group(1))
			faults[index] = faults.get(index, 0) + 1

	return [
		{
			'market': job['market'],
			'seed': job['seed'],
			'index': p['index'],
			'group': p['name'],
			'happiness': p['happiness'],
			'positive': p['positive'],
			'married': p['married'],
			'married_on': p['married_on'],
			'faults': faults.get(p['index'], 0),
		}
		for p in results['players']
	]


def ci(values: list[float]) -> float:
	"""Half-width of a 95% interval for the mean."""
	if len(values) < 2:
		return 0.0
	return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def summarise(rows: list[dict], bonus: float) -> dict:
	happy = [r['happiness'] for r in rows]
	positive = [1.0 if r['positive'] else 0.0 for r in rows]
	married = [r for r in rows if r['married']]
	rounds = [r['married_on'] for r in married]
	married_happy = [r['happiness'] for r in married]
	return {
		'n': len(rows),
		'positive': statistics.fmean(positive),
		'positive_ci': ci(positive),
		'mean': statistics.fmean(happy),
		'mean_ci': ci(happy),
		'married': len(married) / len(rows),
		'married_mean': statistics.fmean(married_happy) if married else float('nan'),
		'weak': (
			sum(1 for h in married_happy if h < 0.5 * bonus) / len(married) if married else 0.0
		),
		'bad': sum(1 for r in married if r['happiness'] <= 0) / len(rows),
		'round': statistics.fmean(rounds) if rounds else float('nan'),
		'faults': sum(r['faults'] for r in rows),
	}


def by_group(rows: list[dict], market: str, bonus: float) -> dict[str, dict]:
	groups: dict[str, list[dict]] = {}
	for r in rows:
		if r['market'] == market:
			groups.setdefault(r['group'], []).append(r)
	return {name: summarise(g, bonus) for name, g in groups.items()}


HEADER = (
	f'{"rank":>4}  {"group":<14}{"positive":>14}{"mean happiness":>20}'
	f'{"married":>9}{"if married":>11}{"weak":>6}{"bad":>6}{"round":>7}{"faults":>8}'
)


def line(rank: str, name: str, s: dict, mark: str = '') -> str:
	return (
		f'{rank:>4}  {name:<14}'
		f'{s["positive"]:>8.0%} +/-{s["positive_ci"]:<4.0%}'
		f'{s["mean"]:>12.1f} +/-{s["mean_ci"]:<6.1f}'
		f'{s["married"]:>8.0%}{s["married_mean"]:>11.1f}{s["weak"]:>6.0%}'
		f'{s["bad"]:>6.0%}{s["round"]:>7.1f}{s["faults"]:>8}{mark}'
	)


MD_COLUMNS = (
	'| Positive | Mean happiness | Married | Avg if married | Weak marriages '
	'| Bad marriages | Avg round | Faults |'
)
MD_RULE = '|---:|---:|---:|---:|---:|---:|---:|---:|'
MD_HEADER = f'| Rank | Group {MD_COLUMNS}\n|---:|---{MD_RULE}'
MD_MARKET_HEADER = f'| Market {MD_COLUMNS}\n|---{MD_RULE}'


def md_row(rank: str, name: str, s: dict) -> str:
	lead = f'| {rank} | {name} ' if rank else f'| {name} '
	return (
		f'{lead}| {s["positive"]:.0%} ± {s["positive_ci"]:.0%} '
		f'| {s["mean"]:.1f} ± {s["mean_ci"]:.1f} | {s["married"]:.0%} | {s["married_mean"]:.1f} '
		f'| {s["weak"]:.0%} | {s["bad"]:.0%} '
		f'| {s["round"]:.1f} | {s["faults"]} |'
	)


def write_summary(path: Path, config: str, ranked: list, ours_name: str, solo: dict | None) -> None:
	"""A Markdown summary that can be pasted into GitHub, Ed or chat."""
	mixed = dict(ranked)
	lines = [
		f'# {ours_name} evaluation',
		'',
		f'**Config:** {config}',
		'',
		f'**Command:** `uv run tools/evaluate.py {" ".join(sys.argv[1:])}`',
		'',
		'## All groups, mixed market',
		'',
		MD_HEADER,
	]
	for rank, (name, stats) in enumerate(ranked, 1):
		us = name == ours_name
		row = md_row(str(rank), f'**{name} (us)**' if us else name, stats)
		lines.append(row)
	if solo is not None:
		lines += [
			'',
			f'## {ours_name}: mixed market vs everyone running our code',
			'',
			'Same seeds and same people in both rows.',
			'',
			MD_MARKET_HEADER,
		]
		if ours_name in mixed:
			lines.append(md_row('', 'mixed (all groups)', mixed[ours_name]))
		lines.append(md_row('', f'solo (all {ours_name})', solo))
	lines += [
		'',
		'*Positive* = share ending above 0 (primary goal). *Mean happiness* counts singles as 0 '
		'(secondary goal). *Avg if married* = mean over married people only. '
		'*Weak marriages* = share of marriages scoring under half the bonus. '
		'*Bad marriages* = married but ending at or below 0. '
		'± = 95% interval. Ranked by positive rate, then mean happiness.',
		'',
	]
	path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
	extra, s = parse_args(sys.argv[1:])
	available, errors = discover()
	for err in errors:
		print(f"warning: player '{err.code}' failed to load - {err.message}", file=sys.stderr)

	roster = []
	for code, count in s.players:
		if code not in available:
			print(f"warning: unknown or unloadable player '{code}', skipping", file=sys.stderr)
			continue
		roster.append((code, count))
	if extra.ours not in available:
		raise SystemExit(f"error: --ours '{extra.ours}' is not a loadable player")
	ours_name = available[extra.ours].__name__
	people = sum(count for _, count in roster)
	if people < 2 or people % 2:
		raise SystemExit(f'error: N must be even and at least 2, got {people}')

	seeds = list(range(s.seed, s.seed + extra.seeds))
	markets = {'mixed': roster}
	if not extra.no_solo and roster != [(extra.ours, people)]:
		markets['solo'] = [(extra.ours, people)]

	jobs = [
		{'market': m, 'seed': seed, 'players': r, 'settings': s}
		for m, r in markets.items()
		for seed in seeds
	]
	roster_text = ', '.join(f'{available[code].__name__} x{count}' for code, count in roster)
	config = (
		f'N={people} ({roster_text}), t={s.rounds}, B={s.bonus:g}, d={s.dims}, m={s.monotonic}, '
		f'dist={s.dist_file or s.dim_dist or s.dim_dist_file or s.distribution}, '
		f'seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds)'
	)
	print(f'Running {len(jobs)} games: {config}', flush=True)
	with ProcessPoolExecutor(max_workers=max(1, extra.jobs)) as pool:
		rows = [row for result in pool.map(run_one, jobs) for row in result]

	mixed = by_group(rows, 'mixed', s.bonus)
	ranked = sorted(mixed.items(), key=lambda kv: (-kv[1]['positive'], -kv[1]['mean']))
	samples = ', '.join(sorted({f'{v["n"]}' for v in mixed.values()}))
	print(f'\nMixed market (people per group across all seeds: {samples})')
	print(HEADER)
	for rank, (name, stats) in enumerate(ranked, 1):
		mark = '   <- us' if name == ours_name else ''
		print(line(f'{">" if mark else ""}{rank}', name, stats, mark))

	solo = by_group(rows, 'solo', s.bonus)[ours_name] if 'solo' in markets else None
	if solo is not None:
		print(f'\n{ours_name}: mixed market vs everyone running our code (same seeds, same people)')
		print(HEADER)
		if ours_name in mixed:
			print(line('', 'mixed', mixed[ours_name]))
		print(line('', 'solo', solo))

	print(
		'\npositive = share ending above 0 (primary goal) | mean = average happiness, singles count '
		'as 0\nif married = mean over married only | weak = share of marriages under half the '
		'bonus\nbad = married but <= 0 | round = average marriage round | +/- = 95% interval'
	)

	path = Path(extra.csv)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open('w', newline='') as fh:
		writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)
	summary = path.with_name(f'{path.stem}_summary.md')
	write_summary(summary, config, ranked, ours_name, solo)
	print(f'Raw rows: {path}\nSummary:  {summary}')


if __name__ == '__main__':
	main()
