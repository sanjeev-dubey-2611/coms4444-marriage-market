"""Sweep harness: run the engine across a grid of configurations and aggregate.

The simulator answers one question per invocation. The interesting questions
are comparative - what happens when B is very high, does a normally
distributed population marry differently from a uniform one - and a single
run cannot answer those, because the spread across seeds is wider than the
effect being measured. This runs each configuration over a set of seeds and
reports the distribution.

    uv run sweep.py configs/bonus_sweep.json
    uv run sweep.py configs/distribution_comparison.json --rounds 50 --seeds 1 2 3

Determinism is the point of the whole exercise, so nothing here is allowed to
disturb it. Cells run in separate processes, but each engine seeds its own
``random.Random``, results are sorted back into config order before being
aggregated, and no timestamp is written into the output. Two invocations over
the same seed set produce byte-identical files.
"""

import argparse
import csv
import json
import multiprocessing as mp
import random
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

from core.distributions import discover_distributions, distribution_from_sweep_value, warn_distribution_errors
from core.engine import Engine
from core.registry import discover

METRICS = (
	'mean_happiness',
	'positive_rate',
	'married_rate',
	'mean_married_happiness',
	'mean_married_round',
)

DEFAULTS = {
	'rounds': 100,
	'bonus': 100.0,
	'dims': 5,
	'monotonic': 2,
	'distribution': 'mixed',
	'timeout': 1.0,
	'list_multiple': 4.0,
}


@dataclass(frozen=True)
class Cell:
	"""One configuration in the grid, to be run over every seed."""

	label: str
	players: tuple[tuple[str, int], ...]
	rounds: int
	bonus: float
	dims: int
	monotonic: int
	distribution: Any
	timeout: float
	list_multiple: float = 4.0

	@property
	def people(self) -> int:
		return sum(count for _, count in self.players)

	@property
	def roster_text(self) -> str:
		return ' '.join(f'{code}x{count}' for code, count in self.players)

	@property
	def distribution_text(self) -> str:
		if isinstance(self.distribution, str):
			return self.distribution
		if isinstance(self.distribution, dict):
			if 'P' in self.distribution:
				return 'custom'
			parts = [
				f'{k}:{self.distribution[k]}'
				for k in sorted(self.distribution, key=lambda x: int(x))
			]
			return 'compose:' + ','.join(parts)
		if (
			isinstance(self.distribution, list | tuple)
			and self.distribution
			and isinstance(self.distribution[0], str)
		):
			return 'compose:' + ','.join(self.distribution)
		return 'custom'


@dataclass
class SweepConfig:
	name: str
	description: str
	seeds: list[int]
	cells: list[Cell]
	source: Path = field(default=Path('.'))


# ----------------------------------------------------------------- config


def _players_of(raw: Any) -> tuple[tuple[str, int], ...]:
	"""Normalise a players spec into (code, count) pairs.

	Accepts ``[['g', 10]]``, ``[{'code': 'g', 'count': 10}]`` or the shorthand
	``'g'`` for a single person, so configs stay readable.
	"""
	if isinstance(raw, str):
		return ((raw, 1),)

	pairs: list[tuple[str, int]] = []
	for entry in raw:
		if isinstance(entry, dict):
			pairs.append((str(entry['code']), int(entry['count'])))
		elif isinstance(entry, str):
			pairs.append((entry, 1))
		else:
			code, count = entry
			pairs.append((str(code), int(count)))
	return tuple(pairs)


def _distribution_of(raw: Any) -> Any:
	"""Keep presets, dim maps, and inline tables as stored in the config.

	Resolution to a concrete Engine distribution happens in ``validate`` /
	``_run_one``, so sweep labels can still show the dim map.
	"""
	if isinstance(raw, str):
		return raw
	if isinstance(raw, dict):
		# Dim map {"0": "3"} or {"P": [...]}. Keep as plain dict for labels.
		return {str(k): v for k, v in raw.items()}
	if isinstance(raw, list) and raw and isinstance(raw[0], str) and ':' in raw[0]:
		return list(raw)
	return tuple(tuple(float(p) for p in row) for row in raw)


def _cell_from(raw: dict, defaults: dict, index: int) -> Cell:
	merged = {**defaults, **raw}
	if 'players' not in merged:
		raise ValueError(f'cell {index} has no "players" and no default for it')

	return Cell(
		label=str(merged.get('label') or f'cell{index}'),
		players=_players_of(merged['players']),
		rounds=int(merged['rounds']),
		bonus=float(merged['bonus']),
		dims=int(merged['dims']),
		monotonic=int(merged['monotonic']),
		distribution=_distribution_of(merged['distribution']),
		timeout=float(merged['timeout']),
		list_multiple=float(merged['list_multiple']),
	)


def _label_part(value: Any) -> str:
	if isinstance(value, dict):
		if 'P' in value:
			return 'custom'
		return 'compose'
	if isinstance(value, list | tuple):
		if value and isinstance(value[0], str) and ':' in value[0]:
			return 'compose'
		try:
			return ''.join(f'{c}{n}' for c, n in _players_of(value))
		except (TypeError, ValueError):
			# An inline P table, not a roster.
			return 'custom'
	if isinstance(value, float) and value.is_integer():
		return str(int(value))
	return str(value)


def load_config(path: Path | str) -> SweepConfig:
	"""Read a sweep description. ``grid`` is expanded as a cross product,
	``cells`` are taken verbatim, and a config may use either or both."""
	path = Path(path)
	raw = json.loads(path.read_text())
	defaults = {**DEFAULTS, **raw.get('defaults', {})}

	cells: list[Cell] = []

	grid = raw.get('grid') or {}
	if grid:
		keys = sorted(grid)
		for combo in product(*(grid[k] for k in keys)):
			entry = dict(zip(keys, combo, strict=True))
			if 'label' not in entry:
				entry['label'] = ' '.join(f'{k}={_label_part(entry[k])}' for k in keys)
			cells.append(_cell_from(entry, defaults, len(cells)))

	for entry in raw.get('cells', []):
		cells.append(_cell_from(entry, defaults, len(cells)))

	if not cells:
		raise ValueError(f'{path} defines no cells (need "grid" or "cells")')

	seeds = [int(s) for s in raw.get('seeds', [4444])]
	if not seeds:
		raise ValueError(f'{path} defines an empty seed list')

	return SweepConfig(
		name=str(raw.get('name') or path.stem),
		description=str(raw.get('description', '')),
		seeds=seeds,
		cells=cells,
		source=path,
	)


# ----------------------------------------------------------------- running


def _run_one(job: dict) -> dict:
	"""Run a single (cell, seed) pair. Executed in a worker process.

	Takes player *codes* rather than classes: the registry resolves them
	inside the worker, so nothing student-authored has to survive pickling
	and a spawn-based process still finds the roster.
	"""
	available, _ = discover()
	roster = []
	for code, count in job['players']:
		roster.extend([available[code]] * count)

	# Seed global random as a compatibility aid for older student players;
	# new players should use their private self.rng.
	random.seed(job['seed'])

	distribution, distribution_name = distribution_from_sweep_value(
		job['distribution'],
		dims=job['dims'],
		monotonic=job['monotonic'],
	)

	results = Engine(
		players=roster,
		rounds=job['rounds'],
		bonus=job['bonus'],
		dims=job['dims'],
		monotonic=job['monotonic'],
		seed=job['seed'],
		distribution=distribution,
		distribution_name=distribution_name,
		timeout=job['timeout'],
		keep_records=False,
		list_multiple=job.get('list_multiple', 4.0),
	).run()

	married_rounds = [m['round'] for m in results['marriages']]
	return {
		'cell': job['cell'],
		'seed': job['seed'],
		'mean_happiness': results['mean_happiness'],
		'positive_rate': results['positive_rate'],
		'married_rate': results['married_rate'],
		'mean_married_happiness': results['mean_married_happiness'],
		'mean_married_round': statistics.fmean(married_rounds) if married_rounds else None,
		'rounds_played': results['rounds_played'],
		'faults': len(results['faults']),
	}


def build_jobs(config: SweepConfig) -> list[dict]:
	jobs = []
	for i, cell in enumerate(config.cells):
		for seed in config.seeds:
			jobs.append(
				{
					'cell': i,
					'seed': seed,
					'players': list(cell.players),
					'rounds': cell.rounds,
					'bonus': cell.bonus,
					'dims': cell.dims,
					'monotonic': cell.monotonic,
					'distribution': cell.distribution,
					'timeout': cell.timeout,
					'list_multiple': cell.list_multiple,
				}
			)
	return jobs


def validate(config: SweepConfig) -> None:
	"""Fail before spawning anything. A grid typo should not surface as ten
	identical tracebacks from ten workers."""
	available, errors = discover()
	for err in errors:
		print(f"warning: player '{err.code}' failed to load - {err.message}", file=sys.stderr)

	catalog, dist_errors = discover_distributions()
	warn_distribution_errors(dist_errors)

	for cell in config.cells:
		unknown = [code for code, _ in cell.players if code not in available]
		if unknown:
			codes = ', '.join(sorted(available)) or 'none'
			raise SystemExit(
				f"cell '{cell.label}' names unloadable player(s) {unknown}. Available: {codes}"
			)
		try:
			distribution, distribution_name = distribution_from_sweep_value(
				cell.distribution,
				dims=cell.dims,
				monotonic=cell.monotonic,
				catalog=catalog,
			)
			Engine(
				players=[available[cell.players[0][0]]] * cell.people,
				rounds=cell.rounds,
				bonus=cell.bonus,
				dims=cell.dims,
				monotonic=cell.monotonic,
				distribution=distribution,
				distribution_name=distribution_name,
				seed=0,
				timeout=0,
				list_multiple=cell.list_multiple,
			)
		except ValueError as exc:
			raise SystemExit(f"cell '{cell.label}' is not a legal configuration: {exc}") from exc


def execute(
	jobs: list[dict],
	workers: int,
	worker: Callable[[dict], dict] = _run_one,
	on_progress: Callable[[dict, int, int], None] | None = None,
) -> list[dict]:
	"""Run every job and return the results in job order.

	Workers hand back results as they finish, so the list is sorted before it
	goes anywhere near the aggregates. Parallelism must not be visible in the
	output.

	``worker`` is a parameter so the tournament runner can reuse this instead
	of growing its own copy of the pool, the progress line and the sort.
	"""
	done: list[dict] = []

	def take(result: dict) -> None:
		done.append(result)
		n = len(done)
		if on_progress is None:
			_track(n, len(jobs))
		else:
			on_progress(result, n, len(jobs))

	if workers <= 1 or len(jobs) == 1:
		for job in jobs:
			take(worker(job))
	else:
		ctx = mp.get_context('spawn')
		with ctx.Pool(processes=workers) as pool:
			for result in pool.imap_unordered(worker, jobs):
				take(result)

	return sorted(done, key=lambda r: (r['cell'], r['seed']))


def execute_live(
	jobs: list[dict],
	workers: int,
	worker: Callable[[dict], dict],
	on_progress: Callable[[dict, int, int], None],
	idle: Callable[[], None],
	pace: float = 0.0,
) -> list[dict]:
	"""Like ``execute``, but yields the event loop between results.

	The GUI has to keep pumping pygame while workers run, otherwise macOS
	shows a frozen first frame and then the finished table. ``pace`` is extra
	seconds to hold each new ranking on screen so a projector can actually see
	the numbers move.
	"""
	import time

	done: list[dict] = []
	ctx = mp.get_context('spawn')
	with ctx.Pool(processes=max(1, workers)) as pool:
		iterator = pool.imap_unordered(worker, jobs)
		while len(done) < len(jobs):
			idle()
			try:
				result = iterator.next(timeout=0.03)
			except mp.TimeoutError:
				continue
			done.append(result)
			on_progress(result, len(done), len(jobs))
			if pace > 0:
				hold = time.monotonic() + pace
				while time.monotonic() < hold:
					idle()

	return sorted(done, key=lambda r: (r['cell'], r['seed']))


def _track(done: int, total: int) -> None:
	print(f'\r  {done}/{total} runs complete', end='', file=sys.stderr, flush=True)
	if done == total:
		print(file=sys.stderr)


# ----------------------------------------------------------------- aggregation


def summarise(values: list[float | None]) -> dict:
	"""Mean, median, spread and range. ``None`` entries are skipped: a run in
	which nobody married has no mean married happiness, and treating that as
	zero would quietly drag the average toward it."""
	present = [v for v in values if v is not None]
	if not present:
		return {'mean': None, 'median': None, 'stddev': None, 'min': None, 'max': None}
	return {
		'mean': statistics.fmean(present),
		'median': statistics.median(present),
		'stddev': statistics.stdev(present) if len(present) > 1 else 0.0,
		'min': min(present),
		'max': max(present),
	}


def aggregate(config: SweepConfig, runs: list[dict]) -> dict:
	by_cell: dict[int, list[dict]] = {i: [] for i in range(len(config.cells))}
	for run in runs:
		by_cell[run['cell']].append(run)

	cells = []
	for i, cell in enumerate(config.cells):
		mine = by_cell[i]
		if not mine:
			continue
		cells.append(
			{
				'label': cell.label,
				'players': [list(p) for p in cell.players],
				'people': cell.people,
				'rounds': cell.rounds,
				'bonus': cell.bonus,
				'dims': cell.dims,
				'monotonic': cell.monotonic,
				'distribution': cell.distribution_text,
				'seeds': [r['seed'] for r in mine],
				'aggregates': {m: summarise([r[m] for r in mine]) for m in METRICS},
				'faults': sum(r['faults'] for r in mine),
				'runs': [{k: v for k, v in r.items() if k != 'cell'} for r in mine],
			}
		)

	return {
		'sweep': config.name,
		'description': config.description,
		'config': str(config.source),
		'seeds': config.seeds,
		'cells': cells,
	}


# ----------------------------------------------------------------- output


def _blank(value: Any) -> Any:
	return '' if value is None else value


def write_outputs(report: dict, out_dir: Path) -> list[Path]:
	out_dir.mkdir(parents=True, exist_ok=True)
	stem = report['sweep']

	json_path = out_dir / f'{stem}.json'
	json_path.write_text(json.dumps(report, indent=2) + '\n')

	csv_path = out_dir / f'{stem}.csv'
	header = [
		'label',
		'players',
		'people',
		'rounds',
		'bonus',
		'dims',
		'monotonic',
		'distribution',
		'seeds',
		'faults',
	]
	for metric in METRICS:
		header += [f'{metric}_{stat}' for stat in ('mean', 'median', 'stddev', 'min', 'max')]

	with csv_path.open('w', newline='') as fh:
		writer = csv.writer(fh)
		writer.writerow(header)
		for cell in report['cells']:
			row = [
				cell['label'],
				' '.join(f'{c}x{n}' for c, n in cell['players']),
				cell['people'],
				cell['rounds'],
				cell['bonus'],
				cell['dims'],
				cell['monotonic'],
				cell['distribution'],
				len(cell['seeds']),
				cell['faults'],
			]
			for metric in METRICS:
				stats = cell['aggregates'][metric]
				row += [_blank(stats[s]) for s in ('mean', 'median', 'stddev', 'min', 'max')]
			writer.writerow(row)

	runs_path = out_dir / f'{stem}_runs.csv'
	with runs_path.open('w', newline='') as fh:
		writer = csv.writer(fh)
		writer.writerow(['label', 'seed', *METRICS, 'rounds_played', 'faults'])
		for cell in report['cells']:
			for run in cell['runs']:
				writer.writerow(
					[
						cell['label'],
						run['seed'],
						*(_blank(run[m]) for m in METRICS),
						run['rounds_played'],
						run['faults'],
					]
				)

	return [json_path, csv_path, runs_path]


def _pct(value: float | None) -> str:
	return '-' if value is None else f'{100 * value:.0f}%'


def print_table(report: dict) -> None:
	width = max(len(c['label']) for c in report['cells'])
	print(f'\n{report["sweep"]}  ({len(report["seeds"])} seeds)')
	print(
		f'{"cell":<{width}}  {"happiness":>16}  {"positive":>8}  {"married":>8}  '
		f'{"wed round":>9}  faults'
	)
	for cell in report['cells']:
		agg = cell['aggregates']
		happy = agg['mean_happiness']
		happy_text = f'{happy["mean"]:,.1f} +/- {happy["stddev"]:,.1f}'
		wed = agg['mean_married_round']['mean']
		wed_text = '-' if wed is None else f'{wed:.1f}'
		print(
			f'{cell["label"]:<{width}}  {happy_text:>16}  '
			f'{_pct(agg["positive_rate"]["mean"]):>8}  {_pct(agg["married_rate"]["mean"]):>8}  '
			f'{wed_text:>9}  {cell["faults"]:>6}'
		)


# ----------------------------------------------------------------- cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description='Run the marriage market simulator across a grid of configurations.'
	)
	parser.add_argument(
		'config', type=Path, help='Sweep description, e.g. configs/bonus_sweep.json'
	)
	parser.add_argument(
		'--out-dir', type=Path, default=Path('results'), help='Where to write JSON and CSV.'
	)
	parser.add_argument('--seeds', type=int, nargs='+', help='Override the seed list.')
	parser.add_argument('--rounds', type=int, help='Override t for every cell.')
	parser.add_argument('--bonus', type=float, help='Override B for every cell.')
	parser.add_argument('--timeout', type=float, help='Override the player time budget.')
	parser.add_argument(
		'--list-multiple',
		type=float,
		help='Override the returned interest-list limit; 0 disables.',
	)
	parser.add_argument(
		'--workers',
		type=int,
		default=0,
		help='Worker processes. 0 picks one per CPU, 1 runs sequentially.',
	)
	return parser.parse_args(argv)


def apply_overrides(config: SweepConfig, args: argparse.Namespace) -> SweepConfig:
	if args.seeds:
		config.seeds = args.seeds

	changes = {
		key: getattr(args, key)
		for key in ('rounds', 'bonus', 'timeout', 'list_multiple')
		if getattr(args, key) is not None
	}
	if changes:
		config.cells = [replace(cell, **changes) for cell in config.cells]
	return config


def main(argv: list[str] | None = None) -> None:
	args = parse_args(argv)
	if not args.config.exists():
		raise SystemExit(f'no such config: {args.config}')

	config = apply_overrides(load_config(args.config), args)
	validate(config)

	jobs = build_jobs(config)
	workers = args.workers or min(len(jobs), mp.cpu_count())
	print(
		f'{config.name}: {len(config.cells)} cells x {len(config.seeds)} seeds '
		f'= {len(jobs)} runs on {workers} worker(s)',
		file=sys.stderr,
	)

	runs = execute(jobs, workers)
	report = aggregate(config, runs)
	written = write_outputs(report, args.out_dir)

	print_table(report)
	print('\nwrote:')
	for path in written:
		print(f'  {path}')


if __name__ == '__main__':
	main()
