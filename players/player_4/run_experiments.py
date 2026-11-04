"""Run headless simulations from a list of configurations and log each to CSV.

Edit RUNS below, then from the repo root:

    uv run python -m players.player_4.run_experiments

Each run writes one CSV to players/player_4/experiment_logs/.
"""

import csv
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from core.engine import Engine
from core.registry import discover

LOG_DIR = Path(__file__).parent / 'experiment_logs'

# Player codes, in the same order as Run.players counts: groups 1-10, greedy, random.
PLAYER_CODES = [str(group) for group in range(1, 11)] + ['g', 'r']


@dataclass
class Run:
	"""One simulation. ``players[i]`` is how many people run PLAYER_CODES[i]."""

	days: int = 40
	marriage_bonus: float = 200
	players: list[int] = field(default_factory=lambda: [2 for _ in range(12)])
	dims: int = 5
	monotonic: int = 2
	distribution: str = 'mixed'
	seed: int = 4444


# Edit this list to choose which runs to do.
RUNS = [
	Run(),
	Run(seed=1),
	Run(seed=2),
]


def build_roster(run: Run) -> tuple[list, list[str]]:
	"""Player classes for the engine, plus the code each person runs."""
	available, _ = discover()
	roster, codes = [], []
	for code, count in zip(PLAYER_CODES, run.players, strict=True):
		if count and code not in available:
			print(f'warning: player {code!r} is not loadable, skipping', file=sys.stderr)
			continue
		roster.extend([available[code]] * count)
		codes.extend([code] * count)
	return roster, codes


def run_one(run: Run) -> Path:
	"""Run one simulation and write its CSV. Returns the CSV path."""
	roster, codes = build_roster(run)
	results = Engine(
		players=roster,
		rounds=run.days,
		bonus=run.marriage_bonus,
		dims=run.dims,
		monotonic=run.monotonic,
		seed=run.seed,
		distribution=run.distribution,
		keep_records=False,
	).run()

	LOG_DIR.mkdir(exist_ok=True)
	timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
	path = LOG_DIR / f'run-{timestamp}-seed{run.seed}.csv'
	with path.open('w', newline='') as file:
		# Comment line with the run's parameters; the analytics script skips it.
		file.write('# ' + ', '.join(f'{key}={value}' for key, value in asdict(run).items()) + '\n')
		writer = csv.writer(file)
		writer.writerow(['player_index', 'team', 'score', 'married_day', 'married_to'])
		for person in results['players']:
			married_day = person['married_on'] if person['married_on'] is not None else -1
			married_to = person['partner'] if person['partner'] is not None else ''
			writer.writerow(
				[
					person['index'],
					codes[person['index']],
					person['happiness'],
					married_day,
					married_to,
				]
			)
	return path


def main() -> None:
	for number, run in enumerate(RUNS, start=1):
		path = run_one(run)
		print(f'[{number}/{len(RUNS)}] wrote {path}')


if __name__ == '__main__':
	main()
