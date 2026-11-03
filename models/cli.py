import argparse
from dataclasses import dataclass
from pathlib import Path

from models.profile import PRESETS, dims_from_dim_map, load_dim_map, parse_dim_map

# Used when ``--dim-dist-file`` is passed with no path.
DEFAULT_DIM_DIST_FILE = str(
	Path(__file__).resolve().parent.parent / 'configs' / 'dim_maps' / 'default.json'
)

DEFAULT_DIMS = 5


@dataclass
class Settings:
	players: list[tuple[str, int]]
	rounds: int
	bonus: float
	dims: int
	monotonic: int
	distribution: str
	dist_file: str | None
	dim_dist: str | None
	dim_dist_file: str | None
	seed: int
	timeout: float
	summary_only: bool
	gui: bool
	log: str | None
	list_multiple: float = 4.0


def settings(argv: list[str] | None = None) -> Settings:
	parser = argparse.ArgumentParser(description='Run a marriage market simulation.')
	parser.add_argument(
		'--player',
		action='append',
		nargs=2,
		metavar=('CODE', 'COUNT'),
		help='Add COUNT people running player CODE (e.g. --player g 10). Repeatable. '
		'N is the total count and must be even.',
	)
	parser.add_argument('-t', '--rounds', type=int, default=100, help='Rounds of dating, t.')
	parser.add_argument('-B', '--bonus', type=float, default=100.0, help='Marriage bonus B.')
	parser.add_argument(
		'-d',
		'--dims',
		type=int,
		default=None,
		help=f'Profile dimensions, d (default {DEFAULT_DIMS}; with --dim-dist / '
		'--dim-dist-file, d is taken from the map instead).',
	)
	parser.add_argument(
		'-m',
		'--monotonic',
		type=int,
		default=2,
		help='How many of the first dimensions are monotonic, m <= d.',
	)
	parser.add_argument(
		'--dist',
		choices=PRESETS,
		default='mixed',
		dest='distribution',
		help='Preset distribution P for every dimension.',
	)
	parser.add_argument(
		'--dist-file',
		default=None,
		metavar='PATH',
		help='JSON file holding P as d rows of 11 probabilities. Overrides --dist '
		'and --dim-dist / --dim-dist-file.',
	)
	parser.add_argument(
		'--dim-dist',
		default=None,
		metavar='MAP',
		help='Per-dimension distribution map, e.g. 0:3,1:normal,2:7. Also sets d to '
		'the number of listed dimensions (must cover 0..d-1). Sources are presets '
		'(uniform/normal/mixed) or a group code whose distribution.json was '
		'discovered. Overrides --dist; overridden by --dist-file.',
	)
	parser.add_argument(
		'--dim-dist-file',
		nargs='?',
		const=DEFAULT_DIM_DIST_FILE,
		default=None,
		metavar='PATH',
		help='Same map as --dim-dist, from a JSON file {"0":"3","1":"normal"}. '
		'Pass the flag alone to use configs/dim_maps/default.json. '
		'd is taken from the map. Overrides --dist; overridden by --dist-file.',
	)
	parser.add_argument('--seed', type=int, default=4444, help='Random seed.')
	parser.add_argument(
		'--list-multiple',
		type=float,
		default=4.0,
		help='Limit returned interest lists to ceil(MULTIPLE * sqrt(N)), capped at N-1. All eligible candidates remain visible. Default 4; 0 disables the interest limit.',
	)
	parser.add_argument(
		'--timeout',
		type=float,
		default=1.0,
		help='Per-call wall-clock budget for player code, in seconds. 0 disables.',
	)
	parser.add_argument(
		'--summary-only',
		action='store_true',
		help="Discard each round's RoundRecord once it has been played. The results JSON is "
		'unchanged; a long run just stops holding one record per round in memory.',
	)
	parser.add_argument('--gui', action='store_true', help='Launch the visualizer.')
	parser.add_argument(
		'--log',
		default='',
		metavar='PATH',
		help='Write a per-round debug log (default: logs/marriage-*.log). Pass a path to '
		'choose the file. Sweeps do not write one.',
	)
	parser.add_argument(
		'--no-log',
		action='store_true',
		help='Do not write a debug log.',
	)

	args = parser.parse_args(argv)

	pairs: list[tuple[str, int]] = []
	for code, count in args.player or [('g', '10'), ('r', '10')]:
		pairs.append((code, int(count)))

	if args.dim_dist and args.dim_dist_file:
		parser.error('use only one of --dim-dist and --dim-dist-file')

	dims = args.dims
	if args.dim_dist_file or args.dim_dist:
		try:
			dim_map = (
				load_dim_map(args.dim_dist_file)
				if args.dim_dist_file
				else parse_dim_map(args.dim_dist)
			)
			inferred = dims_from_dim_map(dim_map)
		except ValueError as exc:
			parser.error(str(exc))
		if dims is not None and dims != inferred:
			parser.error(
				f'--dims={dims} conflicts with dim map (lists dimensions 0..{inferred - 1}, so d={inferred})'
			)
		dims = inferred
	elif dims is None:
		dims = DEFAULT_DIMS

	monotonic = args.monotonic
	if monotonic > dims:
		# Dim maps often have d < the default m=2; clamp instead of failing the run.
		monotonic = dims

	return Settings(
		players=pairs,
		rounds=args.rounds,
		bonus=args.bonus,
		dims=dims,
		monotonic=monotonic,
		distribution=args.distribution,
		dist_file=args.dist_file,
		dim_dist=args.dim_dist,
		dim_dist_file=args.dim_dist_file,
		seed=args.seed,
		timeout=args.timeout,
		summary_only=args.summary_only,
		gui=args.gui,
		log=None if args.no_log else args.log,
		list_multiple=args.list_multiple,
	)
