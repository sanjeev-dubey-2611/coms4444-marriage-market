import json
import sys

from core.distributions import distribution_from_settings
from core.engine import Engine
from core.registry import discover
from models.cli import settings
from models.player import Player


def build_roster(requested: list[tuple[str, int]]) -> list[type[Player]]:
	available, errors = discover()

	for err in errors:
		print(f"warning: player '{err.code}' failed to load - {err.message}", file=sys.stderr)

	roster: list[type[Player]] = []
	for code, count in requested:
		if code not in available:
			print(f"warning: unknown or unloadable player '{code}', skipping", file=sys.stderr)
			continue
		roster.extend([available[code]] * count)

	if not roster:
		codes = ', '.join(sorted(available)) or 'none'
		raise SystemExit(f'No usable players selected. Available codes: {codes}')

	return roster


def _open_log(args, engine):
	"""Best-effort. A missing logs/ directory must not take down a live demo."""
	if args.log is None:
		return None
	from core.runlog import RunLog

	try:
		log = RunLog.create(args.log, engine)
	except OSError as exc:
		print(f'warning: could not write log - {exc}', file=sys.stderr)
		return None
	print(f'log: {log.path}', file=sys.stderr)
	print(f'csv: {log.csv_path}', file=sys.stderr)
	return log


def main() -> None:
	args = settings()
	roster = build_roster(args.players)

	try:
		distribution, distribution_name = distribution_from_settings(
			distribution=args.distribution,
			dims=args.dims,
			monotonic=args.monotonic,
			dist_file=args.dist_file,
			dim_dist=args.dim_dist,
			dim_dist_file=args.dim_dist_file,
		)
	except ValueError as exc:
		raise SystemExit(f'error: {exc}') from exc

	options = dict(
		players=roster,
		rounds=args.rounds,
		bonus=args.bonus,
		dims=args.dims,
		monotonic=args.monotonic,
		seed=args.seed,
		distribution=distribution,
		distribution_name=distribution_name,
		timeout=args.timeout,
		keep_records=not args.summary_only,
		list_multiple=args.list_multiple,
	)
	if args.gui:
		from core.gui_worker import SimulationWorker
		from ui.gui import run_gui

		# GUI only needs the latest display state; every round still reaches logs.
		options['keep_records'] = False
		worker = SimulationWorker(options, args.log)
		try:
			run_gui(worker=worker)
		except RuntimeError as exc:
			raise SystemExit(f'error: {exc}') from exc
		finally:
			worker.close()
		return

	try:
		engine = Engine(**options)
	except ValueError as exc:
		raise SystemExit(f'error: {exc}') from exc

	log = _open_log(args, engine)
	on_round = None
	if log is not None:
		from core.runlog import bind

		on_round = bind(log, engine)

	try:
		print(json.dumps(engine.run(on_round=on_round), indent=2))
	finally:
		if log is not None:
			log.finish(engine)


if __name__ == '__main__':
	main()
