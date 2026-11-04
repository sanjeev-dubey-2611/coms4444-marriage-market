"""Compare Player3 reports using matched configurations and seeds.

Example:
    uv run python players/player_3/tools/compare_reports.py before.json after.json

All deltas are after minus before. Rate deltas are fractions, not percentages.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from players.player_3.tools.evaluate import CONFIG_FIELDS, _bootstrap_ci  # noqa: E402

METRICS = ('positive_rate', 'mean_happiness', 'married_rate', 'faults', 'player_faults')


def _fixed_context(report: dict) -> dict:
	metadata = report.get('metadata')
	if not isinstance(metadata, dict):
		raise ValueError('missing experiment provenance metadata')
	hashes = metadata.get('source_sha256')
	if not isinstance(hashes, dict) or not hashes:
		raise ValueError('missing experiment source fingerprints')
	if not {'core/engine.py', 'players/player_3/player.py'} <= hashes.keys():
		raise ValueError('missing simulator or Player3 fingerprint')
	if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
		raise ValueError('invalid source fingerprint')
	context = {}
	for field in ('python_version', 'platform'):
		if not isinstance(metadata.get(field), str) or not metadata[field]:
			raise ValueError(f'missing runtime metadata: {field}')
		context[field] = metadata[field]
	context['source_sha256'] = {
		path: digest for path, digest in hashes.items() if not path.startswith('players/player_3/')
	}
	return context


def _index_rows(report: dict, variant: str) -> dict[tuple[str, int], dict]:
	if not isinstance(report.get('raw'), list):
		raise ValueError('report must contain a raw result list')
	indexed = {}
	for row in report['raw']:
		if row.get('variant') != variant:
			continue
		if not isinstance(row.get('label'), str) or type(row.get('seed')) is not int:
			raise ValueError('each selected row needs a label and integer seed')
		key = (row['label'], row['seed'])
		if key in indexed:
			raise ValueError(f'duplicate run: {key}')
		config = row.get('config')
		if not isinstance(config, dict) or any(field not in config for field in CONFIG_FIELDS):
			raise ValueError(f'missing full configuration: {key}')
		if type(row.get('n')) is not int or row['n'] < 1:
			raise ValueError(f'invalid player count: {key}')
		for metric in METRICS:
			value = row.get(metric)
			if not isinstance(value, (int, float)) or not math.isfinite(value):
				raise ValueError(f'missing or nonfinite {metric}: {key}')
		indexed[key] = row
	if not indexed:
		raise ValueError(f'no runs selected for variant {variant!r}')
	return indexed


def compare_reports(
	before: dict,
	after: dict,
	*,
	before_variant: str = 'default',
	after_variant: str = 'default',
) -> dict:
	"""Reject unmatched samples rather than comparing different seed populations."""
	if _fixed_context(before) != _fixed_context(after):
		raise ValueError(
			'incompatible runtime, simulator, dependency lock or opponent source versions'
		)
	left = _index_rows(before, before_variant)
	right = _index_rows(after, after_variant)
	if left.keys() != right.keys():
		raise ValueError(
			f'unmatched runs: missing after={sorted(left.keys() - right.keys())}, '
			f'missing before={sorted(right.keys() - left.keys())}'
		)
	groups: dict[str, list[tuple[dict, dict, dict]]] = {}
	paired = []
	for key in sorted(left):
		a, b = left[key], right[key]
		if a['config'] != b['config'] or a['n'] != b['n']:
			raise ValueError(f'incompatible configuration or player count: {key}')
		delta = {metric: b[metric] - a[metric] for metric in METRICS}
		paired.append({'label': key[0], 'seed': key[1], **delta})
		groups.setdefault(key[0], []).append((a, b, delta))

	by_cell = []
	for label, pairs in sorted(groups.items()):
		metrics = {}
		for metric in METRICS:
			deltas = [delta[metric] for _, _, delta in pairs]
			lo, hi = _bootstrap_ci(deltas)
			metrics[metric] = {
				'before': statistics.fmean(a[metric] for a, _, _ in pairs),
				'after': statistics.fmean(b[metric] for _, b, _ in pairs),
				'delta': statistics.fmean(deltas),
				'delta_ci_lo': lo,
				'delta_ci_hi': hi,
			}
		by_cell.append({'label': label, 'pairs': len(pairs), 'metrics': metrics})
	return {
		'before_variant': before_variant,
		'after_variant': after_variant,
		'direction': 'after - before',
		'pairs': len(paired),
		'by_cell': by_cell,
		'paired_deltas': paired,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('before', type=Path)
	parser.add_argument('after', type=Path)
	parser.add_argument('--before-variant', default='default')
	parser.add_argument('--after-variant', default='default')
	args = parser.parse_args()
	try:
		report = compare_reports(
			json.loads(args.before.read_text(encoding='utf-8')),
			json.loads(args.after.read_text(encoding='utf-8')),
			before_variant=args.before_variant,
			after_variant=args.after_variant,
		)
	except (OSError, ValueError, KeyError, TypeError) as exc:
		parser.error(str(exc))
	print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
	main()
