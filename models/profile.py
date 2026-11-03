"""Profiles, the value distribution P, and the happiness arithmetic.

Everything here is public knowledge. The handout gives every player the
scoring formula, so ``decrement`` and ``happiness`` are importable from player
code; what a player does not get is anybody else's true profile, nor the
distribution P a given simulation drew its people from.
"""

import json
import math
import random
from pathlib import Path

MIN_VALUE = 0
MAX_VALUE = 10
VALUES = tuple(range(MIN_VALUE, MAX_VALUE + 1))

Profile = tuple[int, ...]
Distribution = tuple[tuple[float, ...], ...]

PRESETS = ('uniform', 'normal', 'mixed')

# Rows of P are allowed this much slack when checking they sum to 1, so a
# hand-written config with values like 0.1 x 10 + 0.0 does not trip on float
# rounding.
SUM_TOLERANCE = 1e-6

DistributionRow = tuple[float, ...]


def uniform_row() -> DistributionRow:
	return tuple(1.0 / len(VALUES) for _ in VALUES)


def normal_row(mean: float = 5.0, sd: float = 2.0) -> DistributionRow:
	"""A discretised normal over 0-10, renormalised so the row sums to 1."""
	weights = [math.exp(-((v - mean) ** 2) / (2 * sd * sd)) for v in VALUES]
	total = sum(weights)
	return tuple(w / total for w in weights)


def build_distribution(name: str, dims: int, monotonic: int) -> Distribution:
	"""One of the named presets.

	``mixed`` is the default and the one closest to the handout's description:
	monotonic dimensions are roughly normal (most people are middling, a 10 is
	rare), and the remaining dimensions alternate between uniform and normal.
	"""
	if name == 'uniform':
		return tuple(uniform_row() for _ in range(dims))
	if name == 'normal':
		return tuple(normal_row() for _ in range(dims))
	if name == 'mixed':
		rows = []
		for i in range(dims):
			if i < monotonic:
				rows.append(normal_row(5.0, 2.0))
			elif (i - monotonic) % 2 == 0:
				rows.append(uniform_row())
			else:
				rows.append(normal_row(5.0, 2.5))
		return tuple(rows)
	raise ValueError(f'unknown distribution {name!r}; choose one of {", ".join(PRESETS)}')


def validate_row(row: object, *, label: str = 'P') -> DistributionRow:
	"""Check one dimension's PMF: 11 finite nonnegative probabilities summing to 1."""
	if not isinstance(row, (list, tuple)) or isinstance(row, (str, bytes)):
		raise ValueError(f'{label} must be a list of {len(VALUES)} probabilities')
	parsed = tuple(float(p) for p in row)
	if len(parsed) != len(VALUES):
		raise ValueError(f'{label} must have {len(VALUES)} entries (values 0-10)')
	if any(not math.isfinite(p) for p in parsed):
		raise ValueError(f'{label} probabilities must be finite')
	if any(p < 0 for p in parsed):
		raise ValueError(f'{label} has a negative probability')
	if abs(sum(parsed) - 1.0) > SUM_TOLERANCE:
		raise ValueError(f'{label} sums to {sum(parsed):g}, not 1')
	return parsed


def validate_distribution(rows: object, dims: int) -> Distribution:
	"""Check P is a d x 11 table of probabilities whose rows each sum to 1."""
	if not isinstance(rows, (list, tuple)) or isinstance(rows, (str, bytes)):
		raise ValueError('P must be a list of rows')
	table = tuple(validate_row(row, label=f'P[{i}]') for i, row in enumerate(rows))
	if len(table) != dims:
		raise ValueError(f'P must have one row per dimension: expected {dims}, got {len(table)}')
	return table


def _rows_from_json(raw: object) -> object:
	if isinstance(raw, dict):
		if 'P' not in raw:
			raise ValueError('distribution JSON object must contain a "P" key')
		return raw['P']
	return raw


def load_distribution(path: str | Path, dims: int) -> Distribution:
	"""Read a full P table from JSON: bare list of rows or ``{"P": [...]}``."""
	raw = json.loads(Path(path).read_text())
	return validate_distribution(_rows_from_json(raw), dims)


def load_distribution_row(path: str | Path) -> DistributionRow:
	"""Read one dimension's PMF from JSON.

	Accepted shapes:
	- bare list of 11 probabilities
	- ``{"P": [11 probabilities]}``
	- ``{"P": [[11 probabilities]]}`` (a one-row table)
	"""
	raw = json.loads(Path(path).read_text())
	rows = _rows_from_json(raw)
	if (
		isinstance(rows, (list, tuple))
		and rows
		and isinstance(rows[0], (list, tuple))
		and not isinstance(rows[0], (str, bytes))
	):
		if len(rows) != 1:
			raise ValueError(
				f'{path} must define a single dimension row; got {len(rows)} rows. '
				'Use --dist-file for a full d x 11 table.'
			)
		return validate_row(rows[0], label=str(path))
	return validate_row(rows, label=str(path))


def parse_dim_map(text: str) -> dict[int, str]:
	"""Parse ``0:3,1:normal,2:7`` or newline-separated ``0: 3`` lines into a map.

	``#`` starts a comment. Commas separate entries only after comments are
	stripped, so prose in comments may contain commas safely.
	"""
	mapping: dict[int, str] = {}
	for raw_line in text.splitlines():
		line = raw_line.split('#', 1)[0].strip()
		if not line:
			continue
		for part in line.split(','):
			entry = part.strip()
			if not entry:
				continue
			if ':' not in entry:
				raise ValueError(
					f'dim map entry {entry!r} must look like dim:source (e.g. 0:3 or 1:normal)'
				)
			dim_text, source = entry.split(':', 1)
			try:
				dim = int(dim_text.strip())
			except ValueError as exc:
				raise ValueError(
					f'dim map key must be an integer dimension, got {dim_text!r}'
				) from exc
			source = source.strip()
			if not source:
				raise ValueError(f'dim map entry for dimension {dim} is missing a source')
			if dim in mapping:
				raise ValueError(f'dimension {dim} is listed more than once in the dim map')
			mapping[dim] = source
	if not mapping:
		raise ValueError('dim map is empty')
	return mapping


def dims_from_dim_map(dim_map: dict[int, str]) -> int:
	"""Infer d from a dim map: every index from 0 to d-1 must be listed."""
	if not dim_map:
		raise ValueError('dim map is empty')
	negative = sorted(dim for dim in dim_map if dim < 0)
	if negative:
		raise ValueError(f'dim map has negative dimension(s) {negative}')
	dims = max(dim_map) + 1
	missing = [i for i in range(dims) if i not in dim_map]
	if missing:
		raise ValueError(
			f'dim map implies d={dims} but is missing dimension(s) {missing}; '
			f'list every dimension from 0 to {dims - 1}'
		)
	return dims


def load_dim_map(path: str | Path) -> dict[int, str]:
	"""Read a per-dimension source map from JSON or a ``dim: source`` text file.

	JSON object keys are dimension indices (``{"0": "3", "1": "normal"}``).
	Text files use one ``dim: source`` entry per line (commas also allowed).
	The highest listed dimension plus one becomes d when this map drives a run.
	"""
	path = Path(path)
	text = path.read_text()
	stripped = text.strip()
	if not stripped:
		raise ValueError(f'{path} is empty')
	if stripped[0] == '{':
		raw = json.loads(stripped)
		if not isinstance(raw, dict):
			raise ValueError(f'{path} must be a JSON object mapping dim -> source')
		mapping: dict[int, str] = {}
		for key, value in raw.items():
			try:
				dim = int(key)
			except (TypeError, ValueError) as exc:
				raise ValueError(f'{path}: dim keys must be integers, got {key!r}') from exc
			if dim in mapping:
				raise ValueError(f'{path}: dimension {dim} is listed more than once')
			source = str(value).strip()
			if not source:
				raise ValueError(f'{path}: dimension {dim} is missing a source')
			mapping[dim] = source
		if not mapping:
			raise ValueError(f'{path} is an empty dim map')
		return mapping
	return parse_dim_map(text)


def compose_distribution(
	dim_map: dict[int, str],
	dims: int,
	monotonic: int,
	catalog: dict[str, DistributionRow],
	*,
	default: str = 'mixed',
) -> tuple[Distribution, str]:
	"""Build a full P table by choosing a source for each dimension.

	``dim_map`` keys are dimension indices. Values are ``uniform``, ``normal``,
	``mixed``, or a catalog key (usually a group code whose
	``distribution.json`` was discovered). Dimensions missing from the map fall
	back to ``default`` (a preset name).

	Returns ``(table, label)`` where ``label`` is a stable summary for logs.
	"""
	if dims < 1:
		raise ValueError('d must be at least 1.')
	unknown_dims = sorted(dim for dim in dim_map if dim < 0 or dim >= dims)
	if unknown_dims:
		raise ValueError(
			f'dim map names dimension(s) {unknown_dims} but d={dims} only allows 0..{dims - 1}'
		)
	if default not in PRESETS:
		raise ValueError(f'unknown default distribution {default!r}; choose one of {", ".join(PRESETS)}')

	fallback = build_distribution(default, dims, monotonic)
	rows: list[DistributionRow] = []
	labels: list[str] = []
	for i in range(dims):
		source = dim_map.get(i, default)
		if source in PRESETS:
			row = build_distribution(source, dims, monotonic)[i]
		elif source in catalog:
			row = catalog[source]
		else:
			known = ', '.join([*PRESETS, *sorted(catalog)]) or ', '.join(PRESETS)
			raise ValueError(
				f'unknown distribution source {source!r} for dimension {i}; '
				f'choose a preset or group code ({known})'
			)
		rows.append(row)
		labels.append(f'{i}:{source}')
	return tuple(rows), 'compose:' + ','.join(labels)


def sample_profile(rng: random.Random, distribution: Distribution) -> Profile:
	return tuple(rng.choices(VALUES, weights=row)[0] for row in distribution)


def decrement(desired: Profile, partner: Profile, monotonic: int) -> int:
	"""How far ``partner``'s true profile falls short of ``desired``.

	The first ``monotonic`` dimensions only count a shortfall: if the partner
	meets or beats what was wanted there is no penalty. Every other dimension
	is penalised by the squared distance in either direction.
	"""
	total = 0
	for i, (want, have) in enumerate(zip(desired, partner, strict=True)):
		gap = want - have
		if i < monotonic and gap <= 0:
			continue
		total += gap * gap
	return total


def happiness(
	bonus: float,
	own_profile: Profile,
	own_ideal: Profile,
	partner_profile: Profile,
	partner_ideal: Profile,
	monotonic: int,
) -> float:
	"""A married person's final score: the bonus, minus their own decrement,
	minus half of their partner's."""
	mine = decrement(own_ideal, partner_profile, monotonic)
	theirs = decrement(partner_ideal, own_profile, monotonic)
	return bonus - mine - theirs / 2
