"""Discover per-group one-dimension distribution rows.

Convention:
    players/player_<k>/distribution.json  -> code '<k>'

Each file defines a single PMF over values 0-10 (11 probabilities summing to 1).
The simulator composes a full d x 11 table by assigning these rows (or presets)
to individual dimensions via ``--dim-dist`` / ``--dim-dist-file``.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models.profile import (
	Distribution,
	DistributionRow,
	compose_distribution,
	load_dim_map,
	load_distribution_row,
	parse_dim_map,
)

_GROUP_DIR = re.compile(r'^player_(\d+)$')
_PLAYERS_DIR = Path(__file__).resolve().parent.parent / 'players'


@dataclass
class DistLoadError:
	code: str
	message: str


def discover_distributions(
	players_dir: Path | None = None,
) -> tuple[dict[str, DistributionRow], list[DistLoadError]]:
	"""Return loadable group rows keyed by CLI code, plus whatever failed."""
	root = players_dir if players_dir is not None else _PLAYERS_DIR
	loaded: dict[str, DistributionRow] = {}
	errors: list[DistLoadError] = []

	if not root.is_dir():
		return loaded, errors

	for entry in sorted(root.iterdir()):
		match = _GROUP_DIR.match(entry.name)
		if not match or not entry.is_dir():
			continue
		code = match.group(1)
		path = entry / 'distribution.json'
		if not path.is_file():
			continue
		try:
			loaded[code] = load_distribution_row(path)
		except Exception as exc:
			errors.append(DistLoadError(code=code, message=f'{type(exc).__name__}: {exc}'))

	return loaded, errors


def warn_distribution_errors(errors: list[DistLoadError]) -> None:
	for err in errors:
		print(
			f"warning: distribution '{err.code}' failed to load - {err.message}",
			file=sys.stderr,
		)


def resolve_composed_distribution(
	dim_map: dict[int, str],
	*,
	dims: int,
	monotonic: int,
	default: str = 'mixed',
	catalog: dict[str, DistributionRow] | None = None,
) -> tuple[Distribution, str]:
	"""Compose P from a dim map, discovering group rows when no catalog is given."""
	if catalog is None:
		catalog, errors = discover_distributions()
		warn_distribution_errors(errors)
	return compose_distribution(
		dim_map,
		dims,
		monotonic,
		catalog,
		default=default,
	)


def distribution_from_settings(
	*,
	distribution: str,
	dims: int,
	monotonic: int,
	dist_file: str | None = None,
	dim_dist: str | None = None,
	dim_dist_file: str | None = None,
) -> tuple[str | Distribution, str | None]:
	"""Resolve CLI distribution options into an Engine argument plus optional name.

	Priority: ``dist_file`` > dim map > preset name. Returns
	``(distribution, distribution_name)``; ``distribution_name`` is only set for
	composed maps so logs show ``compose:0:3,1:normal,...``.
	"""
	if dist_file:
		from models.profile import load_distribution

		return load_distribution(dist_file, dims), 'custom'

	if dim_dist and dim_dist_file:
		raise ValueError('use only one of --dim-dist and --dim-dist-file')

	if dim_dist_file:
		dim_map = load_dim_map(dim_dist_file)
	elif dim_dist:
		dim_map = parse_dim_map(dim_dist)
	else:
		return distribution, None

	table, label = resolve_composed_distribution(
		dim_map,
		dims=dims,
		monotonic=monotonic,
		default=distribution,
	)
	return table, label


def distribution_from_sweep_value(
	raw: Any,
	*,
	dims: int,
	monotonic: int,
	catalog: dict[str, DistributionRow] | None = None,
) -> tuple[str | Distribution, str | None]:
	"""Interpret a sweep cell's ``distribution`` field.

	Accepts a preset name, an inline d x 11 table, a ``{"P": [...]}`` wrapper,
	a dim-map object ``{"0": "3", "1": "normal"}``, or a list of ``"0:3"``
	strings.
	"""
	from models.profile import validate_distribution

	if isinstance(raw, str):
		return raw, None

	if isinstance(raw, dict):
		if 'P' in raw:
			return validate_distribution(raw['P'], dims), 'custom'
		dim_map = {int(k): str(v).strip() for k, v in raw.items()}
		table, label = resolve_composed_distribution(
			dim_map,
			dims=dims,
			monotonic=monotonic,
			catalog=catalog,
		)
		return table, label

	if isinstance(raw, list) and raw and isinstance(raw[0], str) and ':' in raw[0]:
		dim_map = parse_dim_map('\n'.join(raw))
		table, label = resolve_composed_distribution(
			dim_map,
			dims=dims,
			monotonic=monotonic,
			catalog=catalog,
		)
		return table, label

	return validate_distribution(raw, dims), 'custom'
