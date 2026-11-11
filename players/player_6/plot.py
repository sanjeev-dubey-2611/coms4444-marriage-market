"""Charts for a players/player_6/tune.py results directory. Plays no games.

    uv run --with matplotlib python -m players.player_6.plot
    uv run --with matplotlib python -m players.player_6.plot results/player6_robust

matplotlib is pulled in only for this command (``--with``), so the shared
pyproject stays untouched. Writes PNG + SVG into ``<results>/plots/``:

    sensitivity_<honesty>.png            positive happiness vs each constant, one
                                         honesty setting, others at baseline (3x3)
    sensitivity_happiness_<honesty>.png  the same for mean happiness
    sensitivity_overlay.png              Δ positive rate vs current defaults, all four
                                         honesty settings per constant
    honesty.png                          stage 1: Δ positive rate and Δ happiness
    ranking.png                          stage 2: every variant, sorted, with 95% CI
    tradeoff.png                         stage 2: Δ positive rate vs Δ happiness
    where_honesty.png                    Δ positive rate by market feature (training)
    where_validation.png                 the same for the stage-3 candidates (fresh markets)
    validation.png                       training gain vs fresh-market gain per candidate
    coverage.png                         what the sampled markets covered

All Δ values are paired per market against the current defaults; error bars are
95% intervals of that paired difference. Absolute levels (the sensitivity
grids) are plain means over markets.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from players.player_6.tune import Store, _pairs, breakdown, summarise_pairs  # noqa: E402

# Reference categorical palette, fixed order (slot 1-4); markers are the
# secondary encoding so identity never rests on colour alone.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MARKERS = ['o', 's', '^', 'D']
LINE = SERIES[0]
BASELINE = '#eb6834'
NEUTRAL = '#8a8a85'
INK = '#2b2b2a'
DIVERGING = LinearSegmentedColormap.from_list('pos', ['#e34948', '#f0efec', '#2a78d6'])

plt.rcParams.update(
	{
		'axes.spines.top': False,
		'axes.spines.right': False,
		'axes.grid': True,
		'axes.axisbelow': True,
		'grid.color': '#e8e8e6',
		'grid.linewidth': 0.8,
		'axes.edgecolor': '#555553',
		'axes.labelcolor': INK,
		'xtick.color': INK,
		'ytick.color': INK,
		'text.color': INK,
		'axes.titlesize': 12,
		'axes.labelsize': 10,
		'figure.dpi': 110,
		'savefig.dpi': 150,
		'savefig.bbox': 'tight',
		'lines.linewidth': 1.6,
		'lines.markersize': 6,
	}
)


def key(overrides: dict) -> str:
	return json.dumps(overrides, sort_keys=True)


def slug(text: str) -> str:
	return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


class Results:
	def __init__(self, path: Path) -> None:
		self.path = path
		experiment = json.loads((path / 'experiment.json').read_text())
		self.cfg = experiment['config']
		self.defaults = experiment['player6_defaults']
		self.store = Store(path / 'games.jsonl')
		self.train_set = self.cfg['market_seed']
		self.valid_set = self.cfg['validation_seed']
		self.baseline = key(
			{'LIE': self.defaults['LIE'], 'ASSUME_LIES': self.defaults['ASSUME_LIES']}
		)
		self.honesty = [
			(h['label'], {'LIE': bool(h['LIE']), 'ASSUME_LIES': bool(h['ASSUME_LIES'])})
			for h in self.cfg['honesty']
		]
		selection = path / 'selection.json'
		self.selection = json.loads(selection.read_text()) if selection.exists() else {}

	def markets(self, market_set: int) -> int:
		indices = [m for s, m, _ in self.store.rows if s == market_set]
		return max(indices) + 1 if indices else 0

	def stats(self, market_set: int, variant_key: str, ref: str | None = None) -> dict:
		n = self.markets(market_set)
		return summarise_pairs(
			list(_pairs(self.store, market_set, n, variant_key, ref or self.baseline))
		)

	def variant_key(self, base: dict, param: str | None = None, value=None) -> str:
		if param is None or value == self.defaults[param]:
			return key(base)
		return key({**base, param: value})

	def market_rows(self, market_set: int) -> list[dict]:
		path = self.path / f'markets_{market_set}.jsonl'
		if not path.exists():
			return []
		return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(fig, out: Path, name: str) -> None:
	for ext in ('png', 'svg'):
		fig.savefig(out / f'{name}.{ext}')
	plt.close(fig)
	print(f'  {name}.png')


def _ci_err(stats: list[dict], short: str, scale: float) -> list[float]:
	return [scale * (s[f'd_{short}_ci'] or 0.0) for s in stats]


# ----------------------------------------------------------------- charts


def sensitivity(r: Results, out: Path, label: str, base: dict, metric: str) -> None:
	"""The 3x3 single-parameter grid: absolute level vs value, baseline dashed."""
	params = list(r.cfg['parameters'])
	n = r.markets(r.train_set)
	cols = 3
	rows = math.ceil(len(params) / cols)
	fig, axes = plt.subplots(rows, cols, figsize=(13, 3.0 * rows + 0.6), squeeze=False)
	is_pos = metric == 'positive_rate'
	ylabel = 'Positive happiness (%)' if is_pos else 'Mean happiness'
	for ax, param in zip(axes.flat, params, strict=False):
		values = sorted(r.cfg['parameters'][param])
		ys = []
		for v in values:
			s = r.stats(r.train_set, r.variant_key(base, param, v))
			level = s[metric]
			ys.append(None if level is None else (100 * level if is_pos else level))
		xs = [v for v, y in zip(values, ys, strict=True) if y is not None]
		ax.plot(xs, [y for y in ys if y is not None], color=LINE, marker='o')
		ax.axvline(r.defaults[param], color=BASELINE, linestyle='--', linewidth=1.4)
		ax.set_title(param)
		ax.set_ylabel(ylabel)
	for ax in list(axes.flat)[len(params) :]:
		ax.set_visible(False)
	what = 'positive happiness' if is_pos else 'mean happiness'
	fig.suptitle(
		f'Single-parameter sensitivity (other constants held at baseline)\n'
		f'{label}, {what}, {n} markets; dashed = current default',
		fontsize=12,
	)
	fig.tight_layout()
	prefix = 'sensitivity' if is_pos else 'sensitivity_happiness'
	save(fig, out, f'{prefix}_{slug(label)}')


def sensitivity_overlay(r: Results, out: Path) -> None:
	params = list(r.cfg['parameters'])
	cols = 3
	rows = math.ceil(len(params) / cols)
	fig, axes = plt.subplots(rows, cols, figsize=(13, 3.0 * rows + 0.9), squeeze=False)
	for ax, param in zip(axes.flat, params, strict=False):
		values = sorted(r.cfg['parameters'][param])
		for i, (label, base) in enumerate(r.honesty):
			stats = [r.stats(r.train_set, r.variant_key(base, param, v)) for v in values]
			ys = [100 * s['d_pos'] if s['d_pos'] is not None else math.nan for s in stats]
			ax.errorbar(
				values,
				ys,
				yerr=_ci_err(stats, 'pos', 100),
				color=SERIES[i % 4],
				marker=MARKERS[i % 4],
				capsize=2,
				elinewidth=0.9,
				label=label,
			)
		ax.axhline(0, color=NEUTRAL, linewidth=0.9)
		ax.axvline(r.defaults[param], color=NEUTRAL, linestyle='--', linewidth=1.1)
		ax.set_title(param)
		ax.set_ylabel('Δ positive (pts)')
	for ax in list(axes.flat)[len(params) :]:
		ax.set_visible(False)
	handles, labels = axes.flat[0].get_legend_handles_labels()
	fig.legend(
		handles,
		labels,
		loc='upper center',
		ncol=len(labels),
		frameon=False,
		bbox_to_anchor=(0.5, 0.965),
	)
	fig.suptitle(
		f'Δ positive rate vs current defaults, per constant and honesty setting '
		f'({r.markets(r.train_set)} markets, 95% CI); dashed = current default',
		fontsize=12,
	)
	fig.tight_layout(rect=(0, 0, 1, 0.93))
	save(fig, out, 'sensitivity_overlay')


def honesty_chart(r: Results, out: Path) -> None:
	labels = [label for label, _ in r.honesty]
	stats = [r.stats(r.train_set, key(base)) for _, base in r.honesty]
	fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
	for ax, short, scale, title in (
		(axes[0], 'pos', 100, 'Δ positive rate (pts)'),
		(axes[1], 'hap', 1, 'Δ mean happiness'),
	):
		ys = [scale * (s[f'd_{short}'] or 0.0) for s in stats]
		for i, (y, err) in enumerate(zip(ys, _ci_err(stats, short, scale), strict=True)):
			ax.errorbar(
				y, i, xerr=err, color=SERIES[i % 4], marker=MARKERS[i % 4], capsize=3, markersize=8
			)
		ax.axvline(0, color=NEUTRAL, linewidth=0.9)
		ax.set_yticks(range(len(labels)), labels)
		ax.invert_yaxis()
		ax.set_title(title)
		ax.grid(axis='y', visible=False)
	fig.suptitle(
		f'Honesty settings vs current defaults ({r.markets(r.train_set)} markets, 95% CI)',
		fontsize=12,
	)
	fig.tight_layout()
	save(fig, out, 'honesty')


def _full_variants(r: Results) -> list[tuple[str, int, str]]:
	"""(label, honesty index, key) for every stage-2 variant."""
	out = []
	for i, (label, base) in enumerate(r.honesty):
		out.append((f'{label} | defaults', i, key(base)))
		for param, values in r.cfg['parameters'].items():
			for v in values:
				if v != r.defaults[param]:
					out.append((f'{label} | {param}={v}', i, key({**base, param: v})))
	return out


def ranking(r: Results, out: Path) -> None:
	variants = _full_variants(r)
	rows = [(label, i, r.stats(r.train_set, k)) for label, i, k in variants]
	rows = [row for row in rows if row[2]['d_pos'] is not None]
	rows.sort(key=lambda row: (row[2]['d_pos'], row[2]['d_hap'] or 0.0))
	fig, ax = plt.subplots(figsize=(10, 0.2 * len(rows) + 1.5))
	for y, (_, i, s) in enumerate(rows):
		ax.errorbar(
			100 * s['d_pos'],
			y,
			xerr=100 * s['d_pos_ci'],
			color=SERIES[i % 4],
			marker=MARKERS[i % 4],
			markersize=4.5,
			elinewidth=0.9,
			capsize=1.5,
		)
	ax.axvline(0, color=NEUTRAL, linewidth=0.9)
	ax.set_yticks(range(len(rows)), [label for label, _, _ in rows], fontsize=7.5)
	ax.set_ylim(-1, len(rows))
	ax.grid(axis='y', visible=False)
	ax.set_xlabel('Δ positive rate vs current defaults (pts, 95% CI)')
	for i, (label, _) in enumerate(r.honesty):
		ax.plot([], [], color=SERIES[i % 4], marker=MARKERS[i % 4], linestyle='', label=label)
	ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(r.honesty), frameon=False)
	ax.set_title(f'All stage-2 variants, {r.markets(r.train_set)} markets (best at top)', pad=28)
	fig.tight_layout()
	save(fig, out, 'ranking')


def tradeoff(r: Results, out: Path) -> None:
	variants = _full_variants(r)
	fig, ax = plt.subplots(figsize=(8, 6))
	for i, (label, _) in enumerate(r.honesty):
		pts = [r.stats(r.train_set, k) for _, hi, k in variants if hi == i]
		pts = [s for s in pts if s['d_pos'] is not None]
		ax.scatter(
			[100 * s['d_pos'] for s in pts],
			[s['d_hap'] for s in pts],
			color=SERIES[i % 4],
			marker=MARKERS[i % 4],
			s=36,
			edgecolor='white',
			linewidth=0.8,
			label=label,
		)
	ax.axhline(0, color=NEUTRAL, linewidth=0.9)
	ax.axvline(0, color=NEUTRAL, linewidth=0.9)
	ax.set_xlabel('Δ positive rate (pts)')
	ax.set_ylabel('Δ mean happiness')
	ax.legend(frameon=False)
	ax.set_title(
		'Trade-off: every stage-2 variant vs current defaults\n(top right beats it on both)'
	)
	fig.tight_layout()
	save(fig, out, 'tradeoff')


def where(
	r: Results, out: Path, market_set: int, variants: list[tuple[str, str]], name: str, title: str
) -> None:
	"""Heatmap of Δ positive rate per market-feature bucket (rows) and variant (cols)."""
	variants = [(label, k) for label, k in variants if k != r.baseline]
	if not variants:
		return
	fake = [{'label': label, 'key': k} for label, k in variants]
	rows = breakdown(r.store, market_set, r.markets(market_set), fake, r.baseline)
	cells, counts, order = {}, {}, []
	for row in rows:
		rk = (row['feature'], row['bucket'])
		if rk not in order:
			order.append(rk)
		cells[(rk, row['variant'])] = row['d_pos']
		counts[rk] = row['markets']
	grid = [
		[
			100 * cells[(rk, label)] if cells.get((rk, label)) is not None else math.nan
			for label, _ in variants
		]
		for rk in order
	]
	finite = [abs(v) for line in grid for v in line if not math.isnan(v)]
	limit = max(finite + [1.0])

	fig, ax = plt.subplots(figsize=(2.0 * len(variants) + 4.5, 0.32 * len(order) + 2.2))
	image = ax.imshow(grid, cmap=DIVERGING, norm=TwoSlopeNorm(0, -limit, limit), aspect='auto')
	ax.grid(False)
	for y, line in enumerate(grid):
		for x, v in enumerate(line):
			if not math.isnan(v):
				ax.text(x, y, f'{v:+.1f}', ha='center', va='center', fontsize=8, color=INK)
	ax.set_yticks(
		range(len(order)), [f'{f}: {b}  (n={counts[(f, b)]})' for f, b in order], fontsize=8
	)
	ax.set_xticks(
		range(len(variants)),
		[label for label, _ in variants],
		rotation=20,
		ha='right',
		fontsize=8.5,
	)
	# Separate features with thin surface-coloured rules.
	for y in range(1, len(order)):
		if order[y][0] != order[y - 1][0]:
			ax.axhline(y - 0.5, color='white', linewidth=2)
	fig.colorbar(image, ax=ax, label='Δ positive rate vs current defaults (pts)', shrink=0.8)
	ax.set_title(title)
	fig.tight_layout()
	save(fig, out, name)


def _candidates(r: Results) -> list[tuple[str, str]]:
	return [(c['label'], key(c['overrides'])) for c in r.selection.get('candidates', [])]


def validation(r: Results, out: Path) -> None:
	cands = [(label, k) for label, k in _candidates(r) if k != r.baseline]
	if not cands or not r.markets(r.valid_set):
		return
	fig, ax = plt.subplots(figsize=(10, 1.0 * len(cands) + 1.6))
	for y, (_, k) in enumerate(cands):
		for offset, (market_set, i, name) in zip(
			(-0.12, 0.12),
			((r.train_set, 0, 'training markets'), (r.valid_set, 1, 'fresh markets')),
			strict=True,
		):
			s = r.stats(market_set, k)
			if s['d_pos'] is None:
				continue
			ax.errorbar(
				100 * s['d_pos'],
				y + offset,
				xerr=100 * s['d_pos_ci'],
				color=SERIES[i],
				marker=MARKERS[i],
				capsize=3,
				markersize=7,
				label=name if y == 0 else None,
			)
	ax.axvline(0, color=NEUTRAL, linewidth=0.9)
	ax.set_yticks(range(len(cands)), [label for label, _ in cands], fontsize=9)
	ax.invert_yaxis()
	ax.grid(axis='y', visible=False)
	ax.set_xlabel('Δ positive rate vs current defaults (pts, 95% CI)')
	ax.legend(frameon=False, loc='best')
	ax.set_title('Did the stage-2 winners hold up on fresh markets?')
	fig.tight_layout()
	save(fig, out, 'validation')


def coverage(r: Results, out: Path) -> None:
	markets = r.market_rows(r.train_set)
	if not markets:
		return
	numeric = [
		('bonus_ratio', 'B / random-marriage cost (log)', True),
		('bonus', 'B', False),
		('dims', 'd (traits)', False),
		('mono_share', 'm / d (monotonic share)', False),
		('people', 'N (people)', False),
		('rounds', 't (rounds)', False),
		('our_share', 'Player6 seat share', False),
	]
	categorical = [
		('dist_kind', 'Distribution kind'),
		('arena', 'Arena'),
		('list_multiple', 'List multiple'),
	]
	panels = len(numeric) + len(categorical)
	cols = 4
	rows = math.ceil(panels / cols)
	fig, axes = plt.subplots(rows, cols, figsize=(15, 2.8 * rows + 0.6), squeeze=False)
	flat = list(axes.flat)
	for ax, (field, title, log) in zip(flat, numeric, strict=False):
		values = [m[field] for m in markets]
		if log:
			lo, hi = min(values), max(values)
			bins = [lo * (hi / lo) ** (i / 20) for i in range(21)]
			ax.set_xscale('log')
			ax.axvline(1.0, color=NEUTRAL, linestyle='--', linewidth=1)
		elif field in ('dims', 'people'):
			lo, hi = min(values), max(values)
			step = 2 if field == 'people' else 1
			bins = [v - step / 2 for v in range(lo, hi + step + 1, step)]
		else:
			bins = 20
		ax.hist(values, bins=bins, color=LINE, edgecolor='white', linewidth=1.5)
		ax.set_title(title, fontsize=10.5)
	for ax, (field, title) in zip(flat[len(numeric) :], categorical, strict=False):
		counts: dict = {}
		for m in markets:
			counts[m[field]] = counts.get(m[field], 0) + 1
		labels = sorted(counts, key=lambda v: (isinstance(v, str), v))
		ax.bar(
			[str(v) for v in labels],
			[counts[v] for v in labels],
			color=LINE,
			edgecolor='white',
			linewidth=1.5,
		)
		ax.set_title(title, fontsize=10.5)
		ax.grid(axis='x', visible=False)
	for ax in flat[panels:]:
		ax.set_visible(False)
	fig.suptitle(f'What the {len(markets)} training markets covered (markets per bin)', fontsize=12)
	fig.tight_layout()
	save(fig, out, 'coverage')


def main(argv=None) -> None:
	parser = argparse.ArgumentParser(
		description='Plot a players/player_6/tune.py results directory.'
	)
	parser.add_argument(
		'results', type=Path, nargs='?', default=ROOT / 'results' / 'player6_robust'
	)
	args = parser.parse_args(argv)
	if not (args.results / 'experiment.json').exists():
		raise SystemExit(f'{args.results} is not a tune.py results directory (no experiment.json)')

	r = Results(args.results)
	out = args.results / 'plots'
	out.mkdir(exist_ok=True)
	print(f'writing {out}/')

	if r.markets(r.train_set):
		for label, base in r.honesty:
			sensitivity(r, out, label, base, 'positive_rate')
			sensitivity(r, out, label, base, 'mean_happiness')
		sensitivity_overlay(r, out)
		honesty_chart(r, out)
		ranking(r, out)
		tradeoff(r, out)
		where(
			r,
			out,
			r.train_set,
			[(label, key(base)) for label, base in r.honesty],
			'where_honesty',
			'Where each honesty setting wins or loses (training markets)',
		)
		coverage(r, out)
	if r.markets(r.valid_set):
		where(
			r,
			out,
			r.valid_set,
			_candidates(r),
			'where_validation',
			'Where each stage-3 candidate wins or loses (fresh markets)',
		)
		validation(r, out)


if __name__ == '__main__':
	main()
