"""Three-stage robustness tuning for Player6 over randomly generated markets.

    uv run python -m players.player_6.tune                      # all three stages
    uv run python -m players.player_6.tune --markets 10         # quick smoke run
    uv run python -m players.player_6.tune --stages 3           # re-run validation only

A *market* fixes everything the engine takes: N, t, d, m, the trait
distribution P, B, the interest-list limit, the roster of opponents, the seat
order and the engine seed. Markets are sampled from the ``space`` block of the
config, so one run covers many bonuses, trait counts, monotonic shares,
distributions and opponent mixes. Every Player6 variant replays the *same*
markets, so variant differences are paired rather than drowned in market noise.

Stage 1  honesty    the four LIE x ASSUME_LIES settings, default constants.
Stage 2  full       every honesty setting x every one-at-a-time constant value
                    on the same markets (stage 1 games are reused, not re-run).
Stage 3  validate   fresh markets (``validation_seed``): the current defaults
                    against the stage-2 winners (best honesty, best single
                    variant, and best honesty + every constant that beat it).

Games are appended to ``games.jsonl`` as each market finishes, so an
interrupted run resumes where it stopped when re-invoked with the same
``--out``. Changing the config or player.py makes the old directory stale; the
runner refuses to mix them and asks for a new ``--out``.

Outputs (``--out``, default ``results/player6_robust``):
    experiment.json        effective config, Player6 defaults and source hash
    markets_<seed>.jsonl   every sampled market, including its full P table
    games.jsonl            one row per (market, variant) game
    stage<k>_summary.csv   per-variant means and paired deltas vs the current defaults
    stage<k>_breakdown.csv the same deltas split by B ratio, d, m/d, P kind, arena, ...
    selection.json         stage-2 winners and the stage-3 verdict
    report.md              human-readable summary of all three stages

Objective: positive-happiness rate first, mean happiness as tie-break.
Confidence intervals are mean +/- 1.96 standard errors of the per-market
paired difference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from core.distributions import discover_distributions  # noqa: E402
from core.engine import Engine  # noqa: E402
from core.registry import discover  # noqa: E402
from models.profile import build_distribution, normal_row, uniform_row  # noqa: E402
from players.player_6.player import Player6  # noqa: E402

OURS = '6'
DEFAULT_CONFIG = Path(__file__).parent / 'configs' / 'robust_tuning.json'
PLAYER_SOURCE = Path(__file__).parent / 'player.py'
FAULT_INDEX = re.compile(r'^\S+?\[(\d+)\]')

METRICS = (
	'positive_rate',
	'mean_happiness',
	'married_rate',
	'bad_rate',
	'mean_married_round',
	'opponent_mean_happiness',
)
FEATURES = (
	'people',
	'rounds',
	'dims',
	'monotonic',
	'mono_share',
	'dist_kind',
	'bonus',
	'bonus_ratio',
	'random_cost',
	'list_multiple',
	'arena',
	'our_seats',
	'our_share',
	'opponents',
)

# Ordered buckets for the breakdown tables. Categorical features use their value.
BUCKETS = {
	'bonus_ratio': [
		('<0.5', lambda v: v < 0.5),
		('0.5-1', lambda v: v < 1),
		('1-2', lambda v: v < 2),
		('>=2', lambda v: True),
	],
	'dims': [('1-3', lambda v: v <= 3), ('4-6', lambda v: v <= 6), ('7-10', lambda v: True)],
	'mono_share': [
		('none', lambda v: v == 0),
		('<half', lambda v: v < 0.5),
		('>=half', lambda v: v < 1),
		('all', lambda v: True),
	],
	'dist_kind': None,
	'arena': None,
	'people': [('10-20', lambda v: v <= 20), ('22-32', lambda v: v <= 32), ('34+', lambda v: True)],
	'rounds': [('<25', lambda v: v < 25), ('25-75', lambda v: v <= 75), ('>75', lambda v: True)],
	'list_multiple': [
		('no limit', lambda v: v == 0),
		('1-2', lambda v: v <= 2),
		('3-4', lambda v: v <= 4),
		('6-8', lambda v: True),
	],
	'our_share': [
		('<25%', lambda v: v < 0.25),
		('25-50%', lambda v: v < 0.5),
		('50-99%', lambda v: v < 1),
		('solo', lambda v: True),
	],
}


# ----------------------------------------------------------------- markets


def _loguniform(rng: random.Random, lo: float, hi: float) -> float:
	return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _weighted(rng: random.Random, weights: dict) -> str:
	keys = list(weights)
	return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _normalise(weights: list[float]) -> tuple[float, ...]:
	total = sum(weights)
	return tuple(w / total for w in weights)


def _random_row(rng: random.Random) -> tuple[str, tuple[float, ...]]:
	shape = rng.choice(('dirichlet', 'normal', 'skewed', 'bimodal'))
	if shape == 'dirichlet':
		alpha = rng.choice((0.3, 1.0, 3.0))
		return shape, _normalise([rng.gammavariate(alpha, 1.0) + 1e-9 for _ in range(11)])
	if shape == 'normal':
		return shape, normal_row(rng.uniform(0, 10), rng.uniform(0.7, 3.5))
	if shape == 'skewed':
		p = rng.uniform(0.55, 0.9)
		weights = [p**v for v in range(11)]
		if rng.random() < 0.5:
			weights.reverse()
		return shape, _normalise(weights)
	low = normal_row(rng.uniform(0, 4), rng.uniform(0.6, 1.8))
	high = normal_row(rng.uniform(6, 10), rng.uniform(0.6, 1.8))
	mix = rng.uniform(0.2, 0.8)
	return shape, _normalise([mix * a + (1 - mix) * b for a, b in zip(low, high, strict=True)])


def _preset_row(rng: random.Random) -> tuple[str, tuple[float, ...]]:
	if rng.random() < 0.5:
		return 'uniform', uniform_row()
	return 'normal', normal_row()


def sample_distribution(
	rng: random.Random, kind: str, dims: int, monotonic: int, catalog: dict
) -> tuple[list[str], list[tuple[float, ...]]]:
	if kind == 'preset':
		name = rng.choice(('uniform', 'normal', 'mixed'))
		return [name] * dims, list(build_distribution(name, dims, monotonic))

	sources, rows = [], []
	group_codes = sorted(catalog, key=int)
	for _ in range(dims):
		dim_kind = rng.choice(('preset', 'groups', 'random')) if kind == 'hybrid' else kind
		if dim_kind == 'preset':
			source, row = _preset_row(rng)
		elif dim_kind == 'groups':
			code = rng.choice([*group_codes, 'uniform', 'normal'])
			if code in catalog:
				source, row = f'g{code}', tuple(catalog[code])
			else:
				source, row = code, uniform_row() if code == 'uniform' else normal_row()
		else:
			source, row = _random_row(rng)
		sources.append(source)
		rows.append(row)
	return sources, rows


def random_marriage_cost(table: list, monotonic: int) -> float:
	"""Expected own decrement + half the partner's for a uniformly random pair."""
	total = 0.0
	for dim, row in enumerate(table):
		for want, p_want in enumerate(row):
			for have, p_have in enumerate(row):
				gap = want - have
				if dim < monotonic and gap <= 0:
					continue
				total += p_want * p_have * gap * gap
	return 1.5 * total


def _pick_without_replacement(rng, codes, weights, k) -> list[str]:
	pool, chosen = list(codes), []
	for _ in range(k):
		code = rng.choices(pool, weights=[weights.get(c, weights['default']) for c in pool])[0]
		chosen.append(code)
		pool.remove(code)
	return chosen


def sample_roster(rng, kind, people, opponents, space) -> list[str]:
	weights = space['opponent_weights']
	if kind == 'solo' or not opponents:
		seats = [OURS] * people
	elif kind == 'duel':
		share = rng.uniform(*space['duel_share'])
		ours = min(people - 2, max(2, round(share * people)))
		opponent = _pick_without_replacement(rng, opponents, weights, 1)[0]
		seats = [OURS] * ours + [opponent] * (people - ours)
	else:
		share = rng.uniform(*space['field_share'])
		ours = min(people - 2, max(2, round(share * people)))
		rest = people - ours
		k = rng.randint(min(2, len(opponents)), min(len(opponents), rest))
		chosen = _pick_without_replacement(rng, opponents, weights, k)
		seats = [OURS] * ours + chosen + [rng.choice(chosen) for _ in range(rest - k)]
	rng.shuffle(seats)
	return seats


def sample_market(market_set: int, index: int, space: dict, catalog: dict, opponents) -> dict:
	"""A market is a pure function of (market_set, index) and the config."""
	rng = random.Random(f'player6-market:{market_set}:{index}')
	lo, hi = space['people']
	people = rng.randrange(lo + lo % 2, hi + 1, 2)
	rounds = max(1, round(_loguniform(rng, *space['rounds'])))
	dims = rng.randint(*space['dims'])
	monotonic = rng.randint(0, dims)
	dist_kind = _weighted(rng, space['distribution_kinds'])
	sources, table = sample_distribution(rng, dist_kind, dims, monotonic, catalog)
	cost = random_marriage_cost(table, monotonic)
	bonus = round(_loguniform(rng, *space['bonus_ratio']) * max(cost, 1.0), 1)
	list_multiple = rng.choice(space['list_multiple'])
	arena = _weighted(rng, space['arena_kinds'])
	seats = sample_roster(rng, arena, people, opponents, space)
	return {
		'market_set': market_set,
		'market': index,
		'engine_seed': rng.randrange(2**31),
		'people': people,
		'rounds': rounds,
		'dims': dims,
		'monotonic': monotonic,
		'mono_share': monotonic / dims,
		'dist_kind': dist_kind,
		'dist_sources': sources,
		'distribution': [list(row) for row in table],
		'random_cost': round(cost, 3),
		'bonus': bonus,
		'bonus_ratio': round(bonus / max(cost, 1.0), 4),
		'list_multiple': list_multiple,
		'arena': arena,
		'seats': seats,
		'our_seats': seats.count(OURS),
		'our_share': seats.count(OURS) / people,
		'opponents': ' '.join(sorted(set(seats) - {OURS}, key=lambda c: (not c.isdigit(), c))),
	}


# ----------------------------------------------------------------- variants


def variant(label: str, overrides: dict, **extra) -> dict:
	return {
		'label': label,
		'key': json.dumps(overrides, sort_keys=True),
		'overrides': overrides,
		**extra,
	}


def baseline_variant() -> dict:
	return variant('current defaults', {'LIE': Player6.LIE, 'ASSUME_LIES': Player6.ASSUME_LIES})


def _honesty_overrides(h: dict) -> dict:
	return {'LIE': bool(h['LIE']), 'ASSUME_LIES': bool(h['ASSUME_LIES'])}


def honesty_variants(cfg: dict) -> list[dict]:
	out = [variant(h['label'], _honesty_overrides(h), honesty=h['label']) for h in cfg['honesty']]
	base = baseline_variant()
	if base['key'] not in {v['key'] for v in out}:
		out.insert(0, base)
	return out


def full_variants(cfg: dict) -> list[dict]:
	out = []
	for h in cfg['honesty']:
		base = _honesty_overrides(h)
		out.append(variant(f'{h["label"]} | defaults', base, honesty=h['label'], param=None))
		for param, values in cfg['parameters'].items():
			default = getattr(Player6, param)
			for value in values:
				if value == default:
					continue
				out.append(
					variant(
						f'{h["label"]} | {param}={value}',
						{**base, param: value},
						honesty=h['label'],
						param=param,
						value=value,
					)
				)
	return out


# ----------------------------------------------------------------- playing

_AVAILABLE: dict | None = None


def _available() -> dict:
	global _AVAILABLE
	if _AVAILABLE is None:
		_AVAILABLE, _ = discover()
	return _AVAILABLE


def _mean(values: list) -> float | None:
	return statistics.fmean(values) if values else None


def play(market: dict, overrides: dict, timeout: float) -> dict:
	available = _available()
	cls = type('Player6', (available[OURS],), dict(overrides))
	seats = market['seats']
	results = Engine(
		players=[cls if code == OURS else available[code] for code in seats],
		rounds=market['rounds'],
		bonus=market['bonus'],
		dims=market['dims'],
		monotonic=market['monotonic'],
		seed=market['engine_seed'],
		distribution=tuple(tuple(row) for row in market['distribution']),
		distribution_name=market['dist_kind'],
		timeout=timeout,
		keep_records=False,
		list_multiple=market['list_multiple'],
	).run()

	ours = [p for p in results['players'] if seats[p['index']] == OURS]
	others = [p['happiness'] for p in results['players'] if seats[p['index']] != OURS]
	married = [p for p in ours if p['married']]
	fault_seats = [int(m.group(1)) for f in results['faults'] if (m := FAULT_INDEX.match(f))]
	return {
		'positive_rate': sum(p['happiness'] > 0 for p in ours) / len(ours),
		'mean_happiness': statistics.fmean(p['happiness'] for p in ours),
		'married_rate': len(married) / len(ours),
		'bad_rate': sum(p['happiness'] <= 0 for p in married) / len(ours),
		'mean_married_round': _mean([p['married_on'] for p in married]),
		'opponent_mean_happiness': _mean(others),
		'rounds_played': results['rounds_played'],
		'our_faults': sum(seats[i] == OURS for i in fault_seats if i < len(seats)),
		'faults': len(results['faults']),
	}


def run_market(job: dict) -> list[dict]:
	"""Worker: play every requested variant on one market."""
	market = job['market']
	features = {k: market[k] for k in FEATURES}
	rows = []
	for v in job['variants']:
		row = {'market_set': market['market_set'], 'market': market['market'], **features}
		row |= {'variant': v['label'], 'variant_key': v['key']}
		try:
			row |= play(market, v['overrides'], job['timeout'])
			row['error'] = None
		except Exception as exc:  # an engine/config bug, not a player fault
			row['error'] = f'{type(exc).__name__}: {exc}'
		rows.append(row)
	return rows


# ----------------------------------------------------------------- storage


class Store:
	"""Append-only game log, indexed by (market_set, market, variant_key)."""

	def __init__(self, path: Path) -> None:
		self.path = path
		self.rows: dict[tuple, dict] = {}
		if path.exists():
			for line in path.read_text().splitlines():
				try:
					row = json.loads(line)
				except json.JSONDecodeError:
					continue  # a line cut off by an interrupted run
				self.rows[(row['market_set'], row['market'], row['variant_key'])] = row

	def done(self, market_set: int, market: int, key: str) -> bool:
		row = self.rows.get((market_set, market, key))
		return row is not None and row.get('error') is None

	def get(self, market_set: int, market: int, key: str) -> dict | None:
		row = self.rows.get((market_set, market, key))
		return row if row is not None and row.get('error') is None else None

	def add(self, rows: list[dict]) -> None:
		with self.path.open('a') as fh:
			for row in rows:
				fh.write(json.dumps(row) + '\n')
				self.rows[(row['market_set'], row['market'], row['variant_key'])] = row


def _fmt_time(seconds: float) -> str:
	seconds = int(seconds)
	return (
		f'{seconds // 3600}h{seconds % 3600 // 60:02d}m'
		if seconds >= 3600
		else f'{seconds // 60}m{seconds % 60:02d}s'
	)


def run_stage(title, markets, variants, store, workers, timeout) -> None:
	jobs = []
	for market in markets:
		todo = [
			v for v in variants if not store.done(market['market_set'], market['market'], v['key'])
		]
		if todo:
			jobs.append({'market': market, 'variants': todo, 'timeout': timeout})

	total_games = len(markets) * len(variants)
	pending = sum(len(j['variants']) for j in jobs)
	print(
		f'\n{title}: {len(markets)} markets x {len(variants)} variants = {total_games} games'
		f' ({total_games - pending} already done, {pending} to play, {workers} workers)',
		file=sys.stderr,
	)
	if not jobs:
		return

	start, games, errors = time.monotonic(), 0, 0
	ctx = mp.get_context('spawn')
	with ctx.Pool(processes=max(1, min(workers, len(jobs)))) as pool:
		for n, rows in enumerate(pool.imap_unordered(run_market, jobs), 1):
			store.add(rows)
			games += len(rows)
			errors += sum(r['error'] is not None for r in rows)
			elapsed = time.monotonic() - start
			eta = elapsed / games * (pending - games)
			print(
				f'\r  {n}/{len(jobs)} markets, {games}/{pending} games, '
				f'{_fmt_time(elapsed)} elapsed, ~{_fmt_time(eta)} left'
				+ (f', {errors} ERRORS' if errors else '')
				+ ' ' * 4,
				end='',
				file=sys.stderr,
				flush=True,
			)
	print(file=sys.stderr)
	if errors:
		print(
			f'  warning: {errors} games raised; see "error" in games.jsonl. Re-run to retry them.',
			file=sys.stderr,
		)


# ----------------------------------------------------------------- analysis


def _ci(diffs: list[float]) -> tuple[float | None, float | None]:
	if not diffs:
		return None, None
	mean = statistics.fmean(diffs)
	half = 1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
	return mean, half


def _pairs(store, market_set, n_markets, key, ref_key):
	for i in range(n_markets):
		row, ref = store.get(market_set, i, key), store.get(market_set, i, ref_key)
		if row is not None and ref is not None:
			yield row, ref


def summarise_pairs(pairs: list[tuple[dict, dict]]) -> dict:
	out = {'markets': len(pairs)}
	for metric in METRICS:
		out[metric] = _mean([r[metric] for r, _ in pairs if r[metric] is not None])
	for metric, short in (('positive_rate', 'pos'), ('mean_happiness', 'hap')):
		mean, half = _ci([r[metric] - ref[metric] for r, ref in pairs])
		out[f'd_{short}'], out[f'd_{short}_ci'] = mean, half
	out['our_faults'] = sum(r['our_faults'] for r, _ in pairs)
	return out


def summarise(store, market_set, n_markets, variants, ref_key) -> dict[str, dict]:
	return {
		v['key']: summarise_pairs(list(_pairs(store, market_set, n_markets, v['key'], ref_key)))
		for v in variants
	}


def _bucket(feature: str, value) -> str:
	spec = BUCKETS[feature]
	if spec is None:
		return str(value)
	return next(label for label, test in spec if test(value))


def breakdown(store, market_set, n_markets, variants, ref_key) -> list[dict]:
	rows = []
	for v in variants:
		pairs = list(_pairs(store, market_set, n_markets, v['key'], ref_key))
		for feature, spec in BUCKETS.items():
			groups: dict[str, list] = {}
			for pair in pairs:
				groups.setdefault(_bucket(feature, pair[0][feature]), []).append(pair)
			order = [label for label, _ in spec] if spec else sorted(groups)
			for label in order:
				if label in groups:
					rows.append(
						{
							'variant': v['label'],
							'feature': feature,
							'bucket': label,
							**summarise_pairs(groups[label]),
						}
					)
	return rows


def rank_key(stats: dict) -> tuple:
	pos = stats['positive_rate']
	return (-1.0, 0.0) if pos is None else (round(pos, 4), stats['mean_happiness'])


def _pct(x) -> str:
	return '-' if x is None else f'{100 * x:5.1f}%'


def _delta(mean, half, scale=1.0, unit='') -> str:
	if mean is None:
		return '-'
	return f'{scale * mean:+6.2f}{unit} ±{scale * half:.2f}'


def write_csv(path: Path, rows: list[dict]) -> None:
	if not rows:
		return
	with path.open('w', newline='') as fh:
		writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)


def stage_tables(out, stage, store, market_set, n_markets, variants, ref_key) -> dict[str, dict]:
	stats = summarise(store, market_set, n_markets, variants, ref_key)
	write_csv(
		out / f'stage{stage}_summary.csv',
		[{'variant': v['label'], **stats[v['key']]} for v in variants],
	)
	write_csv(
		out / f'stage{stage}_breakdown.csv',
		breakdown(store, market_set, n_markets, variants, ref_key),
	)
	return stats


def format_table(variants, stats, limit=None) -> list[str]:
	ordered = sorted(variants, key=lambda v: rank_key(stats[v['key']]), reverse=True)[:limit]
	width = max(len(v['label']) for v in ordered)
	lines = [
		f'{"variant":<{width}}  {"positive":>8}  {"Δpos (pts, 95% CI)":>19}  {"happiness":>9}  '
		f'{"Δhap (95% CI)":>15}  {"married":>7}  {"bad":>6}  markets'
	]
	for v in ordered:
		s = stats[v['key']]
		hap = '-' if s['mean_happiness'] is None else f'{s["mean_happiness"]:9.2f}'
		lines.append(
			f'{v["label"]:<{width}}  {_pct(s["positive_rate"]):>8}  '
			f'{_delta(s["d_pos"], s["d_pos_ci"], 100):>19}  {hap:>9}  '
			f'{_delta(s["d_hap"], s["d_hap_ci"]):>15}  {_pct(s["married_rate"]):>7}  '
			f'{_pct(s["bad_rate"]):>6}  {s["markets"]}'
		)
	return lines


def format_breakdown(rows: list[dict], variants: list[dict], ref_label: str) -> list[str]:
	"""Δpos per bucket, one column per non-reference variant."""
	labels = [v['label'] for v in variants if v['label'] != ref_label]
	index = {(r['variant'], r['feature'], r['bucket']): r for r in rows}
	lines = [f'Δ positive rate (pts) vs {ref_label}, by market feature:']
	lines += [f'  [{i}] {lab}' for i, lab in enumerate(labels, 1)]
	lines.append(
		f'  {"feature":<14}{"bucket":<10}{"markets":>8}'
		+ ''.join(f'{f"[{i}]":>16}' for i in range(1, len(labels) + 1))
	)
	seen = []
	for r in rows:
		if (r['feature'], r['bucket']) not in seen:
			seen.append((r['feature'], r['bucket']))
	for feature, bucket in seen:
		cells = [index.get((lab, feature, bucket)) for lab in labels]
		n = next((c['markets'] for c in cells if c), 0)
		text = ''.join(
			f'{"-" if not c or c["d_pos"] is None else f"{100 * c['d_pos']:+.1f}±{100 * c['d_pos_ci']:.1f}":>16}'
			for c in cells
		)
		lines.append(f'  {feature:<14}{bucket:<10}{n:>8}{text}')
	return lines


def select_candidates(cfg, full, stats, baseline) -> tuple[list[dict], dict]:
	bases = [v for v in full if v['param'] is None]
	best_h = max(bases, key=lambda v: rank_key(stats[v['key']]))
	chosen = {}
	for param in cfg['parameters']:
		options = [v for v in full if v['honesty'] == best_h['honesty'] and v['param'] == param]
		if not options:
			continue
		best = max(options, key=lambda v: rank_key(stats[v['key']]))
		if rank_key(stats[best['key']]) > rank_key(stats[best_h['key']]):
			chosen[param] = best['value']
	best_single = max(full, key=lambda v: rank_key(stats[v['key']]))
	combo_label = best_h['honesty'] + (
		' + ' + ', '.join(f'{p}={v}' for p, v in chosen.items()) if chosen else ''
	)

	candidates, seen = [], set()
	for v in (
		baseline,
		variant(f'best honesty: {best_h["honesty"]}', best_h['overrides']),
		variant(f'best single: {best_single["label"]}', best_single['overrides']),
		variant(f'combined: {combo_label}', {**best_h['overrides'], **chosen}),
	):
		if v['key'] not in seen:
			seen.add(v['key'])
			candidates.append(v)
	selection = {
		'best_honesty': best_h['honesty'],
		'params_beating_best_honesty': chosen,
		'best_single_variant': best_single['label'],
		'candidates': [{'label': c['label'], 'overrides': c['overrides']} for c in candidates],
	}
	return candidates, selection


def verdict(candidates, stats, baseline) -> tuple[str, dict | None]:
	challengers = [c for c in candidates if c['key'] != baseline['key']]
	clear = [
		c
		for c in challengers
		if stats[c['key']]['d_pos'] is not None
		and stats[c['key']]['d_pos'] - stats[c['key']]['d_pos_ci'] > 0
	]
	if clear:
		best = max(clear, key=lambda c: rank_key(stats[c['key']]))
		s = stats[best['key']]
		return (
			f'ADOPT "{best["label"]}": positive rate {100 * s["d_pos"]:+.2f} pts '
			f'(95% CI {100 * (s["d_pos"] - s["d_pos_ci"]):+.2f} to {100 * (s["d_pos"] + s["d_pos_ci"]):+.2f}), '
			f'mean happiness {s["d_hap"]:+.2f} on fresh markets.',
			best,
		)
	ahead = [c for c in challengers if (stats[c['key']]['d_pos'] or 0) > 0]
	if ahead:
		best = max(ahead, key=lambda c: rank_key(stats[c['key']]))
		s = stats[best['key']]
		return (
			f'INCONCLUSIVE: best challenger "{best["label"]}" is {100 * s["d_pos"]:+.2f} pts but its 95% CI '
			f'(±{100 * s["d_pos_ci"]:.2f}) includes zero. Keep the current defaults unless other evidence favours it.',
			None,
		)
	return 'KEEP the current defaults: no stage-2 winner beat them on fresh markets.', None


# ----------------------------------------------------------------- driver


def load_markets(out, market_set, n, space, catalog, opponents) -> list[dict]:
	markets = [sample_market(market_set, i, space, catalog, opponents) for i in range(n)]
	with (out / f'markets_{market_set}.jsonl').open('w') as fh:
		for m in markets:
			fh.write(json.dumps(m) + '\n')
	return markets


def describe_markets(markets: list[dict]) -> str:
	def spread(key):
		values = [m[key] for m in markets]
		return f'{min(values)}-{max(values)}'

	def counts(key):
		c: dict = {}
		for m in markets:
			c[m[key]] = c.get(m[key], 0) + 1
		return ', '.join(f'{k} {v}' for k, v in sorted(c.items()))

	return (
		f'  N {spread("people")}, t {spread("rounds")}, d {spread("dims")}, m {spread("monotonic")}, '
		f'B {min(m["bonus"] for m in markets):g}-{max(m["bonus"] for m in markets):g} '
		f'(ratio {min(m["bonus_ratio"] for m in markets):.2f}-{max(m["bonus_ratio"] for m in markets):.2f})\n'
		f'  P: {counts("dist_kind")}\n  arenas: {counts("arena")}\n  list multiple: {counts("list_multiple")}'
	)


def prepare_out(out: Path, cfg: dict) -> None:
	out.mkdir(parents=True, exist_ok=True)
	source = PLAYER_SOURCE.read_bytes()
	experiment = {
		'config': cfg,
		'player6_sha256': hashlib.sha256(source).hexdigest(),
		'player6_defaults': {
			name: getattr(Player6, name)
			for name in [*cfg['parameters'], 'LIE', 'ASSUME_LIES', 'LIE_DISCOUNT', 'END_MARGIN']
		},
	}
	path = out / 'experiment.json'
	if path.exists():
		old = json.loads(path.read_text())
		# Market i is the same whatever the count, so growing or shrinking
		# --markets reuses finished games; anything else would mix setups.
		ignore = {'markets'}
		changed = [
			k
			for k in experiment
			if k == 'config'
			and {a: b for a, b in old.get(k, {}).items() if a not in ignore}
			!= {a: b for a, b in experiment[k].items() if a not in ignore}
			or k != 'config'
			and old.get(k) != experiment[k]
		]
		if changed:
			raise SystemExit(
				f'{out} holds games from a different setup ({", ".join(changed)} changed). '
				f'Use a new --out so old and new games are not mixed.'
			)
	path.write_text(json.dumps(experiment, indent=2) + '\n')


def parse_args(argv=None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
	parser.add_argument('--out', type=Path, default=ROOT / 'results' / 'player6_robust')
	parser.add_argument('--markets', type=int, help='Override the number of markets per stage.')
	parser.add_argument('--stages', type=int, nargs='+', default=[1, 2, 3], choices=[1, 2, 3])
	parser.add_argument('--workers', type=int, default=mp.cpu_count())
	parser.add_argument(
		'--timeout', type=float, help='Per-call player budget in seconds (0 disables).'
	)
	return parser.parse_args(argv)


def main(argv=None) -> None:
	args = parse_args(argv)
	sys.stdout.reconfigure(line_buffering=True)  # keep tables and stderr progress in order
	cfg = json.loads(args.config.read_text())
	if args.markets:
		cfg['markets'] = args.markets
	if args.timeout is not None:
		cfg['timeout'] = args.timeout
	n, timeout, space = cfg['markets'], cfg['timeout'], cfg['space']
	out = args.out
	prepare_out(out, cfg)

	available, errors = discover()
	for err in errors:
		print(f"warning: player '{err.code}' failed to load - {err.message}", file=sys.stderr)
	if OURS not in available:
		raise SystemExit('player 6 failed to load')
	opponents = sorted(
		(c for c in available if c != OURS), key=lambda c: (not c.isdigit(), c.zfill(3))
	)
	catalog, _ = discover_distributions()

	store = Store(out / 'games.jsonl')
	baseline = baseline_variant()
	honesty, full = honesty_variants(cfg), full_variants(cfg)
	train = load_markets(out, cfg['market_seed'], n, space, catalog, opponents)
	report = [
		'# Player6 robustness tuning\n',
		f'Config: `{args.config}`. Opponents: {", ".join(opponents)}.\n',
	]
	report += ['Training markets:\n```', describe_markets(train), '```\n']
	print(
		f'training markets (seed {cfg["market_seed"]}):\n{describe_markets(train)}', file=sys.stderr
	)

	def show(lines: list[str]) -> None:
		print('\n'.join(lines))
		report.extend(['```', *lines, '```\n'])

	if 1 in args.stages:
		run_stage('stage 1 (honesty)', train, honesty, store, args.workers, timeout)
	if 1 in args.stages or 2 in args.stages:
		stats = stage_tables(out, 1, store, cfg['market_seed'], n, honesty, baseline['key'])
		if any(s['markets'] for s in stats.values()):
			print(f'\n=== Stage 1: honesty settings on {n} markets ===')
			report.append(f'## Stage 1: honesty settings, {n} markets\n')
			show(format_table(honesty, stats))
			rows = breakdown(store, cfg['market_seed'], n, honesty, baseline['key'])
			show(
				format_breakdown(
					rows, honesty, next(v['label'] for v in honesty if v['key'] == baseline['key'])
				)
			)

	selection: dict = {}
	candidates = None
	if 2 in args.stages:
		run_stage('stage 2 (honesty x constants)', train, full, store, args.workers, timeout)
	stats = (
		stage_tables(out, 2, store, cfg['market_seed'], n, full, baseline['key'])
		if (2 in args.stages or 3 in args.stages)
		else {}
	)
	if stats and all(s['markets'] for s in stats.values()):
		if 2 in args.stages:
			print(f'\n=== Stage 2: {len(full)} variants on the same {n} markets (top 20) ===')
			report.append(f'## Stage 2: {len(full)} variants, same {n} markets (top 20)\n')
			show(format_table(full, stats, limit=20))
		candidates, selection = select_candidates(cfg, full, stats, baseline)
		lines = [
			f'best honesty setting: {selection["best_honesty"]}',
			f'constants that beat it: {selection["params_beating_best_honesty"] or "none"}',
			f'best single variant: {selection["best_single_variant"]}',
			'stage-3 candidates: ' + '; '.join(c['label'] for c in candidates),
		]
		print('\n' + '\n'.join(lines))
		report += ['### Stage-2 selection\n', *[f'- {line}' for line in lines], '']
	elif 3 in args.stages:
		print('\nstage 3 needs complete stage-2 games; run stage 2 first.', file=sys.stderr)

	if 3 in args.stages and candidates:
		valid = load_markets(out, cfg['validation_seed'], n, space, catalog, opponents)
		print(
			f'validation markets (seed {cfg["validation_seed"]}):\n{describe_markets(valid)}',
			file=sys.stderr,
		)
		run_stage('stage 3 (validation)', valid, candidates, store, args.workers, timeout)
		vstats = stage_tables(out, 3, store, cfg['validation_seed'], n, candidates, baseline['key'])
		print(f'\n=== Stage 3: winners re-checked on {n} fresh markets ===')
		report += [
			f'## Stage 3: validation on {n} fresh markets\n',
			'Validation markets:\n```',
			describe_markets(valid),
			'```\n',
		]
		show(format_table(candidates, vstats))
		rows = breakdown(store, cfg['validation_seed'], n, candidates, baseline['key'])
		show(format_breakdown(rows, candidates, baseline['label']))
		text, adopted = verdict(candidates, vstats, baseline)
		print('\n' + text)
		report += ['### Verdict\n', text, '']
		selection['validation'] = {c['label']: vstats[c['key']] for c in candidates}
		selection['verdict'] = text
		selection['adopt'] = adopted['overrides'] if adopted else None

	if selection:
		(out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
	(out / 'report.md').write_text('\n'.join(report) + '\n')
	print(
		f'\nwrote {out}/ (report.md, selection.json, stage*_summary.csv, stage*_breakdown.csv, games.jsonl)'
	)


if __name__ == '__main__':
	main()
