import math
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

from core.sandbox import PlayerFault, call_player
from models.player import (
	Advertisement,
	Candidate,
	DateContext,
	DateOutcome,
	Decision,
	GameContext,
	Player,
	PlayerSnapshot,
	RoundContext,
)
from models.profile import (
	MAX_VALUE,
	MIN_VALUE,
	Distribution,
	Profile,
	build_distribution,
	decrement,
	sample_profile,
	validate_distribution,
)

# A proposal is only legal once two people have been on at least this many
# dates with each other, counting the one that just ended.
MIN_DATES_TO_PROPOSE = 2

# How a pair came to be on a date, in the order the simulator fills them.
REPEAT = 'repeat'
MUTUAL = 'mutual'
ONE_SIDED = 'one-sided'
RANDOM = 'random'
STAGES = (REPEAT, MUTUAL, ONE_SIDED, RANDOM)


def _wrong_type(values: tuple[object, ...]) -> str | None:
	"""Name the first value that is not a plain ``int``, or None if all are.

	Worth its own message: ``np.int64`` does not subclass ``int``, so a
	vectorised strategy handing back a perfectly sensible profile would
	otherwise be told its values are out of range. ``bool`` is rejected too:
	it does subclass ``int``, but ``True`` as a profile value is a bug.
	"""
	for value in values:
		if isinstance(value, bool) or not isinstance(value, int):
			cls = type(value)
			if cls.__module__ == 'builtins':
				return cls.__qualname__
			return f'{cls.__module__}.{cls.__qualname__}'
	return None


def _profile(value: object, dims: int, what: str) -> Profile:
	values = tuple(value)
	if len(values) != dims:
		raise PlayerFault(f'{what} must have {dims} values, not {len(values)}')
	bad = _wrong_type(values)
	if bad is not None:
		raise PlayerFault(f'{what} values must be plain ints, not {bad}')
	if not all(MIN_VALUE <= v <= MAX_VALUE for v in values):
		raise PlayerFault(f'{what} values must lie in [{MIN_VALUE}, {MAX_VALUE}]')
	return values


def _flag(value: object, what: str) -> bool:
	if not isinstance(value, bool):
		raise PlayerFault(f'{what} must be a bool, not {type(value).__qualname__}')
	return value


def _key(a: int, b: int) -> tuple[int, int]:
	return (a, b) if a < b else (b, a)


class _Unbuilt(Player):
	"""Stand-in for a group whose constructor failed.

	Every call faults with the construction error, so the fault report names
	the culprit every round rather than once at startup, and the engine's
	ordinary fallbacks (honest behaviour, no interest, no decisions) apply.
	"""

	def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext, name: str, reason: str) -> None:
		super().__init__(snapshot, ctx)
		self.name = name
		self.reason = reason

	def advertise(self) -> Advertisement:
		raise PlayerFault(f'unusable: {self.reason}')

	def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
		raise PlayerFault(f'unusable: {self.reason}')

	def project(self, date: DateContext) -> Profile:
		raise PlayerFault(f'unusable: {self.reason}')

	def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
		raise PlayerFault(f'unusable: {self.reason}')

	def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
		raise PlayerFault(f'unusable: {self.reason}')


@dataclass
class DateRecord:
	"""One date, as the GUI and the debug log see it."""

	a: int
	b: int
	how: str
	number: int
	shown: dict[int, Profile] = field(default_factory=dict)
	stated: dict[int, Profile] = field(default_factory=dict)
	decisions: dict[int, Decision] = field(default_factory=dict)
	married: bool = False
	again: bool = False


@dataclass
class RoundRecord:
	"""Everything the GUI or a replay needs about one round of dating."""

	round: int
	singles: list[int] = field(default_factory=list)
	interest: dict[int, tuple[int, ...]] = field(default_factory=dict)
	dates: list[DateRecord] = field(default_factory=list)
	unmatched: list[int] = field(default_factory=list)
	marriages: list[tuple[int, int]] = field(default_factory=list)
	faults: list[str] = field(default_factory=list)


class Engine:
	def __init__(
		self,
		players: list[type[Player]],
		rounds: int = 100,
		bonus: float = 100.0,
		dims: int = 5,
		monotonic: int = 2,
		seed: int = 4444,
		distribution: str | Distribution = 'mixed',
		distribution_name: str | None = None,
		timeout: float = 1.0,
		keep_records: bool = True,
		list_multiple: float = 4.0,
	) -> None:
		people = len(players)
		if people < 2 or people % 2:
			raise ValueError(f'N must be even and at least 2, got {people}.')
		if dims < 1:
			raise ValueError('d must be at least 1.')
		if not 0 <= monotonic <= dims:
			raise ValueError(f'm must lie between 0 and d ({dims}), got {monotonic}.')
		if rounds < 1:
			raise ValueError('t must be at least 1.')
		if not math.isfinite(bonus):
			raise ValueError('B must be finite.')
		if not math.isfinite(timeout) or timeout < 0:
			raise ValueError('timeout must be finite and nonnegative.')
		if not math.isfinite(list_multiple) or list_multiple < 0:
			raise ValueError('list_multiple must be finite and nonnegative.')

		if isinstance(distribution, str):
			self.distribution_name = distribution_name or distribution
			self.distribution = build_distribution(distribution, dims, monotonic)
		else:
			self.distribution_name = distribution_name or 'custom'
			self.distribution = validate_distribution(distribution, dims)

		self.rng = random.Random(seed)
		self.list_multiple = float(list_multiple)
		scaled_limit = list_multiple * math.sqrt(people)
		self.list_limit = (
			(people - 1 if scaled_limit >= people - 1 else math.ceil(scaled_limit))
			if list_multiple
			else None
		)
		self.people = people
		self.rounds = rounds
		self.bonus = float(bonus)
		self.dims = dims
		self.monotonic = monotonic
		self.seed = seed
		self.timeout = timeout
		self.keep_records = keep_records

		# Engine truth. Players are handed only their own pair, at construction;
		# the GUI and the log read these, and nothing a player returns is ever
		# allowed to change them.
		self.profiles: list[Profile] = []
		self.ideals: list[Profile] = []
		for _ in range(people):
			self.profiles.append(sample_profile(self.rng, self.distribution))
			self.ideals.append(sample_profile(self.rng, self.distribution))

		self.round = 0
		self.spouse: list[int | None] = [None] * people
		self.married_on: list[int | None] = [None] * people
		self.marriages: list[dict] = []
		self.date_counts: dict[tuple[int, int], int] = {}
		# Directed: (x, y) means x marked y incompatible. Either direction keeps
		# the pair apart for good.
		self.blocked: set[tuple[int, int]] = set()
		# Pairs where both asked for another date last round. Paired first.
		self.repeat_pairs: list[tuple[int, int]] = []
		self.dates_had: list[int] = [0] * people
		self.proposals_made: list[int] = [0] * people
		self.proposals_received: list[int] = [0] * people

		self.records: list[RoundRecord] = []
		self.__faults: list[str] = []

		ctx = GameContext(
			people=people,
			dims=dims,
			monotonic=monotonic,
			rounds=rounds,
			bonus=self.bonus,
			list_limit=self.list_limit,
		)
		# Construction runs inside the sandbox for the same reason moves do: a
		# group whose __init__ raises or hangs must not take down every other
		# group's demo. It is replaced by a stand-in that faults on every call.
		self.players: list[Player] = []
		self.player_names: list[str] = []
		for i, cls in enumerate(players):
			snapshot = PlayerSnapshot(
				id=uuid.uuid5(uuid.NAMESPACE_OID, f'marriage-market:{seed}:{i}'),
				index=i,
				profile=self.profiles[i],
				ideal=self.ideals[i],
			)
			name = getattr(cls, '__name__', 'player')
			try:
				player = call_player(
					lambda cls=cls, snapshot=snapshot: cls(snapshot=snapshot, ctx=ctx),
					timeout=timeout,
				)
			except PlayerFault as exc:
				self.__faults.append(f'{name}[{i}]: could not be constructed - {exc}')
				player = _Unbuilt(snapshot, ctx, name, str(exc))
			self.players.append(player)
			self.player_names.append(getattr(player, 'name', name))

		self.ads: list[Advertisement] = [self.__advertise(i) for i in range(people)]

	# ------------------------------------------------------------------
	# Player interaction
	# ------------------------------------------------------------------

	def __ask(self, index: int, what: str, faults: list[str], fallback: Any, fn, *args) -> Any:
		"""Run one player call in the sandbox, falling back on any fault.

		``fn`` both calls the player and validates the answer, so a malformed
		reply is caught in the same guard as an exception or a timeout.
		"""
		try:
			return call_player(fn, *args, timeout=self.timeout)
		except PlayerFault as exc:
			faults.append(f'{self.player_names[index]}[{index}] {what}: {exc}')
			return fallback

	def __advertise(self, index: int) -> Advertisement:
		def ask() -> Advertisement:
			ad = self.players[index].advertise()
			fields = (ad.own_dim, ad.own_value, ad.seek_dim, ad.seek_value)
			bad = _wrong_type(fields)
			if bad is not None:
				raise PlayerFault(f'advertisement fields must be plain ints, not {bad}')
			own_dim, own_value, seek_dim, seek_value = fields
			if not (0 <= own_dim < self.dims and 0 <= seek_dim < self.dims):
				raise PlayerFault(f'advertised dimensions must lie in [0, {self.dims})')
			if not (MIN_VALUE <= own_value <= MAX_VALUE and MIN_VALUE <= seek_value <= MAX_VALUE):
				raise PlayerFault(f'advertised values must lie in [{MIN_VALUE}, {MAX_VALUE}]')
			return Advertisement(*fields)

		honest = Advertisement(0, self.profiles[index][0], 0, self.ideals[index][0])
		return self.__ask(index, 'advertise', self.__faults, honest, ask)

	def __choose(
		self, index: int, candidates: tuple[Candidate, ...], turn: RoundContext, record: RoundRecord
	) -> set[int]:
		allowed = {c.index for c in candidates}

		def ask() -> set[int]:
			answer = self.players[index].choose_dates(candidates, turn)
			chosen = (
				tuple(islice(answer, self.list_limit + 1))
				if self.list_limit is not None
				else tuple(answer)
			)
			bad = _wrong_type(chosen)
			if bad is not None:
				raise PlayerFault(f'chosen people must be plain int indices, not {bad}')
			if self.list_limit is not None and len(chosen) > self.list_limit:
				raise PlayerFault(f'interest list must contain at most {self.list_limit} entries')
			chosen_set = set(chosen)
			stray = sorted(chosen_set - allowed)
			if stray:
				raise PlayerFault(f"{stray} not among this round's candidates")
			return chosen_set

		return self.__ask(index, 'choose_dates', record.faults, set(), ask)

	def __project(self, index: int, date: DateContext, record: RoundRecord) -> Profile:
		def ask() -> Profile:
			return _profile(self.players[index].project(date), self.dims, 'projected profile')

		return self.__ask(index, 'project', record.faults, self.profiles[index], ask)

	def __describe(
		self, index: int, date: DateContext, observed: Profile, record: RoundRecord
	) -> Profile:
		def ask() -> Profile:
			stated = self.players[index].describe_ideal(date, observed)
			return _profile(stated, self.dims, 'described ideal')

		return self.__ask(index, 'describe_ideal', record.faults, self.ideals[index], ask)

	def __decide(
		self,
		index: int,
		date: DateContext,
		observed: Profile,
		stated: Profile,
		record: RoundRecord,
	) -> Decision:
		def ask() -> Decision:
			decision = self.players[index].decide(date, observed, stated)
			again = _flag(decision.again, 'again')
			propose = _flag(decision.propose, 'propose')
			incompatible = _flag(decision.incompatible, 'incompatible')
			if propose and date.date_number < MIN_DATES_TO_PROPOSE:
				raise PlayerFault(
					f'can only propose after {MIN_DATES_TO_PROPOSE} dates with the same person '
					f'(this was date {date.date_number})'
				)
			if incompatible and (again or propose):
				raise PlayerFault('cannot mark someone incompatible and also ask again or propose')
			return Decision(again=again, propose=propose, incompatible=incompatible)

		return self.__ask(index, 'decide', record.faults, Decision(), ask)

	def __tell(self, index: int, outcome: DateOutcome, record: RoundRecord) -> None:
		self.__ask(
			index, 'on_outcome', record.faults, None, self.players[index].on_outcome, outcome
		)

	# ------------------------------------------------------------------
	# Matching
	# ------------------------------------------------------------------

	def allowed(self, a: int, b: int) -> bool:
		"""False once either person has marked the other incompatible."""
		return (a, b) not in self.blocked and (b, a) not in self.blocked

	def dates_between(self, a: int, b: int) -> int:
		return self.date_counts.get(_key(a, b), 0)

	def singles(self) -> list[int]:
		return [i for i in range(self.people) if self.spouse[i] is None]

	def __match(self, singles: list[int], record: RoundRecord) -> list[tuple[int, int, str]]:
		taken: set[int] = set()
		pairs: list[tuple[int, int, str]] = []

		for a, b in self.repeat_pairs:
			if self.spouse[a] is None and self.spouse[b] is None and self.allowed(a, b):
				taken.update((a, b))
				pairs.append((a, b, REPEAT))

		pool = [i for i in singles if i not in taken]
		turn = RoundContext(
			round=self.round, rounds_left=self.rounds - self.round + 1, singles=len(singles)
		)
		interest: dict[int, set[int]] = {}
		# Advertisements are fixed and Candidate is immutable. Fresh candidates
		# can be shared across lists; only a person's own past dates need overrides.
		fresh = {j: Candidate(index=j, ad=self.ads[j], dates=0) for j in pool}
		pool_candidates = tuple(fresh.values())
		history: dict[int, dict[int, int]] = {}
		for (a, b), count in self.date_counts.items():
			if a in fresh and b in fresh:
				history.setdefault(a, {})[b] = count
				history.setdefault(b, {})[a] = count
		blocked_by: dict[int, set[int]] = {}
		for a, b in self.blocked:
			blocked_by.setdefault(a, set()).add(b)
		for i in pool:
			past = history.get(i)
			blocked = blocked_by.get(i)
			if not past and not blocked:
				candidates = tuple(c for c in pool_candidates if c.index != i)
			else:
				past = past or {}
				blocked = blocked or set()
				past_candidates = {j: Candidate(j, self.ads[j], count) for j, count in past.items()}
				candidates = tuple(
					past_candidates.get(c.index, c)
					for c in pool_candidates
					if c.index != i and c.index not in blocked
				)
			interest[i] = self.__choose(i, candidates, turn, record)
		record.interest = {i: tuple(sorted(chosen)) for i, chosen in interest.items()}

		# Visit players in random order, rather than building/sorting all edges.
		# Each stage is maximal: a pair of players still unmatched at its end
		# cannot have had an eligible edge when either endpoint was visited.
		order = list(pool)
		self.rng.shuffle(order)
		for how in (MUTUAL, ONE_SIDED):
			for a in order:
				if a in taken:
					continue
				for b in interest[a]:
					if b in taken or (how == MUTUAL and a not in interest[b]):
						continue
					if self.allowed(a, b):
						taken.update((a, b))
						pairs.append((*_key(a, b), how))
						break

		rest = [i for i in order if i not in taken]
		self.rng.shuffle(rest)
		for position, a in enumerate(rest):
			if a in taken:
				continue
			for k in range(position + 1, len(rest)):
				b = rest[k]
				if b not in taken and self.allowed(a, b):
					taken.update((a, b))
					pairs.append((*_key(a, b), RANDOM))
					break

		record.unmatched = [i for i in pool if i not in taken]
		return pairs

	# ------------------------------------------------------------------
	# Dating
	# ------------------------------------------------------------------

	def __date(
		self, a: int, b: int, how: str, record: RoundRecord, repeats: list[tuple[int, int]]
	) -> None:
		key = _key(a, b)
		number = self.date_counts.get(key, 0) + 1
		self.date_counts[key] = number
		self.dates_had[a] += 1
		self.dates_had[b] += 1

		partner = {a: b, b: a}
		ctx = {
			p: DateContext(
				round=self.round,
				rounds_left=self.rounds - self.round + 1,
				partner=partner[p],
				partner_ad=self.ads[partner[p]],
				date_number=number,
				repeat=how == REPEAT,
			)
			for p in (a, b)
		}
		date = DateRecord(a=a, b=b, how=how, number=number)

		# Each phase finishes for both people before the next begins, so
		# neither side can react to something the other has not yet revealed.
		for p in (a, b):
			date.shown[p] = self.__project(p, ctx[p], record)
		for p in (a, b):
			date.stated[p] = self.__describe(p, ctx[p], date.shown[partner[p]], record)
		for p in (a, b):
			q = partner[p]
			date.decisions[p] = self.__decide(p, ctx[p], date.shown[q], date.stated[q], record)

		for p in (a, b):
			decision = date.decisions[p]
			if decision.incompatible:
				self.blocked.add((p, partner[p]))
			if decision.propose:
				self.proposals_made[p] += 1
				self.proposals_received[partner[p]] += 1

		date.married = date.decisions[a].propose and date.decisions[b].propose
		date.again = (
			not date.married
			and date.decisions[a].again
			and date.decisions[b].again
			and self.allowed(a, b)
		)
		if date.married:
			self.__marry(a, b, record)
		elif date.again:
			repeats.append(key)
		record.dates.append(date)

		for p in (a, b):
			outcome = DateOutcome(
				round=self.round,
				partner=partner[p],
				date_number=number,
				partner_proposed=date.decisions[partner[p]].propose,
				married=date.married,
				again=date.again,
			)
			self.__tell(p, outcome, record)

	def __marry(self, a: int, b: int, record: RoundRecord) -> None:
		self.spouse[a] = b
		self.spouse[b] = a
		self.married_on[a] = self.round
		self.married_on[b] = self.round
		record.marriages.append((a, b))
		self.marriages.append(
			{
				'round': self.round,
				'a': a,
				'b': b,
				'happiness_a': self.happiness(a),
				'happiness_b': self.happiness(b),
			}
		)

	# ------------------------------------------------------------------
	# Simulation
	# ------------------------------------------------------------------

	@property
	def finished(self) -> bool:
		return self.round >= self.rounds or len(self.singles()) < 2

	def step(self) -> RoundRecord | None:
		if self.finished:
			return None

		self.round += 1
		singles = self.singles()
		record = RoundRecord(round=self.round, singles=singles)

		repeats: list[tuple[int, int]] = []
		for a, b, how in self.__match(singles, record):
			self.__date(a, b, how, record, repeats)
		self.repeat_pairs = repeats

		self.__faults.extend(record.faults)
		if self.keep_records:
			self.records.append(record)
		return record

	def run(self, on_round: Callable[[RoundRecord], None] | None = None) -> dict:
		"""Run every remaining round. ``on_round`` is how the debug log observes
		each RoundRecord without the engine opening a file."""
		while True:
			record = self.step()
			if record is None:
				break
			if on_round is not None:
				on_round(record)
		return self.results()

	# ------------------------------------------------------------------
	# Reporting
	# ------------------------------------------------------------------

	def own_decrement(self, index: int) -> int | None:
		partner = self.spouse[index]
		if partner is None:
			return None
		return decrement(self.ideals[index], self.profiles[partner], self.monotonic)

	def happiness(self, index: int) -> float:
		"""Final score for one person. Unmarried people score exactly 0."""
		partner = self.spouse[index]
		if partner is None:
			return 0.0
		mine = self.own_decrement(index)
		theirs = self.own_decrement(partner)
		return self.bonus - mine - theirs / 2

	def results(self) -> dict:
		per_player = []
		for i, name in enumerate(self.player_names):
			score = self.happiness(i)
			partner = self.spouse[i]
			per_player.append(
				{
					'index': i,
					'name': name,
					'married': partner is not None,
					'partner': partner,
					'married_on': self.married_on[i],
					'happiness': score,
					'positive': score > 0,
					'own_decrement': self.own_decrement(i),
					'partner_decrement': None if partner is None else self.own_decrement(partner),
					'dates': self.dates_had[i],
					'proposals_made': self.proposals_made[i],
					'proposals_received': self.proposals_received[i],
					'profile': list(self.profiles[i]),
					'ideal': list(self.ideals[i]),
					'ad': {
						'own_dim': self.ads[i].own_dim,
						'own_value': self.ads[i].own_value,
						'seek_dim': self.ads[i].seek_dim,
						'seek_value': self.ads[i].seek_value,
					},
				}
			)

		scores = [p['happiness'] for p in per_player]
		married = [p for p in per_player if p['married']]
		married_scores = [p['happiness'] for p in married]

		return {
			'parameters': {
				'people': self.people,
				'dims': self.dims,
				'monotonic': self.monotonic,
				'rounds': self.rounds,
				'bonus': self.bonus,
				'distribution': self.distribution_name,
				'seed': self.seed,
				'list_multiple': self.list_multiple,
				'list_limit': self.list_limit,
				'candidate_policy': 'all',
			},
			'rounds_played': self.round,
			'married_people': len(married),
			'married_rate': len(married) / self.people,
			'positive_rate': sum(1 for s in scores if s > 0) / self.people,
			'mean_happiness': sum(scores) / self.people,
			'mean_married_happiness': (
				sum(married_scores) / len(married_scores) if married_scores else None
			),
			'marriages': list(self.marriages),
			'players': per_player,
			'faults': list(self.__faults),
		}
