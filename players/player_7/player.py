import math

from models.player import Advertisement, Decision, Player as BasePlayer


class Player7(BasePlayer):
    """Learn continuously and marry when a match beats further searching."""

    def __init__(self, snapshot, ctx):
        super().__init__(snapshot, ctx)
        self.contamination = 0.10
        self.positive_weight = 10.0
        self.opportunity_cap = 12
        self.opportunity_rate = 0.70

        # One total pseudo-observation per dimension, initially uniform.
        self.counts = [[1 / 11] * 11 for _ in range(self.dims)]
        self.seen = {}
        self.sample_scores = []
        self.singles = self.people
        self.round = 0
        self.cached = {}
        self.population_costs = None
        self.reservation_cache = {}

    def _cost(self, i, want, have):
        gap = want - have
        return max(0, gap) ** 2 if i < self.monotonic else gap ** 2

    def _pmf(self, i):
        total = sum(self.counts[i])
        return [n / total for n in self.counts[i]]

    def advertise(self):
        # Reveal our most costly-to-miss regular desire and our own value in
        # that dimension. Fall back to monotonic traits when necessary.
        pool = list(range(self.monotonic, self.dims)) or list(range(self.dims))
        seek = max(pool, key=lambda i: sum(self._cost(i, self.ideal[i], v)
                                          for v in range(11)) / 11)
        own = seek if seek >= self.monotonic else max(
            pool, key=lambda i: self.profile[i])
        return Advertisement(own, self.profile[own], seek, self.ideal[seek])

    def _score(self, who):
        """Estimated happiness, allowing for potentially misleading reports."""
        if who not in self.cached:
            record = self.seen[who]
            mean = 0.0
            for i in range(self.dims):
                q = self._pmf(i)
                eps = self.contamination
                population_own = sum(p * self._cost(i, self.ideal[i], v)
                                     for v, p in enumerate(q))
                population_other = sum(p * self._cost(i, w, self.profile[i])
                                       for w, p in enumerate(q))
                reported_own = max(self._cost(i, self.ideal[i], p[i])
                                   for p in record['shown'])
                reported_other = max(self._cost(i, w[i], self.profile[i])
                                     for w in record['stated'])
                own = (1 - eps) * reported_own + eps * population_own
                other = (1 - eps) * reported_other + eps * population_other
                mean += own + other / 2
            self.cached[who] = self.bonus - mean
        return self.cached[who]

    def _utility(self, score):
        # Secondary preference for positive outcomes, in happiness units.
        return score + self.positive_weight if score > 0 else score

    def _opportunities(self, left):
        remaining = max(0, self.people - 1 - len(self.seen))
        capacity = min(max(0, self.singles - 1), remaining + 2)
        return min(self.opportunity_cap, max(0, (left - 2) // 3), capacity)

    def _future_utilities(self):
        if self.population_costs is None:
            self.population_costs = []
            for _ in range(48):
                cost = 0
                for i in range(self.dims):
                    v, w = self.rng.choices(range(11), weights=self.counts[i], k=2)
                    cost += self._cost(i, self.ideal[i], v)
                    cost += self._cost(i, w, self.profile[i]) / 2
                self.population_costs.append(cost)
        prior = [self._utility(self.bonus - c) for c in self.population_costs]
        actual = [self._utility(s) for s in self.sample_scores[-40:]]
        if not actual:
            return prior
        # Keep a weak prior alongside the observed date scores.
        return actual + prior[:max(2, 8 - len(actual))]

    def _reservation(self, left):
        key = (left, self.singles, len(self.seen), len(self.sample_scores))
        if key in self.reservation_cache:
            return self.reservation_cache[key]
        if left <= max(3, math.ceil(0.10 * self.rounds)):
            return 0.0
        values = self._future_utilities()
        value = 0.0
        for _ in range(self._opportunities(left)):
            value += self.opportunity_rate * sum(max(0, u - value) for u in values) / len(values)
        self.reservation_cache[key] = value
        return value

    def _accept(self, score, left):
        return score > 0 and self._utility(score) >= self._reservation(left)

    def _ad_score(self, ad):
        cost = 0.0
        for i in range(self.dims):
            q = self._pmf(i)
            mine = sum(p * self._cost(i, self.ideal[i], v) for v, p in enumerate(q))
            theirs = sum(p * self._cost(i, w, self.profile[i]) for w, p in enumerate(q))
            # Ads guide the shortlist; they do not prove compatibility.
            if i == ad.own_dim:
                mine = 0.5 * mine + 0.5 * self._cost(i, self.ideal[i], ad.own_value)
            if i == ad.seek_dim:
                theirs = 0.5 * theirs + 0.5 * self._cost(i, ad.seek_value, self.profile[i])
            cost += mine + theirs / 2
        return self.bonus - cost

    def choose_dates(self, candidates, turn):
        self.round, self.singles = turn.round, turn.singles
        known = [c for c in candidates if c.index in self.seen]
        fresh = [c for c in candidates if c.index not in self.seen]
        acceptable = [c for c in known if self._accept(self._score(c.index), turn.rounds_left)]
        acceptable.sort(key=lambda c: (-self._utility(self._score(c.index)), c.index))
        if acceptable:
            chosen = acceptable[:2]
        else:
            fresh.sort(key=lambda c: (-self._ad_score(c.ad), c.index))
            k = min(6, max(2, math.ceil(math.sqrt(len(fresh)))))
            chosen = fresh[:max(1, k - 1)]
            tail = fresh[max(1, k - 1):]
            if tail:
                chosen.append(self.rng.choice(tail))
            if not chosen:
                known.sort(key=lambda c: (-max(0, self._score(c.index)), c.index))
                chosen = known[:2]
        limit = len(chosen) if self.list_limit is None else self.list_limit
        return [c.index for c in chosen[:limit]]

    def decide(self, date, observed, stated_ideal):
        self.round = date.round
        who = date.partner
        fresh = who not in self.seen
        self.cached.clear()
        if fresh:
            self.population_costs = None
            # One effective observation per distinct person, split between
            # their projected traits and stated desires.
            for i, (v, w) in enumerate(zip(observed, stated_ideal)):
                self.counts[i][v] += 0.5
                self.counts[i][w] += 0.5
            self.seen[who] = {'shown': [], 'stated': []}
        record = self.seen[who]
        record['shown'].append(tuple(observed))
        record['stated'].append(tuple(stated_ideal))
        score = self._score(who)
        if fresh:
            self.sample_scores.append(score)
        willing = self._accept(score, date.rounds_left)
        return Decision(again=willing, propose=willing and date.date_number >= 2)
