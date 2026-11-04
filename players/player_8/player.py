"""Group 8's strategy: don't trust cheap talk, learn the pool."""

from models.player import (
    Advertisement,
    Candidate,
    DateContext,
    DateOutcome,
    Decision,
    GameContext,
    PlayerSnapshot,
    RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import Profile


class Player10(BasePlayer):
    def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
        super().__init__(snapshot, ctx)
        self.estimates: dict[int, float] = {}
        self.proposed_to_me: set[int] = set()
        self.obs: dict[int, list[tuple]] = {} 
        self.stated: dict[int, list[tuple]] = {} 
        self.sent: dict[int, tuple] = {}  
        self.pop: list[tuple] = [] 
        self.est_hist: list[float] = []

    def _g(self, dim: int, want: float, have: float) -> float:
        gap = want - have
        if dim < self.monotonic and gap <= 0:
            return 0.0
        return gap * gap

    def _pop_mean(self, dim: int) -> float:
        if len(self.pop) < 3:
            return 5.0
        return sum(p[dim] for p in self.pop) / len(self.pop)

    def _prior_gap(self, dim: int, have: float) -> float:
        """Expected partner decrement on `dim` if their ideal ~ population."""
        samples = [p[dim] for p in self.pop] if len(self.pop) >= 3 else list(range(11))
        return sum(self._g(dim, w, have) for w in samples) / len(samples)

    def _quantile(self, xs: list[float], q: float) -> float:
        s = sorted(xs)
        return s[min(len(s) - 1, max(0, int(q * (len(s) - 1))))]

    def _ad_cost(self, ad: Advertisement) -> float:
        mine = self._g(ad.own_dim, self.ideal[ad.own_dim], ad.own_value)
        theirs = self._g(ad.seek_dim, ad.seek_value, self.profile[ad.seek_dim])
        return mine + theirs / 2

    def _partner_profile(self, partner: int, ad: Advertisement) -> list[float]:
        obs = self.obs[partner]
        n = len(obs)
        est = []
        for i in range(self.dims):
            vals = sorted(o[i] for o in obs)
            v = float(vals[len(vals) // 2])
            if i < self.monotonic and ad.own_dim == i:
                v = min(v, ad.own_value)  # conservative on boastful claims
            lam = 0.3 / n  # shrink toward pool mean, less with more evidence
            est.append((1 - lam) * v + lam * self._pop_mean(i))
        return est

    def _their_cost(self, partner: int, ad: Advertisement) -> float:
        """Expected total decrement partner suffers from our TRUE profile."""
        stated = self.stated[partner]
        sent = self.sent.get(partner)
        w_stated = 0.5
        if sent is not None:
            d = sum(abs(stated[-1][i] - sent[i]) for i in range(self.dims)) / self.dims
            if d < 1.0:
                w_stated = 0.05
        total = 0.0
        for i in range(self.dims):
            have = self.profile[i]
            from_stated = sum(self._g(i, s[i], have) for s in stated) / len(stated)
            prior = self._prior_gap(i, have)
            w = w_stated
            if i == ad.seek_dim:  # the fixed ad is somewhat more credible
                from_stated = (from_stated + self._g(i, ad.seek_value, have)) / 2
                w = max(w, 0.6)
            total += w * from_stated + (1 - w) * prior
        return total

    def advertise(self) -> Advertisement:
        showable = range(self.monotonic) if self.monotonic else range(self.dims)
        own = max(showable, key=lambda i: self.profile[i])
        seek = max(
            range(self.dims),
            key=lambda i: (abs(self.ideal[i] - 5), i >= self.monotonic),
        )
        return Advertisement(
            own_dim=own, own_value=self.profile[own],
            seek_dim=seek, seek_value=self.ideal[seek],
        )

    def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
        chosen = [c.index for c in candidates if self.estimates.get(c.index, 0.0) > 0]
        fresh = sorted(
            (c for c in candidates if c.index not in self.estimates),
            key=lambda c: (self._ad_cost(c.ad), c.index),
        )
        frac_left = turn.rounds_left / max(1, self.rounds)
        k = max(1, int(len(fresh) * (0.3 + 0.5 * frac_left)))
        chosen += [c.index for c in fresh[:k]]
        return chosen[: self.list_limit]

    def project(self, date: DateContext) -> Profile:
        projected = list(self.profile)
        dim, value = date.partner_ad.seek_dim, date.partner_ad.seek_value
        if dim < self.monotonic:
            projected[dim] = min(10, max(projected[dim], min(value, projected[dim] + 2)))
        else:
            step = max(-2, min(2, value - projected[dim]))
            projected[dim] = projected[dim] + step
        out = tuple(projected)
        self.sent[date.partner] = out
        return out

    def describe_ideal(self, date: DateContext, observed: Profile) -> Profile:
        return observed  # flattery only raises odds of mutual proposal

    def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
        p = date.partner
        self.obs.setdefault(p, []).append(tuple(observed))
        self.stated.setdefault(p, []).append(tuple(stated_ideal))
        if len(self.obs[p]) == 1:
            self.pop.append(tuple(observed))

        prof = self._partner_profile(p, date.partner_ad)
        mine = sum(self._g(i, self.ideal[i], prof[i]) for i in range(self.dims))
        theirs = self._their_cost(p, date.partner_ad)
        estimate = self.bonus - mine - theirs / 2
        self.estimates[p] = estimate
        self.est_hist.append(estimate)

        frac_left = date.rounds_left / max(1, self.rounds)
        margin = (0.02 + 0.08 * frac_left) * self.bonus

        hopeless = estimate < -0.25 * self.bonus or (
			date.date_number >= 3 and estimate <= 0 and frac_left > 0.3
		)

        if hopeless:
            return Decision(incompatible=True)
    
        again = estimate > (-0.1 * self.bonus if frac_left > 0.3 else 0)

        if len(self.est_hist) >= 5:
            bar = self._quantile(self.est_hist, 0.75 * frac_left)
        else:
            bar = 0.5 * self.bonus * frac_left
        bar = max(bar, margin)
        if p in self.proposed_to_me:
            bar = margin  # their interest doesn't change our payoff; keep a safety margin

        propose = date.date_number >= 2 and estimate > bar
        return Decision(again=again or propose, propose=propose)

    def on_outcome(self, outcome: DateOutcome) -> None:
        if outcome.partner_proposed:
            self.proposed_to_me.add(outcome.partner)