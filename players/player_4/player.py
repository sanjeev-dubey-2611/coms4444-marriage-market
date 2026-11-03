from models.player import (
    Candidate,
    DateContext,
    Decision,
    GameContext,
    PlayerSnapshot,
    RoundContext,
)
from models.player import Player as BasePlayer
from models.profile import Profile, decrement


class Player4(BasePlayer):
    """Group 4's player: basic happiness filter, plus three quick improvements.

    1. Unseen candidates are ordered by advertisement fit instead of shuffled
       at random (ads can be lies, so this only orders, never filters).
    2. Happiness estimates are averaged across dates, so one noisy date
       cannot decide a partner.
    3. A partner is only marked incompatible after a second bad date; a bad
       first date just passes, and the partner gets exactly one second look.
    """

    def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext) -> None:
        super().__init__(snapshot, ctx)
        self.estimates: dict[int, float] = {}
        self.counts: dict[int, int] = {}

    def _dim_penalty(self, desired: int, value: int, monotonic: bool) -> int:
        """Single-dimension decrement, per the handout's formula."""
        if monotonic and value >= desired:
            return 0
        return (desired - value) ** 2

    def _ad_fit(self, candidate: Candidate) -> int:
        """Lower is better: how well the candidate's ad fits both sides.

        Their advertised own value against my ideal on that dimension, plus
        their advertised sought value against my profile on that dimension.
        """
        ad = candidate.ad
        own = self._dim_penalty(
            self.ideal[ad.own_dim], ad.own_value, ad.own_dim < self.monotonic
        )
        seek = self._dim_penalty(
            ad.seek_value, self.profile[ad.seek_dim], ad.seek_dim < self.monotonic
        )
        return own + seek

    def choose_dates(self, candidates: tuple[Candidate, ...], turn: RoundContext) -> list[int]:
        # Past partners with a positive average estimate first, best first.
        liked = sorted(
            (c.index for c in candidates if self.estimates.get(c.index, 0.0) > 0),
            key=lambda i: -self.estimates[i],
        )
        # Partners with one bad date get exactly one second look, least-bad first.
        second_chance = sorted(
            (
                c.index
                for c in candidates
                if c.index in self.estimates
                and self.estimates[c.index] <= 0
                and c.dates < 2
            ),
            key=lambda i: -self.estimates[i],
        )
        # People never met, best ad fit first; shuffle first so ties break randomly.
        unseen = [c for c in candidates if c.index not in self.estimates]
        self.rng.shuffle(unseen)
        unseen = sorted(unseen, key=self._ad_fit)
        return (liked + second_chance + [c.index for c in unseen])[: self.list_limit]

    def decide(self, date: DateContext, observed: Profile, stated_ideal: Profile) -> Decision:
        mine = decrement(self.ideal, observed, self.monotonic)
        theirs = decrement(stated_ideal, self.profile, self.monotonic)
        estimate = self.bonus - mine - theirs / 2
        # Running average across dates: one noisy date cannot decide a partner.
        partner = date.partner
        n = self.counts.get(partner, 0) + 1
        self.counts[partner] = n
        avg = self.estimates.get(partner, 0.0) + (estimate - self.estimates.get(partner, 0.0)) / n
        self.estimates[partner] = avg

        if avg <= 0:
            # Only burn the bridge after a second date; a bad first date just passes.
            if date.date_number >= 2:
                return Decision(incompatible=True)
            return Decision()
        return Decision(again=True, propose=date.date_number >= 2)
