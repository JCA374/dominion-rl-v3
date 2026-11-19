"""Python reference engine: Dominion base set (2nd edition) with Steam's rules.

The game is a generator. It yields a `Decision` whenever a player has a choice and is resumed
with the answer, so any agent can drive it: a scripted bot, an evolved genome, a person in a
GUI, or a neural net stepping it like an RL environment. `play()` runs a game with agents.

The C engine (c/engine.c) implements the same rules with the same decisions in the same order
and the same RNG, so the two produce identical games (tests/test_parity.py checks this).
Keep them in step: a rules change goes into both files.

Every choice is asked as one Decision: pick one of `options` (card ids, or SENTRY_* codes),
or -1 when `may_pass`. Options are always sorted ascending and distinct. Choices that are
never worse to take are made by the engine: playing all treasures, revealing Moat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Protocol, Sequence

from . import rules
from .cards import (ARTISAN, BANDIT, BUREAUCRAT, CARDS, CELLAR, CHAPEL, COPPER, COUNCIL_ROOM,
                    CURSE, ESTATE, GOLD, HARBINGER, ID, LIBRARY, MERCHANT, MILITIA, MINE, MOAT,
                    MONEYLENDER, NAME, NUM_CARDS, POACHER, PROVINCE, REMODEL, SENTRY, SILVER,
                    THRONE_ROOM, VASSAL, WITCH, WORKSHOP)
from .rng import MASK, Rng

# Decision kinds (same numbers in c/engine.h)
PLAY_ACTION, BUY, THRONE, CHAPEL_TRASH, MONEYLENDER_TRASH, MINE_TRASH, MINE_GAIN = range(7)
REMODEL_TRASH, REMODEL_GAIN, WORKSHOP_GAIN, ARTISAN_GAIN, ARTISAN_TOPDECK = range(7, 12)
CELLAR_DISCARD, HARBINGER_TOPDECK, VASSAL_PLAY, LIBRARY_SKIP, SENTRY_CARD, SENTRY_TOP = range(12, 18)
POACHER_DISCARD, MILITIA_DISCARD, BUREAUCRAT_TOPDECK, BANDIT_TRASH = range(18, 22)
NUM_KINDS = 22
KIND_NAME = ["play_action", "buy", "throne", "chapel_trash", "moneylender_trash", "mine_trash",
             "mine_gain", "remodel_trash", "remodel_gain", "workshop_gain", "artisan_gain",
             "artisan_topdeck", "cellar_discard", "harbinger_topdeck", "vassal_play",
             "library_skip", "sentry_card", "sentry_top", "poacher_discard", "militia_discard",
             "bureaucrat_topdeck", "bandit_trash"]
# Options of SENTRY_CARD
SENTRY_TRASH, SENTRY_DISCARD, SENTRY_KEEP = 0, 1, 2


@dataclass(frozen=True)
class Decision:
    kind: int
    player: int                 # who decides (the victim, for attacks)
    options: tuple[int, ...]
    may_pass: bool
    subject: int = -1           # the card the choice is about, or a count (see each kind)

    def __str__(self) -> str:
        opts = [str(o) if self.kind == SENTRY_CARD else NAME[o] for o in self.options]
        subj = f" [{NAME.get(self.subject, self.subject)}]" if self.subject >= 0 else ""
        return f"P{self.player} {KIND_NAME[self.kind]}{subj}: {opts}{' or pass' if self.may_pass else ''}"


@dataclass
class PlayerState:
    deck: list[int] = field(default_factory=list)      # top of deck = end of list
    hand: list[int] = field(default_factory=list)
    discard: list[int] = field(default_factory=list)
    play: list[int] = field(default_factory=list)      # in play this turn
    turns: int = 0                                     # turns completed
    rng: Rng = field(default_factory=lambda: Rng(1))

    def all_cards(self) -> list[int]:
        return self.deck + self.hand + self.discard + self.play


@dataclass
class Result:
    scores: list[int]
    turns: list[int]
    winners: list[int]          # more than one = shared win
    capped: bool                # stopped by rules.ROUND_CAP (Steam has no cap; rare)
    trace: int                  # hash of every decision and answer, for parity checks
    decks: list[list[int]]


class Agent(Protocol):
    def decide(self, game: "Game", d: Decision) -> int: ...


def _distinct(cards: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(set(cards)))


class Game:
    """One game. `kingdom` is 10 card names or ids; seat 0 moves first."""

    def __init__(self, kingdom: Sequence[str | int], seed: int, num_players: int = 2):
        self.kingdom = [k if isinstance(k, int) else ID[k] for k in kingdom]
        self.supply = rules.supply(self.kingdom, num_players)
        master = Rng(seed)
        self.players = [PlayerState(rng=Rng(master.next())) for _ in range(num_players)]
        for p in self.players:
            p.deck = [COPPER] * 7 + [ESTATE] * 3
            p.rng.shuffle(p.deck)
            self._draw(p, 5)
        self.trash: list[int] = []
        self.current = 0
        self.actions = self.buys = self.coins = self.merchants = 0
        self.trace = 0xcbf29ce484222325
        self.over = False

    # -- view helpers for agents ------------------------------------------------

    @property
    def provinces_left(self) -> int:
        return self.supply[PROVINCE]

    def turn_number(self, player: int) -> int:
        """The player's current (or next) turn, counting from 1."""
        return self.players[player].turns + 1

    def empty_piles(self) -> int:
        return rules.empty_piles(self.supply)

    # -- card movement ------------------------------------------------------------

    def _reveal(self, p: PlayerState) -> int:
        """Take the top card of the deck (shuffling the discard in if needed); -1 if none."""
        if not p.deck:
            if not p.discard:
                return -1
            p.deck, p.discard = p.discard, []
            p.rng.shuffle(p.deck)
        return p.deck.pop()

    def _draw(self, p: PlayerState, n: int) -> None:
        for _ in range(n):
            c = self._reveal(p)
            if c < 0:
                return
            p.hand.append(c)

    def _gain(self, p: PlayerState, card: int, to: str = "discard") -> bool:
        if self.supply[card] <= 0:
            return False
        self.supply[card] -= 1
        {"discard": p.discard, "hand": p.hand, "deck": p.deck}[to].append(card)
        return True

    def _gainable(self, max_cost: int) -> tuple[int, ...]:
        return tuple(i for i in range(NUM_CARDS) if self.supply[i] > 0 and CARDS[i].cost <= max_cost)

    def _others(self, i: int) -> list[int]:
        n = len(self.players)
        return [(i + k) % n for k in range(1, n)]

    # -- decisions ----------------------------------------------------------------

    def _ask(self, kind: int, player: int, options: Sequence[int], may_pass: bool,
             subject: int = -1) -> Generator[Decision, int, int]:
        d = Decision(kind, player, tuple(options), may_pass, subject)
        ans = yield d
        if not ((ans in d.options) or (ans == -1 and may_pass)):
            raise ValueError(f"illegal answer {ans} to {d}")
        h = self.trace
        for x in (kind, player, subject, len(d.options), *d.options, ans):
            h = ((h ^ (x & 0xffffffff)) * 0x100000001b3) & MASK
        self.trace = h
        return ans

    # -- playing cards ------------------------------------------------------------

    def _play(self, i: int, card: int):
        """Resolve a card already put into play."""
        p, c = self.players[i], CARDS[card]
        self.actions += c.actions
        self.buys += c.buys
        self.coins += c.coins
        self._draw(p, c.cards)

        if card == MERCHANT:
            self.merchants += 1
        elif card == THRONE_ROOM:
            t = yield from self._ask(THRONE, i, _distinct([x for x in p.hand if CARDS[x].is_action]), True)
            if t >= 0:
                p.hand.remove(t)
                p.play.append(t)
                yield from self._play(i, t)
                yield from self._play(i, t)
        elif card == CHAPEL:
            n = 0
            while n < 4 and p.hand:
                t = yield from self._ask(CHAPEL_TRASH, i, _distinct(p.hand), True, n)
                if t < 0:
                    break
                p.hand.remove(t)
                self.trash.append(t)
                n += 1
        elif card == CELLAR:
            n = 0
            while p.hand:
                t = yield from self._ask(CELLAR_DISCARD, i, _distinct(p.hand), True, n)
                if t < 0:
                    break
                p.hand.remove(t)
                p.discard.append(t)
                n += 1
            self._draw(p, n)
        elif card == MONEYLENDER:
            if COPPER in p.hand:
                t = yield from self._ask(MONEYLENDER_TRASH, i, (COPPER,), True)
                if t == COPPER:
                    p.hand.remove(COPPER)
                    self.trash.append(COPPER)
                    self.coins += 3
        elif card == MINE:
            ts = _distinct([x for x in p.hand if CARDS[x].is_treasure])
            if ts:
                t = yield from self._ask(MINE_TRASH, i, ts, True)
                if t >= 0:
                    p.hand.remove(t)
                    self.trash.append(t)
                    opts = tuple(x for x in self._gainable(CARDS[t].cost + 3) if CARDS[x].is_treasure)
                    if opts:
                        g = yield from self._ask(MINE_GAIN, i, opts, False, t)
                        self._gain(p, g, "hand")
        elif card == REMODEL:
            if p.hand:
                t = yield from self._ask(REMODEL_TRASH, i, _distinct(p.hand), False)
                p.hand.remove(t)
                self.trash.append(t)
                opts = self._gainable(CARDS[t].cost + 2)
                if opts:
                    g = yield from self._ask(REMODEL_GAIN, i, opts, False, t)
                    self._gain(p, g)
        elif card == WORKSHOP:
            opts = self._gainable(4)
            if opts:
                g = yield from self._ask(WORKSHOP_GAIN, i, opts, False)
                self._gain(p, g)
        elif card == ARTISAN:
            opts = self._gainable(5)
            if opts:
                g = yield from self._ask(ARTISAN_GAIN, i, opts, False)
                self._gain(p, g, "hand")
            if p.hand:
                t = yield from self._ask(ARTISAN_TOPDECK, i, _distinct(p.hand), False)
                p.hand.remove(t)
                p.deck.append(t)
        elif card == HARBINGER:
            if p.discard:
                t = yield from self._ask(HARBINGER_TOPDECK, i, _distinct(p.discard), True)
                if t >= 0:
                    p.discard.remove(t)
                    p.deck.append(t)
        elif card == VASSAL:
            t = self._reveal(p)
            if t >= 0:
                p.discard.append(t)
                if CARDS[t].is_action:
                    a = yield from self._ask(VASSAL_PLAY, i, (t,), True, t)
                    if a == t:
                        p.discard.pop()
                        p.play.append(t)
                        yield from self._play(i, t)
        elif card == POACHER:
            k = rules.empty_piles(self.supply)
            for j in range(min(k, len(p.hand))):
                t = yield from self._ask(POACHER_DISCARD, i, _distinct(p.hand), False, k - j)
                p.hand.remove(t)
                p.discard.append(t)
        elif card == LIBRARY:
            aside = []
            while len(p.hand) < 7:
                t = self._reveal(p)
                if t < 0:
                    break
                if CARDS[t].is_action:
                    a = yield from self._ask(LIBRARY_SKIP, i, (t,), True, t)
                    if a == t:
                        aside.append(t)
                        continue
                p.hand.append(t)
            p.discard.extend(aside)
        elif card == SENTRY:
            seen = [t for t in (self._reveal(p), self._reveal(p)) if t >= 0]
            kept = []
            for t in seen:
                a = yield from self._ask(SENTRY_CARD, i, (SENTRY_TRASH, SENTRY_DISCARD, SENTRY_KEEP), False, t)
                if a == SENTRY_TRASH:
                    self.trash.append(t)
                elif a == SENTRY_DISCARD:
                    p.discard.append(t)
                else:
                    kept.append(t)
            if len(kept) == 2 and kept[0] != kept[1]:
                top = yield from self._ask(SENTRY_TOP, i, _distinct(kept), False)
                kept.remove(top)
                kept.append(top)
            p.deck.extend(kept)
        elif card == COUNCIL_ROOM:
            for j in self._others(i):
                self._draw(self.players[j], 1)
        elif card == BUREAUCRAT:
            self._gain(p, SILVER, "deck")
            for j in self._others(i):
                q = self.players[j]
                if MOAT in q.hand:
                    continue
                vs = _distinct([x for x in q.hand if CARDS[x].is_victory])
                if vs:
                    t = yield from self._ask(BUREAUCRAT_TOPDECK, j, vs, False)
                    q.hand.remove(t)
                    q.deck.append(t)
        elif card == MILITIA:
            for j in self._others(i):
                q = self.players[j]
                if MOAT in q.hand:
                    continue
                while len(q.hand) > 3:
                    t = yield from self._ask(MILITIA_DISCARD, j, _distinct(q.hand), False, len(q.hand) - 3)
                    q.hand.remove(t)
                    q.discard.append(t)
        elif card == WITCH:
            for j in self._others(i):
                if MOAT not in self.players[j].hand:
                    self._gain(self.players[j], CURSE)
        elif card == BANDIT:
            self._gain(p, GOLD)
            for j in self._others(i):
                q = self.players[j]
                if MOAT in q.hand:
                    continue
                seen = [t for t in (self._reveal(q), self._reveal(q)) if t >= 0]
                cand = _distinct([t for t in seen if CARDS[t].is_treasure and t != COPPER])
                if cand:
                    t = cand[0]
                    if len(cand) > 1:
                        t = yield from self._ask(BANDIT_TRASH, j, cand, False)
                    seen.remove(t)
                    self.trash.append(t)
                q.discard.extend(seen)

    # -- turns and the game ---------------------------------------------------------

    def _turn(self, i: int):
        p = self.players[i]
        self.current = i
        self.actions, self.buys, self.coins, self.merchants = 1, 1, 0, 0

        while self.actions > 0:
            opts = _distinct([x for x in p.hand if CARDS[x].is_action])
            if not opts:
                break
            a = yield from self._ask(PLAY_ACTION, i, opts, True)
            if a < 0:
                break
            self.actions -= 1
            p.hand.remove(a)
            p.play.append(a)
            yield from self._play(i, a)

        silver = False
        rest = []
        for c in p.hand:
            if CARDS[c].is_treasure:
                p.play.append(c)
                self.coins += CARDS[c].coins
                if c == SILVER and not silver:
                    silver = True
                    self.coins += self.merchants
            else:
                rest.append(c)
        p.hand = rest

        while self.buys > 0:
            opts = tuple(x for x in range(NUM_CARDS)
                         if self.supply[x] > 0 and CARDS[x].cost <= self.coins)
            if not opts:
                break
            b = yield from self._ask(BUY, i, opts, True)
            if b < 0:
                break
            self.supply[b] -= 1
            p.discard.append(b)
            self.coins -= CARDS[b].cost
            self.buys -= 1

        p.discard.extend(p.play)
        p.discard.extend(p.hand)
        p.play, p.hand = [], []
        self._draw(p, 5)
        p.turns += 1

    def run(self) -> Generator[Decision, int, Result]:
        """Play to the end, yielding each decision; returns the Result."""
        capped = False
        while not self.over:
            for i in range(len(self.players)):
                yield from self._turn(i)
                if rules.game_over(self.supply):
                    self.over = True
                    break
            else:
                if self.players[-1].turns >= rules.ROUND_CAP:
                    self.over = capped = True
        scores = [rules.score(p.all_cards()) for p in self.players]
        turns = [p.turns for p in self.players]
        return Result(scores, turns, rules.winners(scores, turns), capped, self.trace,
                      [sorted(p.all_cards()) for p in self.players])


def play(kingdom: Sequence[str | int], agents: Sequence[Agent], seed: int) -> Result:
    """Play one game; agents[i] sits in seat i (seat 0 moves first)."""
    game = Game(kingdom, seed, len(agents))
    gen = game.run()
    try:
        d = next(gen)
        while True:
            d = gen.send(agents[d.player].decide(game, d))
    except StopIteration as stop:
        return stop.value
