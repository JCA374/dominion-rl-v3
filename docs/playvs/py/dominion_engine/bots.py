"""Reference bots. Each has a twin in c/bots.c that decides exactly the same way."""
from __future__ import annotations

from .cards import GOLD, PROVINCE, SILVER
from .game import BUY, Decision, Game
from .rng import Rng


class BigMoney:
    """Province at $8, Gold at $6, Silver at $3; never plays actions. Other choices: first option."""

    def decide(self, game: Game, d: Decision) -> int:
        if d.kind == BUY:
            for card in (PROVINCE, GOLD, SILVER):
                if card in d.options:
                    return card
            return -1
        return -1 if d.may_pass else d.options[0]


class RandomBot:
    """Uniform over the options (plus pass when allowed). Reaches every rule; used by the parity test."""

    def __init__(self, seed: int):
        self.rng = Rng(seed)

    def decide(self, game: Game, d: Decision) -> int:
        k = self.rng.below(len(d.options) + (1 if d.may_pass else 0))
        return -1 if k == len(d.options) else d.options[k]
