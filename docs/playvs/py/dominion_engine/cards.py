"""Cards of Dominion's base set, 2nd edition (the set Steam Dominion deals from).

Card ids are shared with the C engine (c/engine.h) and never change: 0-21 are the cards
the first engine knew, 22-32 were added for Steam parity. New sets get new ids after 32.
"""
from __future__ import annotations

from dataclasses import dataclass

# Type flags (bit set, same values as c/engine.h)
TREASURE, VICTORY, ACTION, ATTACK, REACTION, CURSE_TYPE = 1, 2, 4, 8, 16, 32


@dataclass(frozen=True)
class Card:
    id: int
    name: str
    cost: int
    types: int
    coins: int = 0     # +$ when played (treasure value for treasures)
    vp: int = 0        # fixed victory points (Gardens is counted in rules.score)
    cards: int = 0     # +Cards
    actions: int = 0   # +Actions
    buys: int = 0      # +Buys

    @property
    def is_action(self) -> bool:
        return bool(self.types & ACTION)

    @property
    def is_treasure(self) -> bool:
        return bool(self.types & TREASURE)

    @property
    def is_victory(self) -> bool:
        return bool(self.types & VICTORY)

    @property
    def is_attack(self) -> bool:
        return bool(self.types & ATTACK)


_TABLE = [
    Card(0, "Copper", 0, TREASURE, coins=1),
    Card(1, "Silver", 3, TREASURE, coins=2),
    Card(2, "Gold", 6, TREASURE, coins=3),
    Card(3, "Estate", 2, VICTORY, vp=1),
    Card(4, "Duchy", 5, VICTORY, vp=3),
    Card(5, "Province", 8, VICTORY, vp=6),
    Card(6, "Village", 3, ACTION, cards=1, actions=2),
    Card(7, "Smithy", 4, ACTION, cards=3),
    Card(8, "Market", 5, ACTION, cards=1, actions=1, buys=1, coins=1),
    Card(9, "Laboratory", 5, ACTION, cards=2, actions=1),
    Card(10, "Festival", 5, ACTION, actions=2, buys=1, coins=2),
    Card(11, "Chapel", 2, ACTION),
    Card(12, "Throne Room", 4, ACTION),
    Card(13, "Council Room", 5, ACTION, cards=4, buys=1),
    Card(14, "Moneylender", 4, ACTION),
    Card(15, "Gardens", 4, VICTORY),
    Card(16, "Mine", 5, ACTION),
    Card(17, "Merchant", 3, ACTION, cards=1, actions=1),
    Card(18, "Curse", 0, CURSE_TYPE, vp=-1),
    Card(19, "Militia", 4, ACTION | ATTACK, coins=2),
    Card(20, "Witch", 5, ACTION | ATTACK, cards=2),
    Card(21, "Moat", 2, ACTION | REACTION, cards=2),
    Card(22, "Cellar", 2, ACTION, actions=1),
    Card(23, "Harbinger", 3, ACTION, cards=1, actions=1),
    Card(24, "Vassal", 3, ACTION, coins=2),
    Card(25, "Workshop", 3, ACTION),
    Card(26, "Bureaucrat", 4, ACTION | ATTACK),
    Card(27, "Poacher", 4, ACTION, cards=1, actions=1, coins=1),
    Card(28, "Remodel", 4, ACTION),
    Card(29, "Library", 5, ACTION),
    Card(30, "Sentry", 5, ACTION, cards=1, actions=1),
    Card(31, "Artisan", 6, ACTION),
    Card(32, "Bandit", 5, ACTION | ATTACK),
]

NUM_CARDS = len(_TABLE)
CARDS: list[Card] = _TABLE                        # indexed by id
BY_NAME: dict[str, Card] = {c.name: c for c in _TABLE}
ID: dict[str, int] = {c.name: c.id for c in _TABLE}
NAME: dict[int, str] = {c.id: c.name for c in _TABLE}

(COPPER, SILVER, GOLD, ESTATE, DUCHY, PROVINCE, VILLAGE, SMITHY, MARKET, LABORATORY, FESTIVAL,
 CHAPEL, THRONE_ROOM, COUNCIL_ROOM, MONEYLENDER, GARDENS, MINE, MERCHANT, CURSE, MILITIA, WITCH,
 MOAT, CELLAR, HARBINGER, VASSAL, WORKSHOP, BUREAUCRAT, POACHER, REMODEL, LIBRARY, SENTRY,
 ARTISAN, BANDIT) = range(NUM_CARDS)

BASE_CARDS = ["Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse"]
# The 26 kingdom cards of the base set, 2nd edition: Steam's random kingdoms draw 10 of these.
KINGDOM_CARDS = sorted(c.name for c in _TABLE if c.name not in BASE_CARDS)
assert len(KINGDOM_CARDS) == 26
