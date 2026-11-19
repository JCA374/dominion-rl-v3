"""Setup, end of game and scoring, as Steam Dominion plays them (base set rules, 2nd edition).

These are the facts both engines must agree on; RULES.md lists where each one comes from.
"""
from __future__ import annotations

from typing import Sequence

from .cards import CARDS, COPPER, CURSE, DUCHY, ESTATE, GARDENS, GOLD, NUM_CARDS, PROVINCE, SILVER

# Steam has no turn limit. This only stops games where nobody can end it (e.g. two agents that
# never buy); real games end far sooner (Steam campaign games took 17-65 turns).
ROUND_CAP = 100


def supply(kingdom: Sequence[int], num_players: int = 2) -> list[int]:
    """Pile sizes indexed by card id; -1 = not in this game.

    2 players: 8 Provinces/Duchies/Estates (and 8 of a Victory kingdom card), 3-4 players: 12.
    Copper is 60 minus the starting decks, Curse is 10 per opponent and always in the Supply.
    """
    assert 2 <= num_players <= 4
    victory = 8 if num_players == 2 else 12
    s = [-1] * NUM_CARDS
    s[COPPER] = 60 - 7 * num_players
    s[SILVER] = 40
    s[GOLD] = 30
    s[ESTATE] = s[DUCHY] = s[PROVINCE] = victory
    s[CURSE] = 10 * (num_players - 1)
    for k in kingdom:
        s[k] = victory if CARDS[k].is_victory else 10
    return s


def empty_piles(supply: Sequence[int]) -> int:
    return sum(1 for n in supply if n == 0)


def game_over(supply: Sequence[int]) -> bool:
    """Checked at the end of every turn: Provinces gone, or any 3 Supply piles empty."""
    return supply[PROVINCE] == 0 or empty_piles(supply) >= 3


def score(cards: Sequence[int]) -> int:
    vp = sum(CARDS[c].vp for c in cards)
    gardens = sum(1 for c in cards if c == GARDENS)
    return vp + gardens * (len(cards) // 10)


def winners(scores: Sequence[int], turns: Sequence[int]) -> list[int]:
    """Most VP wins; on a tie the tied player with fewer turns wins; still tied = shared win."""
    best = max(scores)
    tied = [i for i, s in enumerate(scores) if s == best]
    fewest = min(turns[i] for i in tied)
    return [i for i in tied if turns[i] == fewest]
