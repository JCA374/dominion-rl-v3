"""+Actions, +Cards, +Buys and +$ of every card that has them (Python engine; the engine's parity tests make the C
engine identical), and that a whole turn really lets a player use them: a second action after a Village, a second
buy after a Market, the drawn cards in hand."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))

import pytest  # noqa: E402

from dominion_engine.cards import ID  # noqa: E402
from dominion_engine.game import Game  # noqa: E402

KINGDOM = ["Village", "Smithy", "Market", "Festival", "Laboratory", "Council Room", "Moat", "Witch", "Merchant",
           "Cellar"]

# card: (+cards, +actions, +buys, +$)
VANILLA = {"Village": (1, 2, 0, 0), "Smithy": (3, 0, 0, 0), "Market": (1, 1, 1, 1), "Festival": (0, 2, 1, 2),
           "Laboratory": (2, 1, 0, 0), "Council Room": (4, 0, 1, 0), "Moat": (2, 0, 0, 0), "Witch": (2, 0, 0, 0),
           "Merchant": (1, 1, 0, 0), "Cellar": (0, 1, 0, 0)}


def position(hand, deck):
    g = Game(KINGDOM, seed=1)
    p, q = g.players
    p.hand, p.deck, p.discard = [ID[c] for c in hand], [ID[c] for c in reversed(deck)], []
    q.hand, q.deck, q.discard = [ID["Copper"]] * 5, [ID["Copper"]] * 5, []
    g.actions, g.buys, g.coins = 0, 1, 0
    return g


def play_card(g, card, answers=()):
    p = g.players[0]
    p.hand.remove(ID[card])
    p.play.append(ID[card])
    gen, answers = g._play(0, ID[card]), list(answers)
    try:
        gen.send(None)
        while True:
            gen.send(answers.pop(0) if answers else None)
    except StopIteration:
        pass


@pytest.mark.parametrize("card", sorted(VANILLA))
def test_vanilla_bonus(card):
    cards, actions, buys, coins = VANILLA[card]
    g = position([card, "Estate"], deck=["Gold"] * 6)
    answers = [-1] if card == "Cellar" else []          # Cellar: discard nothing (pass)
    play_card(g, card, answers)
    p = g.players[0]
    assert len(p.hand) == 1 + cards, f"{card}: hand {len(p.hand)}"
    assert g.actions == actions and g.buys == 1 + buys and g.coins == coins


def test_village_lets_you_play_two_more_actions_and_market_gives_a_second_buy():
    """Through the C engine's decision callback (what the network sees): hand Village, Smithy, Market + treasures."""
    from rl.core import GA, NAME, NET, Seat, base_genome, play
    from rl.net import Net
    # A network that always wants to play an action and to buy (any legal non-pass answer): random weights, but
    # sampling picks from all legal answers, so over many games every possibility shows up. Checked: per turn the
    # number of actions played never exceeds what the +Actions allow, and two buys happen only with +Buy cards.
    W = Net(64).flat()
    k = ["Village", "Market", "Festival", "Smithy", "Laboratory", "Council Room", "Moat", "Witch", "Merchant", "Cellar"]
    g = base_genome(k)["ints"]
    saw_two_actions = saw_two_buys = 0
    for seed in range(60):
        r = play([Seat(NET, g, W, sample=True, record=True), Seat(GA, g)], k, seed, rng_seed=seed + 1)
        rec = r["rec"]
        for t in set(rec["TURN"].tolist()):
            m = rec["TURN"] == t
            mine = (rec["SEAT"] == 0) & m
            played = [int(a) for a in rec["A"][mine & (rec["KIND"] == 0) & (rec["A"] != 33)]]
            buys = int((mine & (rec["KIND"] == 1) & (rec["A"] != 33)).sum())
            allowed_actions = 1 + sum(VANILLA[NAME[c]][1] for c in played)
            allowed_buys = 1 + sum(VANILLA[NAME[c]][2] for c in played)
            assert len(played) <= allowed_actions, (seed, t, [NAME[c] for c in played])
            assert buys <= allowed_buys, (seed, t, buys, [NAME[c] for c in played])
            saw_two_actions += len(played) >= 2
            saw_two_buys += buys >= 2
    assert saw_two_actions > 20 and saw_two_buys > 20, (saw_two_actions, saw_two_buys)
