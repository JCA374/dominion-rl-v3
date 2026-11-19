"""Does a player use +Actions, +Cards and +Buys? Per turn, from recorded decisions, for a net and the GA genome on the
same games (both against the GA base genome, kingdoms with Village, Festival, Market, Laboratory or Council Room).

    python experiments/turn_usage.py runs/z1/net_775.npy
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from rl.core import GA, KINDS, NAME, NET, PASS, Seat, base_genome, play, random_kingdom  # noqa: E402

PLUS_ACTIONS = {"Village": 2, "Festival": 2, "Market": 1, "Laboratory": 1, "Merchant": 1, "Cellar": 1,
                "Harbinger": 1, "Poacher": 1, "Sentry": 1}
PLUS_BUYS = {"Market": 1, "Festival": 1, "Council Room": 1}
PLUS_CARDS = {"Smithy": 3, "Council Room": 4, "Laboratory": 2, "Moat": 2, "Witch": 2, "Village": 1, "Market": 1,
              "Merchant": 1, "Harbinger": 1, "Poacher": 1, "Sentry": 1}


def usage(agent, games):
    t = {"turns": 0, "actions": 0, "two_actions": 0, "spare_actions": 0, "buy_turns_extra": 0, "used_extra_buy": 0,
         "drew": 0, "plays": {}}
    for k, seed, seat in games:
        g = base_genome(k)["ints"]
        me = Seat(NET, g, agent, record=True) if agent is not None else Seat(GA, g, record=True)
        seats = [me, Seat(GA, g)] if seat == 0 else [Seat(GA, g), me]
        rec = play(seats, k, seed, rng_seed=seed ^ 0xAB)["rec"]
        for turn in sorted(set(rec["TURN"].tolist())):
            m = (rec["TURN"] == turn) & (rec["SEAT"] == seat)
            if not m.any():
                continue
            played = [NAME[int(a)] for a in rec["A"][m & (rec["KIND"] == KINDS.index("play action"))] if a != PASS]
            buys = int((m & (rec["KIND"] == KINDS.index("buy")) & (rec["A"] != PASS)).sum())
            t["turns"] += 1
            t["actions"] += len(played)
            t["two_actions"] += len(played) >= 2
            t["spare_actions"] += 1 + sum(PLUS_ACTIONS.get(c, 0) for c in played) - len(played)
            extra = sum(PLUS_BUYS.get(c, 0) for c in played)
            if extra:
                t["buy_turns_extra"] += 1
                t["used_extra_buy"] += buys >= 2
            t["drew"] += sum(PLUS_CARDS.get(c, 0) for c in played)
            for c in played:
                t["plays"][c] = t["plays"].get(c, 0) + 1
    n = t["turns"]
    return {"turns": n, "actions per turn": t["actions"] / n, "turns with 2+ actions": t["two_actions"] / n,
            "cards drawn by actions per turn": t["drew"] / n,
            "turns with a +Buy card played": t["buy_turns_extra"] / n,
            "... of those, bought 2+ cards": t["used_extra_buy"] / max(1, t["buy_turns_extra"]),
            "most played": sorted(t["plays"].items(), key=lambda kv: -kv[1])[:8]}


def main():
    rng = random.Random("rl-v2 turn usage")
    want = {"Village", "Festival", "Market", "Laboratory", "Council Room"}
    games = []
    while len(games) < 400:
        k = random_kingdom(rng)
        if want & set(k):
            s = rng.getrandbits(32)
            games += [(k, s, 0), (k, s, 1)]
    for label, agent in [("GA genome", None)] + [(p, np.load(p)) for p in sys.argv[1:]]:
        u = usage(agent, games)
        print(f"== {label}")
        for key, v in u.items():
            print(f"  {key:34s} {v:.3f}" if isinstance(v, float) else f"  {key:34s} {v}")


if __name__ == "__main__":
    main()
