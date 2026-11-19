"""Card audit: does a model handle every card? For each of the 26 kingdom cards, games against the GA base genome on
kingdoms with that card (both seats). Per card: how often the model buys it, plays it, every decision kind the card
asks (answered by the model), its win rate with the card in the kingdom, and errors (illegal answers; must be 0).

    python experiments/card_audit.py runs/c1/net_500.npy --name c1_500     # results/card_audit_<name>.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

# the decision kinds (rl.core.KINDS indices) each card makes its player answer; attacks: what the victim answers
OWN = {"Throne Room": [2], "Chapel": [3], "Moneylender": [4], "Mine": [5, 6], "Remodel": [7, 8], "Workshop": [9],
       "Artisan": [10, 11], "Cellar": [12], "Harbinger": [13], "Vassal": [14], "Library": [15], "Sentry": [16, 17],
       "Poacher": [18]}
VICTIM = {"Militia": [19], "Bureaucrat": [20], "Bandit": [21]}


def _job(job):
    from rl.core import GA, ID, NET, Seat, base_genome, outcome, play
    P = np.load(job["net"])
    out = []
    for card, k, seed, seat in job["games"]:
        g = base_genome(k)["ints"]
        me = Seat(NET, g, P, record=True)
        seats = [me, Seat(GA, g)] if seat == 0 else [Seat(GA, g), me]
        r = play(seats, k, seed, rng_seed=seed ^ 0xA0D1)
        rec = r["rec"]
        m = rec["SEAT"] == seat
        c = ID[card]
        out.append({"card": card, "points": (outcome(r, seat) + 1) / 2, "error": r["error"], "capped": r["capped"],
                    "bought": int(((rec["KIND"] == 1) & (rec["A"] == c) & m).sum()),
                    "played": int(((rec["KIND"] == 0) & (rec["A"] == c) & m).sum()),
                    "kinds": sorted({int(x) for x in rec["KIND"][m]})})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--name", required=True)
    ap.add_argument("--games", type=int, default=40, help="kingdoms per card (x 2 seats)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import rl.core  # noqa: F401
    from dominion_engine.cards import KINGDOM_CARDS
    from rl.core import KINDS
    rng = random.Random("card audit")
    games = []
    for card in KINGDOM_CARDS:
        for _ in range(a.games):
            k = sorted([card] + rng.sample([c for c in KINGDOM_CARDS if c != card], 9))
            s = rng.getrandbits(32)
            games += [(card, k, s, 0), (card, k, s, 1)]
    t = time.time()
    with mp.Pool(a.workers) as pool:
        res = [r for part in pool.map(_job, [{"net": a.net, "games": games[i:i + 40]}
                                             for i in range(0, len(games), 40)]) for r in part]
    cards, problems = {}, []
    for card in KINGDOM_CARDS:
        rs = [r for r in res if r["card"] == card]
        kinds = set().union(*[set(r["kinds"]) for r in rs])
        played = sum(r["played"] for r in rs)
        own = {KINDS[k]: k in kinds for k in OWN.get(card, [])}
        vic = {KINDS[k]: k in kinds for k in VICTIM.get(card, [])}
        row = {"games": len(rs), "win": float(np.mean([r["points"] for r in rs])),
               "bought_in": float(np.mean([r["bought"] > 0 for r in rs])),
               "copies_bought": float(np.mean([r["bought"] for r in rs])),
               "plays_per_game": float(np.mean([r["played"] for r in rs])),
               "asks_answered": own, "as_victim_answered": vic,
               "errors": int(sum(r["error"] for r in rs)), "capped": int(sum(r["capped"] for r in rs))}
        cards[card] = row
        if row["errors"]:
            problems.append(f"{card}: {row['errors']} illegal answers")
        for k, ok in own.items():
            if played and not ok and not (card == "Moneylender"):
                problems.append(f"{card}: played {played} times but never asked '{k}'")
    out = {"t": time.strftime("%Y-%m-%d %H:%M"), "net": a.net, "name": a.name, "secs": round(time.time() - t),
           "games": len(res), "cards": cards, "problems": problems,
           "note": "kingdom = the card + 9 random others; opponent the GA base genome; both seats"}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", f"card_audit_{a.name}.json"), "w"), indent=1)
    print(f"{'card':14s} {'win':>5s} {'buys':>5s} {'copies':>6s} {'plays':>6s}  asks")
    for card, r in cards.items():
        asks = " ".join(f"{k}:{'ok' if v else 'NEVER'}" for k, v in {**r['asks_answered'], **r['as_victim_answered']}.items())
        print(f"{card:14s} {r['win']:.3f} {r['bought_in']:.3f} {r['copies_bought']:6.2f} {r['plays_per_game']:6.2f}  {asks}")
    print("problems:", problems or "none", f"({out['secs']}s)")


if __name__ == "__main__":
    main()
