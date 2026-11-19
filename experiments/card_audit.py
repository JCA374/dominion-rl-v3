"""Card audit: does a model handle every card? For each of the 26 kingdom cards, games against the GA base genome on
kingdoms with that card (both seats). Per card: how often the model buys it, plays it, every decision kind the card
asks (answered by the model), its win rate with the card in the kingdom, and errors (illegal answers; must be 0).
A second set of games forces its first buys to 2 copies of the card (the card school's forcing): can it play the card
and answer its questions, even where it would not buy it, and what does it answer?

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
SENTRY = ["trash", "discard", "keep"]          # the Sentry card decision answers codes, not cards (engine.h)


def _job(job):
    from rl.core import GA, ID, NAME, NET, PASS, Seat, base_genome, outcome, play
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
        row = {"card": card, "points": (outcome(r, seat) + 1) / 2, "error": r["error"], "capped": r["capped"],
               "bought": int(((rec["KIND"] == 1) & (rec["A"] == c) & m).sum()),
               "played": int(((rec["KIND"] == 0) & (rec["A"] == c) & m).sum()),
               "kinds": sorted({int(x) for x in rec["KIND"][m]})}
        force = (card, None) if seat == 0 else (None, card)
        me = Seat(NET, g, P, record=True)
        seats = [me, Seat(GA, g)] if seat == 0 else [Seat(GA, g), me]
        r = play(seats, k, seed, rng_seed=seed ^ 0xF0CE, force=force)
        rec = r["rec"]
        m = rec["SEAT"] == seat
        ans = {}
        for kd in OWN.get(card, []) + VICTIM.get(card, []):
            for a in rec["A"][m & (rec["KIND"] == kd)]:
                name = SENTRY[int(a)] if kd == 16 else "stop" if a == PASS else NAME[int(a)]
                key = f"{kd}:{name}"
                ans[key] = ans.get(key, 0) + 1
        row.update({"f_points": (outcome(r, seat) + 1) / 2, "f_error": r["error"],
                    "f_played": int(((rec["KIND"] == 0) & (rec["A"] == c) & m).sum()),
                    "f_kinds": sorted({int(x) for x in rec["KIND"][m]}), "f_answers": ans})
        out.append(row)
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
        fk = set().union(*[set(r["f_kinds"]) for r in rs])
        fa = {}
        for r in rs:
            for key, v in r["f_answers"].items():
                fa[key] = fa.get(key, 0) + v
        row["forced"] = {"win": float(np.mean([r["f_points"] for r in rs])),
                         "plays_per_game": float(np.mean([r["f_played"] for r in rs])),
                         "asks_answered": {KINDS[k]: k in fk for k in OWN.get(card, [])},
                         "errors": int(sum(r["f_error"] for r in rs)),
                         "answers": {KINDS[int(key.split(":")[0])]: {} for key in fa}}
        for key, v in sorted(fa.items(), key=lambda kv: -kv[1]):
            kd, ans_name = key.split(":", 1)
            row["forced"]["answers"][KINDS[int(kd)]][ans_name] = v
        cards[card] = row
        if row["errors"] or row["forced"]["errors"]:
            problems.append(f"{card}: {row['errors'] + row['forced']['errors']} illegal answers")
        for k, ok in row["forced"]["asks_answered"].items():
            if not ok and card != "Moneylender":
                problems.append(f"{card}: with 2 forced copies it never answered '{k}'")
    out = {"t": time.strftime("%Y-%m-%d %H:%M"), "net": a.net, "name": a.name, "secs": round(time.time() - t),
           "games": len(res), "cards": cards, "problems": problems,
           "note": "kingdom = the card + 9 random others; opponent the GA base genome; both seats"}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", f"card_audit_{a.name}.json"), "w"), indent=1)
    print(f"{'card':14s} {'win':>5s} {'buys':>5s} {'plays':>6s} {'forced plays':>12s}  forced answers")
    for card, r in cards.items():
        f = r["forced"]
        top = "; ".join(f"{k}: " + ", ".join(f"{a} {n}" for a, n in list(v.items())[:4]) for k, v in f["answers"].items())
        print(f"{card:14s} {r['win']:.3f} {r['bought_in']:.3f} {r['plays_per_game']:6.2f} {f['plays_per_game']:12.2f}  {top}")
    print("problems:", problems or "none", f"({out['secs']}s)")


if __name__ == "__main__":
    main()
