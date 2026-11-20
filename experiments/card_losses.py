"""Card losses: which kingdom cards does a model lose against, per opponent? Games of the model against every
opponent it trained with, on random kingdoms (bots: kingdoms with their card), both seats, seeds never used in
training or evaluation. Per game: the kingdom, the result, and both players' decks at their last buy (az-v1's buy
trace, which records any seat kind). The analysis is experiments/card_losses_report.py.

    python experiments/card_losses.py runs/c1/net_2363.npy --name c1_2363     # results/card_losses/<name>/games.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

OPPONENTS = [{"kind": "ga", "tag": "GA base"},
             {"kind": "net", "path": "opponents/rl_r2_2200.npy", "tag": "champion"},
             {"kind": "net", "path": "opponents/rl_r2_200.npy", "tag": "rl-v1 200"},
             {"kind": "net", "path": "opponents/rl_v2_z1h_1100.npy", "tag": "rl-v2"},
             {"kind": "az", "path": "opponents/az_r2_357.npy", "sims": 0, "tag": "az-v1 net"},
             {"kind": "bot", "name": "Big Money", "tag": "Big Money"},
             {"kind": "bot", "name": "Smithy-BM", "tag": "Smithy-BM"},
             {"kind": "bot", "name": "Witch-BM", "tag": "Witch-BM"}]
SEED_BASE = 0xCA5D_0000      # card-loss games only (training and eval seeds come from other generators)


def _job(job):
    from rl.core import NAME, outcome, play
    from rl.jobs import _seat
    out = []
    for gm in job["games"]:
        k, seat = gm["kingdom"], gm["seat"]
        me = _seat({"kind": "net", "path": job["net"]}, k)
        other = _seat(gm["opp"], k)
        seats = [me, other] if seat == 0 else [other, me]
        r = play(seats, k, gm["seed"], rng_seed=gm["seed"] ^ 0x5EED, trace=True)
        T = r["trace"]
        n = (T.shape[1] - 7) // 2 if T is not None and len(T) else 0
        decks = [{}, {}]
        if n:
            nc = len(NAME)
            last = T[-1]
            for p in range(2):
                own = last[7 + p * nc: 7 + (p + 1) * nc]
                decks[p] = {NAME[c]: int(own[c]) for c in range(nc) if own[c] > 0}
        out.append({"opp": gm["opp"]["tag"], "kingdom": k, "seat": seat, "seed": gm["seed"],
                    "points": (outcome(r, seat) + 1) / 2, "margin": r["scores"][seat] - r["scores"][1 - seat],
                    "turns": max(r["turns"]), "capped": r["capped"], "error": r["error"],
                    "me": decks[seat], "them": decks[1 - seat]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--name", required=True)
    ap.add_argument("--kingdoms", type=int, default=300, help="per opponent (x 2 seats)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from rl.core import kingdom_for_bot, random_kingdom
    rng = random.Random(SEED_BASE)
    games = []
    for opp in OPPONENTS:
        for _ in range(a.kingdoms):
            k = kingdom_for_bot(rng, opp["name"]) if opp["kind"] == "bot" else random_kingdom(rng)
            s = SEED_BASE ^ rng.getrandbits(48)
            games += [{"kingdom": k, "seed": s, "seat": seat, "opp": opp} for seat in (0, 1)]
    random.Random(1).shuffle(games)
    jobs = [{"net": a.net, "games": games[i:i + 40]} for i in range(0, len(games), 40)]
    t0 = time.time()
    rows = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, part in enumerate(pool.imap_unordered(_job, jobs)):
            rows += part
            if i % 20 == 0:
                print(f"{len(rows)}/{len(games)} games, {time.time() - t0:.0f}s", flush=True)
    d = os.path.join(ROOT, "results", "card_losses", a.name)
    os.makedirs(d, exist_ok=True)
    with gzip.open(os.path.join(d, "games.jsonl.gz"), "wt") as f:
        f.write(json.dumps({"header": {"net": a.net, "kingdoms_per_opp": a.kingdoms, "seed_base": SEED_BASE,
                                       "opponents": [o["tag"] for o in OPPONENTS], "secs": time.time() - t0}}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pts = {}
    for r in rows:
        pts.setdefault(r["opp"], []).append(r["points"])
    for o, p in pts.items():
        print(f"{o:12s} {np.mean(p):.3f}  n={len(p)}")
    print("errors", sum(r["error"] for r in rows), "capped", sum(r["capped"] for r in rows))


if __name__ == "__main__":
    main()
