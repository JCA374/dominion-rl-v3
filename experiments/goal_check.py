"""Goal check on fresh games (never used in training or the per-iteration evaluation).

1. az-v1's goal-check games exactly (same kingdoms, seeds and seats, so the scores compare directly
   with az-v1's goal table): 1,000 against the GA base genome, 200 against each bot.
2. With --vs_az: 1,000 head-to-head games (500 new kingdoms, both seats) against az-v1's best agent
   (r2 net 357 + IS-MCTS, 400 simulations, leaf ½ net + ½ GA rollout; GA genome for its non-buy play).

    python experiments/goal_check.py runs/r1/net_25.npy --name r1_iter25 [--vs_az]
    python experiments/goal_check.py az --name az_best --baseline   # az-v1's best agent itself on part 1
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

AZ_BEST = {"kind": "az", "path": "opponents/az_r2_357.npy", "sims": 400}
RL_V1 = {"kind": "net", "path": "opponents/rl_r2_2200.npy"}      # rl-v1's r2 at iteration 2,200: the target
BOTS = ["Big Money", "Smithy-BM", "Witch-BM"]


def job(j):
    from rl.jobs import eval_chunk
    return eval_chunk(j)


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", help="a net .npy, 'az' (az-v1's best agent) or 'ga'")
    ap.add_argument("--name", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--nh", type=int, default=256)
    ap.add_argument("--vs_az", action="store_true")
    ap.add_argument("--no_vs_rl", action="store_true", help="skip the 1,000 head-to-head games against rl-v1")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--bake", action="store_true")
    args = ap.parse_args()
    from rl.core import kingdom_for_bot, random_kingdom
    agent = AZ_BEST if args.agent == "az" else {"kind": "ga"} if args.agent == "ga" else \
        {"kind": "net", "path": args.agent}
    rng = random.Random("az-v1 goal check")           # = az-v1/experiments/goal_check.py
    games = []
    for _ in range(500):
        k, s = random_kingdom(rng), rng.getrandbits(32)
        games += [{"kingdom": k, "seed": s, "seat": seat, "opp": {"kind": "ga", "tag": "GA base"}} for seat in (0, 1)]
    for b in BOTS:
        for _ in range(100):
            k, s = kingdom_for_bot(rng, b), rng.getrandbits(32)
            games += [{"kingdom": k, "seed": s, "seat": seat, "opp": {"kind": "bot", "name": b, "tag": b}}
                      for seat in (0, 1)]
    if args.vs_az:
        r2 = random.Random("rl-v1 vs az")
        for _ in range(500):
            k, s = random_kingdom(r2), r2.getrandbits(32)
            games += [{"kingdom": k, "seed": s, "seat": seat, "opp": dict(AZ_BEST, tag="AZ best")} for seat in (0, 1)]
    if not args.no_vs_rl:
        r3 = random.Random("rl-v2 vs rl-v1")
        for _ in range(500):
            k, s = random_kingdom(r3), r3.getrandbits(32)
            games += [{"kingdom": k, "seed": s, "seat": seat, "opp": dict(RL_V1, tag="rl-v1")} for seat in (0, 1)]
    slow = [g for g in games if g["opp"]["kind"] == "az"] if args.agent != "az" else games
    fast = [g for g in games if g not in slow]
    jobs = [{"games": slow[i:i + 4], "agent": agent, "nh": args.nh} for i in range(0, len(slow), 4)] + \
           [{"games": fast[i:i + 50], "agent": agent, "nh": args.nh} for i in range(0, len(fast), 50)]
    t = time.time()
    with mp.Pool(max(1, os.cpu_count() - 1)) as pool:
        res = [r for part in pool.imap_unordered(job, jobs) for r in part]
    by = {}
    for r in res:
        by.setdefault(r["opp"], []).append(r["points"])
    m = float(np.mean(by["GA base"]))
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "label": args.label or args.name, "note": args.note,
           "policy": args.agent, "n": len(by["GA base"]), "score": round(m, 4), "ci": wilson(m, len(by["GA base"])),
           "big_money": round(float(np.mean(by["Big Money"])), 4), "smithy_bm": round(float(np.mean(by["Smithy-BM"])), 4),
           "witch_bm": round(float(np.mean(by["Witch-BM"])), 4), "secs": round(time.time() - t)}
    if "AZ best" in by:
        a = float(np.mean(by["AZ best"]))
        out.update(vs_az=round(a, 4), vs_az_ci=wilson(a, len(by["AZ best"])), vs_az_n=len(by["AZ best"]))
    if "rl-v1" in by:
        a = float(np.mean(by["rl-v1"]))
        out.update(vs_rl=round(a, 4), vs_rl_ci=wilson(a, len(by["rl-v1"])), vs_rl_n=len(by["rl-v1"]))
    os.makedirs("results", exist_ok=True)
    kind = "baseline" if args.baseline else "bake" if args.bake else "goal"
    path = f"results/{kind}_{args.name}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(path, json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
