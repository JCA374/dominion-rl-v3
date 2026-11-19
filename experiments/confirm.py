"""Confirmation on games never used anywhere (new seeds): 2,000 head-to-head games against az-v1's best agent
(1,000 kingdoms, both seats) and 2,000 against the GA base genome. Run once, on the chosen model.

    python experiments/confirm.py models/rl_r2_200.npy --name rl_r2_200
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from experiments.goal_check import AZ_BEST, job, wilson  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--name", required=True)
    ap.add_argument("--kingdoms", type=int, default=1000)
    args = ap.parse_args()
    from rl.core import random_kingdom
    rng = random.Random("rl-v2 confirmation, used once")
    az, ga = [], []
    for _ in range(args.kingdoms):
        k, s = random_kingdom(rng), rng.getrandbits(32)
        az += [{"kingdom": k, "seed": s, "seat": seat, "opp": dict(AZ_BEST, tag="AZ best")} for seat in (0, 1)]
        k, s = random_kingdom(rng), rng.getrandbits(32)
        ga += [{"kingdom": k, "seed": s, "seat": seat, "opp": {"kind": "ga", "tag": "GA base"}} for seat in (0, 1)]
    agent = AZ_BEST if args.net == "az" else {"kind": "net", "path": args.net}
    if args.net == "az":                              # az-v1's best on the same GA games (no mirror match)
        az = []
    n = 4 if args.net == "az" else 50
    jobs = [{"games": az[i:i + 4], "agent": agent, "nh": 256} for i in range(0, len(az), 4)] + \
           [{"games": ga[i:i + n], "agent": agent, "nh": 256} for i in range(0, len(ga), n)]
    t = time.time()
    with mp.Pool(max(1, os.cpu_count() - 1)) as pool:
        res = [r for part in pool.imap_unordered(job, jobs) for r in part]
    out = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "net": args.net, "secs": round(time.time() - t)}
    for tag, key in (("AZ best", "vs_az"), ("GA base", "vs_ga")):
        p = [r["points"] for r in res if r["opp"] == tag]
        if not p:
            continue
        m = float(np.mean(p))
        out[key] = {"score": round(m, 4), "ci": wilson(m, len(p)), "n": len(p)}
    json.dump(out, open(f"results/confirm_{args.name}.json", "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
