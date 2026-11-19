"""Does the turn planner make a net stronger? The net with the planner choosing its action decisions against the
same net without it (same kingdoms, seeds, both seats).

    python experiments/plan_vs.py opponents/rl_r2_2200.npy --k 32 --leaf 0 --games 800
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def _job(job):
    from rl.core import NET, Seat, base_genome, lib, outcome, play, set_planner
    P = np.load(job["net"])
    Q = np.load(job["opp"]) if job["opp"] else P
    set_planner(P)
    lib.rl_set_plan_leaf(job["leaf"])
    out = []
    for k, seed, seat in job["games"]:
        g = base_genome(k)["ints"]
        me, other = Seat(NET, g, P), Seat(NET, g, Q)
        seats = [me, other] if seat == 0 else [other, me]
        plan = (job["k"], 0) if seat == 0 else (0, job["k"])
        lib.rl_set_chain_rule(*((1, 0) if seat == 0 else (0, 1)) if job["rule"] else (0, 0))
        r = play(seats, k, seed, rng_seed=seed ^ 0x51, plan=plan, plan_play=(1, 1))
        out.append((outcome(r, seat) + 1) / 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("--opp", default="", help="opponent net (default: the same net without the planner)")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--leaf", type=int, default=0)
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--seed", default="plan_vs")
    ap.add_argument("--rule", action="store_true", help="the chain rule instead of / with the planner")
    ap.add_argument("--engine", action="store_true", help="engine kingdoms (chain_exam.engine_kingdom)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    a = ap.parse_args()
    from rl.core import random_kingdom
    rng = random.Random(a.seed)
    games = []
    while len(games) < a.games:
        if a.engine:
            from experiments.chain_exam import engine_kingdom
            k = engine_kingdom(rng)
        else:
            k = random_kingdom(rng)
        s = rng.getrandbits(32)
        games += [(k, s, 0), (k, s, 1)]
    t = time.time()
    jobs = [{"net": a.net, "opp": a.opp, "k": a.k, "leaf": a.leaf, "rule": a.rule, "games": games[i:i + 20]}
            for i in range(0, len(games), 20)]
    with mp.Pool(a.workers) as pool:
        pts = [p for part in pool.map(_job, jobs) for p in part]
    m, se = float(np.mean(pts)), float(np.std(pts) / np.sqrt(len(pts)))
    print(f"planner K={a.k} leaf={a.leaf} rule={a.rule} engine={a.engine}: {m:.3f} ± {se:.3f} over {len(pts)} games ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
