"""Runs side by side at the same iterations: the goal check (vs GA base, vs rl-v1) and how they use their turns.

    python experiments/compare_runs.py z1 z1ent --iters 825 850
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--iters", nargs="+", type=int, required=True)
    ap.add_argument("--kingdoms", type=int, default=100)
    a = ap.parse_args()
    import random

    import turn_usage as tu
    from rl.core import random_kingdom
    rng = random.Random("rl-v2 compare runs")
    want = {"Village", "Festival", "Market", "Laboratory", "Council Room"}
    games = []
    while len(games) < 2 * a.kingdoms:
        k = random_kingdom(rng)
        if want & set(k):
            s = rng.getrandbits(32)
            games += [(k, s, 0), (k, s, 1)]
    for it in a.iters:
        for run in a.runs:
            g = f"results/goal_{run}_iter{it}.json"
            n = f"runs/{run}/net_{it}.npy"
            if not os.path.exists(n):
                print(f"{run} {it}: no net")
                continue
            d = json.load(open(g)) if os.path.exists(g) else {}
            u = tu.usage(np.load(n), games)
            print(f"{run:8s} {it:5d}  vs GA {d.get('score', float('nan')):.3f}  vs rl-v1 {d.get('vs_rl', float('nan')):.3f}"
                  f"  actions/turn {u['actions per turn']:.2f}  2+ actions {u['turns with 2+ actions']:.1%}"
                  f"  drawn/turn {u['cards drawn by actions per turn']:.2f}  top {[p[0] for p in u['most played'][:4]]}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
