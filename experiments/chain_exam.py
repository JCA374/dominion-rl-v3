"""The chain exam: can a network play its action cards in the right order, or does it need a planner?

Positions: action decisions (at least two options) from games on engine kingdoms (at least two villages or
cantrips and two terminal draw cards or Throne Room), half played by the GA base genome, half by rl-v1's champion,
both against the GA base. Each position is labelled by the turn planner with the coins judge: K determinisations of
the unseen cards, every option played out to the end of the turn by the champion net, scored by the coins (+$1 per
extra Buy) at the first buy decision. A choice is right when its expected coins are within $0.05 of the best.

    python experiments/chain_exam.py build                 # data/chain_exam.npz (once; ~5 min)
    python experiments/chain_exam.py score [name=net.npy ...]   # results/chain_exam.json (default: the known models)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

DATA = os.path.join(ROOT, "data", "chain_exam.npz")
OUT = os.path.join(ROOT, "results", "chain_exam.json")
CHAMPION = os.path.join(ROOT, "opponents", "rl_r2_2200.npy")
K = 128
TOL = 0.05
VILLAGES = ["Village", "Festival", "Laboratory", "Market", "Merchant", "Cellar", "Harbinger", "Poacher", "Sentry"]
DRAW = ["Smithy", "Council Room", "Library", "Witch", "Moat", "Throne Room"]
MODELS = {"rl-v1 released (r2 net 200)": "opponents/rl_r2_200.npy",
          "rl-v1 champion (r2 net 2,200)": "opponents/rl_r2_2200.npy",
          "rl-v2 from zero (z1h net 1,100)": "opponents/rl_v2_z1h_1100.npy"}


def engine_kingdom(rng):
    import rl.core  # noqa: F401  (engine on the path)
    from dominion_engine.cards import KINGDOM_CARDS
    k = rng.sample(VILLAGES, 2 + rng.randrange(2)) + rng.sample(DRAW, 2 + rng.randrange(2))
    rest = [c for c in KINGDOM_CARDS if c not in k]
    return sorted(k + rng.sample(rest, 10 - len(k)))


def _build(job):
    from rl.core import GA, NET, Seat, base_genome, lib, play, set_planner
    P = np.load(CHAMPION)
    set_planner(P)
    lib.rl_set_plan_leaf(1)
    out = []
    for k, seed, who in job:
        g = base_genome(k)["ints"]
        me = Seat(GA, g, record=True) if who == "GA" else Seat(NET, g, P, record=True)
        r = play([me, Seat(GA, g)], k, seed, rng_seed=seed ^ 0xC4A1, plan=(K, 0), plan_play=(0, 0))
        rec = r["rec"]
        m = (rec["PLAN"] >= 0) & (rec["SEAT"] == 0)
        for i in np.flatnonzero(m):
            out.append((rec["X"][i], rec["LEGAL"][i], int(rec["A"][i]), rec["PVAL"][i], who, " ".join(k)))
    return out


def build(n_games=600, workers=3):
    from rl.core import random_kingdom  # noqa: F401
    rng = random.Random("chain exam")
    jobs, cur = [], []
    for i in range(n_games):
        cur.append((engine_kingdom(rng), rng.getrandbits(32), "GA" if i % 2 else "champion"))
        if len(cur) == 10:
            jobs.append(cur)
            cur = []
    t = time.time()
    with mp.Pool(workers) as pool:
        rows = [r for part in pool.map(_build, jobs) for r in part]
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, X=np.stack([r[0] for r in rows]), LEGAL=np.stack([r[1] for r in rows]),
                        A=np.array([r[2] for r in rows], np.int32), PVAL=np.stack([r[3] for r in rows]),
                        WHO=np.array([r[4] for r in rows]), KINGDOM=np.array([r[5] for r in rows]))
    print(f"{len(rows)} positions from {n_games} games in {time.time() - t:.0f}s -> {DATA}")


def hand_type(x):
    """From the inputs: my hand is x[175:208] (0.2 per copy)."""
    from rl.core import ID, NAME, NUM_CARDS
    import rl.core  # noqa: F401
    from dominion_engine.cards import CARDS
    hand = [c for c in range(NUM_CARDS) for _ in range(int(round(x[175 + c] / 0.2)))]
    acts = [c for c in hand if CARDS[c].is_action]
    plus = [c for c in acts if CARDS[c].actions >= 1]
    terminal = [c for c in acts if CARDS[c].actions == 0 and c != ID["Throne Room"]]
    if ID["Throne Room"] in hand:
        return "Throne Room in hand"
    if plus and terminal:
        return "+Actions and terminals"
    if len(terminal) >= 2:
        return "terminal collision"
    if plus:
        return "+Actions cards only"
    _ = NAME
    return "one terminal (play it or not)"


def forward(params, X):
    from rl.core import NA, lib, net_shape
    import ctypes
    nh, nf = net_shape(params)
    F = ctypes.POINTER(ctypes.c_float)
    lg = np.zeros(NA, np.float32)
    out = np.zeros((len(X), NA), np.float32)
    for i, x in enumerate(np.ascontiguousarray(X, np.float32)):
        lib.rl_forward(params.ctypes.data_as(F), nh, nf, x.ctypes.data_as(F), lg.ctypes.data_as(F))
        out[i] = lg
    return out


def grade(choice, pval):
    best = np.nanmax(pval, 1)
    got = pval[np.arange(len(choice)), choice]
    return got >= best - TOL, best - got


def score(paths: dict):
    d = np.load(DATA)
    X, L, A, PV, WHO = d["X"], d["LEGAL"].astype(bool), d["A"], d["PVAL"], d["WHO"]
    types = np.array([hand_type(x) for x in X])
    res = {"positions": int(len(X)), "K": K, "tolerance": TOL, "judge": "expected coins at the first buy decision "
           "(+$1 per extra Buy), turn played out by rl-v1's champion net", "types": {}, "models": {}}
    for t in sorted(set(types)):
        res["types"][t] = int((types == t).sum())
    def summary(choice, mask):
        ok, reg = grade(choice[mask], PV[mask])
        row = {"n": int(mask.sum()), "right": float(ok.mean()), "coin_regret": float(reg.mean()),
               "by_type": {}}
        for t in res["types"]:
            m = types[mask] == t
            if m.sum():
                row["by_type"][t] = {"n": int(m.sum()), "right": float(ok[m].mean()), "coin_regret": float(reg[m].mean())}
        return row
    ga = WHO == "GA"
    res["models"]["GA base genome (its own positions)"] = summary(A, ga)
    for name, p in paths.items():
        lg = forward(np.load(os.path.join(ROOT, p)), X)
        lg[~L] = -1e9
        res["models"][name] = summary(lg.argmax(1), np.ones(len(X), bool))
    # examples the champion gets wrong
    lg = forward(np.load(CHAMPION), X)
    lg[~L] = -1e9
    ch = lg.argmax(1)
    ok, reg = grade(ch, PV)
    from rl.core import NAME, NUM_CARDS
    def nm(c):
        return NAME[c] if c < NUM_CARDS else "end actions"
    ex = []
    for i in np.argsort(-reg)[:12]:
        if ok[i]:
            break
        hand = [NAME[c] for c in range(NUM_CARDS) for _ in range(int(round(X[i][175 + c] / 0.2)))]
        ex.append({"hand": hand, "actions_left": round(float(X[i][175 + 2 * NUM_CARDS] * 3)),
                   "net": nm(int(ch[i])), "planner": nm(int(np.nanargmax(PV[i]))),
                   "coins": {nm(c): round(float(v), 2) for c, v in enumerate(PV[i]) if not np.isnan(v)}})
    res["champion_misses"] = ex
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    old = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res["models"] = {**old.get("models", {}), **res["models"]}
    res["t"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(res, open(OUT, "w"), indent=1)
    for name, r in res["models"].items():
        print(f"{name:42s} right {r['right']:.3f}  regret ${r['coin_regret']:.3f}  (n={r['n']})")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        extra = dict(a.split("=", 1) if "=" in a else (os.path.basename(a), a) for a in sys.argv[2:])
        score(extra or MODELS)
