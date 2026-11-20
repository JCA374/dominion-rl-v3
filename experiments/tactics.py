"""What the model does, as lessons for a person: docs/data/tactics.json for the Learn page.

    .venv/bin/python experiments/tactics.py [--net runs/c1/net_2300.npy] [--games 3000]

The model (argmax, as on the Play page) plays itself, the GA base and rl-v1's champion on random kingdoms; every
decision it makes is recorded (C engine, rl.core.play). The game state is read back from the recorded inputs
(features of c/rl.c: coins, buys, turn, VP, Provinces left, every card count), the policy's probabilities from the
same net in numpy (play/rlplay.py). Out: openings, the money/green thresholds, the penultimate Province, how often
it buys each kingdom card, and quiz positions (buys where the choice is close or easy to get wrong).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "play"))

from rl import core  # noqa: E402
from rl.core import GA, NET, PASS, Seat, base_genome, random_kingdom  # noqa: E402

import rlplay  # noqa: E402
from dominion_engine.cards import CARDS, KINGDOM_CARDS, NAME, NUM_CARDS  # noqa: E402

N = NUM_CARDS
E = 5 * N
BUY = 1
PROVINCE, DUCHY, ESTATE, GOLD, SILVER, CURSE = (next(i for i in range(N) if NAME[i] == n)
                                                 for n in ("Province", "Duchy", "Estate", "Gold", "Silver", "Curse"))


def state(x):
    r = lambda v: int(round(float(v)))
    return {"coins": r(x[E] * 8), "buys": r(x[E + 1] * 2), "turn": r(x[E + 2] * 20) + 1,
            "vp": r(x[E + 4] * 20), "opp_vp": r(x[E + 5] * 20), "prov_left": r(x[E + 7] * 8),
            "mine": {NAME[c]: r(x[c] * 10) for c in range(N) if r(x[c] * 10)},
            "theirs": {NAME[c]: r(x[2 * N + c] * 10) for c in range(N) if r(x[2 * N + c] * 10)},
            "left": {NAME[c]: r(x[3 * N + c] * 10) for c in range(N) if x[4 * N + c] > 0}}


def label(a):
    return "nothing" if a == PASS else NAME[a]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=os.path.join(ROOT, "runs", "c1", "net_2300.npy"))
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--quiz", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "tactics.json"))
    a = ap.parse_args()
    W = np.load(a.net).astype(np.float32)
    net = rlplay.Net(W)
    rl1 = np.load(os.path.join(ROOT, "opponents", "rl_r2_2200.npy")).astype(np.float32)
    rng = random.Random(2026)
    open_by_split = defaultdict(Counter)            # "5/2" -> Counter("Witch + nothing")
    open_by_card = defaultdict(Counter)             # key card on board -> Counter(opening) for 4/3 and 5/2
    thresholds = {k: defaultdict(Counter) for k in ("8", "6-7", "5", "2-4")}   # coins -> prov_left -> Counter(buy)
    ppr = Counter()                                 # last Province(s): buys it when it would win / lose
    present, bought, copies = Counter(), Counter(), Counter()
    first_buy_turn = defaultdict(list)
    quiz_pool = []
    wins = Counter()
    for gi in range(a.games):
        k = random_kingdom(rng)
        seed = rng.getrandbits(32)
        kind = gi % 3                               # self-play, vs GA base, vs rl-v1
        me = Seat(NET, base_genome(k)["ints"], W, record=True)
        if kind == 0:
            opp, opp_name = Seat(NET, base_genome(k)["ints"], W, record=True), "itself"
        elif kind == 1:
            opp, opp_name = Seat(GA, base_genome(k)["ints"]), "GA base"
        else:
            opp, opp_name = Seat(NET, base_genome(k)["ints"], rl1), "rl-v1"
        seat = gi % 2
        seats = [None, None]
        seats[seat], seats[1 - seat] = me, opp
        res = core.play(seats, k, seed, rng_seed=seed ^ 0x5EED)
        rec = res["rec"]
        mine_seats = {seat} | ({1 - seat} if kind == 0 else set())
        for s in mine_seats:
            wins[opp_name] += (core.outcome(res, s) + 1) / 2 / len(mine_seats)
        X, L, A, K, S = rec["X"], rec["LEGAL"], rec["A"], rec["KIND"], rec["SEAT"]
        for s in mine_seats:
            idx = [j for j in range(len(A)) if S[j] == s]
            buys = [(j, state(X[j])) for j in idx if K[j] == BUY]
            # openings: the buys of turns 1 and 2 (a second buy in a turn is rare there)
            t1 = [(st, A[j]) for j, st in buys if st["turn"] == 1]
            t2 = [(st, A[j]) for j, st in buys if st["turn"] == 2]
            if t1 and t2:
                c1, c2 = t1[0][0]["coins"], t2[0][0]["coins"]
                split = f"{max(c1, c2)}/{min(c1, c2)}"
                hi, lo = (t1[0][1], t2[0][1]) if c1 >= c2 else (t2[0][1], t1[0][1])
                op = f"{label(hi)} + {label(lo)}"
                open_by_split[split][op] += 1
                for c in k:
                    open_by_card[c][op] += 1
            got = Counter()
            for j, st in buys:
                if A[j] != PASS:
                    got[A[j]] += 1
                c, pl = st["coins"], st["prov_left"]
                band = "8" if c >= 8 else "6-7" if c >= 6 else "5" if c == 5 else "2-4" if c >= 2 else None
                if band and pl > 0:
                    thresholds[band][min(pl, 8)][label(A[j])] += 1
                if c >= 8 and pl == 1 and L[j][PROVINCE]:
                    after = st["vp"] + 6 - st["opp_vp"]
                    ppr[("wins" if after > 0 else "ties" if after == 0 else "loses", A[j] == PROVINCE)] += 1
                legal = [o for o in range(len(L[j])) if L[j][o]]
                if len(legal) >= 3 and st["turn"] >= 3:
                    lg, v = net.forward(X[j])
                    z = lg[legal] - lg[legal].max()
                    pr = np.exp(z) / np.exp(z).sum()
                    order = np.argsort(-pr)
                    p1, p2 = float(pr[order[0]]), float(pr[order[1]])
                    if p1 >= 0.5 and p2 >= 0.08:       # a clear favourite, with a tempting second choice
                        quiz_pool.append({"kingdom": k, "state": st, "value": float(v),
                                          "options": [[label(legal[o]), float(pr[o])] for o in order if pr[o] >= 0.01],
                                          "answer": label(A[j]), "opp": opp_name})
            for c in k:
                cid = next(i for i in range(N) if NAME[i] == c)
                present[c] += 1
                if got[cid]:
                    bought[c] += 1
                    copies[c] += got[cid]
                    first_buy_turn[c].append(next(st["turn"] for j, st in buys if A[j] == cid))
    rng.shuffle(quiz_pool)
    seen, quiz = set(), []
    for q in sorted(quiz_pool, key=lambda q: (q["answer"] in ("Gold", "Province"), rng.random())):
        key = (q["answer"], q["state"]["coins"], q["state"]["prov_left"] // 3)
        if key in seen:
            continue
        seen.add(key)
        quiz.append(q)
        if len(quiz) >= a.quiz:
            break
    rng.shuffle(quiz)

    def top(cnt, n=5):
        tot = sum(cnt.values())
        return {"n": tot, "top": [[k_, v / tot] for k_, v in cnt.most_common(n)]}
    out = {
        "net": os.path.relpath(a.net, ROOT), "games": a.games,
        "score": {k_: v / (a.games / 3) for k_, v in wins.items()},
        "openings": {s: top(c) for s, c in sorted(open_by_split.items(), key=lambda kv: -sum(kv[1].values())) if sum(c.values()) >= 30},
        "openings_by_card": {c: top(open_by_card[c], 3) for c in KINGDOM_CARDS},
        "thresholds": {b: {str(pl): top(c, 12) for pl, c in sorted(d.items(), reverse=True)} for b, d in thresholds.items()},
        "penultimate": {f"{w}|{int(bp)}": n for (w, bp), n in ppr.items()},
        "cards": {c: {"present": present[c], "bought": bought[c] / present[c] if present[c] else 0,
                      "copies": copies[c] / bought[c] if bought[c] else 0,
                      "first_turn": float(np.median(first_buy_turn[c])) if first_buy_turn[c] else None,
                      "cost": CARDS[next(i for i in range(N) if NAME[i] == c)].cost}
                  for c in KINGDOM_CARDS},
        "quiz": quiz,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{a.games} games; score {out['score']}; {len(quiz_pool)} quiz candidates, {len(quiz)} kept -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
