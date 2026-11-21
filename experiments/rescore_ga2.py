"""Every model against the old and the new GA base, on the same fresh games (2026-11-21).

    .venv/bin/python experiments/rescore_ga2.py [--kingdoms 1000] [--only label-substring]

Old GA base: az-v1/az/base.py base_genome (portfolio-v2, best-vs-bots eligible member). New GA base: base_genome_v2
(portfolio-v3's members, round robin among the eligible ones on the dealt kingdom). Each model plays 1,000 fresh
kingdoms, both seats, against each base: the same games for both, so the two scores compare directly. Nets and
az-v1's agent keep the old base genome for their own non-buy decisions (part of each model). Out:
results/rescore_ga2.json (a site's data/rescore_ga2.json).
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

MODELS = [
    ("GA base, old rule (portfolio-v2)", {"kind": "ga"}),
    ("portfolio-v2 (the Steam autopilot's GA)", {"kind": "portfolio", "name": "portfolio-v2"}),
    ("portfolio-v3 (17 improved specialists)", {"kind": "portfolio", "name": "portfolio-v3"}),
    ("az-v1 best (net 357 + 400 simulations)", {"kind": "az", "path": "opponents/az_r2_357.npy", "sims": 400}),
    ("rl-v1 confirmed (r2 iteration 200)", {"kind": "net", "path": "opponents/rl_r2_200.npy"}),
    ("rl-v1 champion (r2 iteration 2200)", {"kind": "net", "path": "opponents/rl_r2_2200.npy"}),
    ("rl-v2 from zero (z1h iteration 1100)", {"kind": "net", "path": "opponents/rl_v2_z1h_1100.npy"}),
    ("rl-v3 c1 iteration 2300", {"kind": "net", "path": "runs/c1/net_2300.npy"}),
    ("rl-v3 c2 iteration 2654", {"kind": "net", "path": "runs/c2/net_2654.npy"}),
    ("Big Money (bot)", {"kind": "bot", "name": "Big Money"}),
    ("Witch-BM (bot)", {"kind": "bot", "name": "Witch-BM"}),
]
_P: dict = {}


def _params(path):
    if path not in _P:
        _P[path] = np.load(os.path.join(ROOT, path)).astype(np.float32)
    return _P[path]


def _pick_v2(k):
    from az.base import base_genome_v2
    return base_genome_v2(k)["name"]


def seat(spec, k, pick2=None):
    from az.base import members_v3
    from rl.core import AZ_SEAT, GA, NET, Seat, base_genome, bot
    g = base_genome(k)["ints"]
    kind = spec["kind"]
    if kind == "ga":
        return Seat(GA, g)
    if kind == "ga2":
        return Seat(GA, next(m for m in members_v3() if m["name"] == pick2)["ints"])
    if kind == "bot":
        return Seat(GA, bot(spec["name"], k))
    if kind == "portfolio":                       # the released GA with its own trial picker
        from ga.c_bridge import strategy_to_ints
        from policy.portfolio import choose, members
        from policy.registry import read_manifest, resolve
        key = ("pf", spec["name"])
        if key not in _P:
            man = read_manifest(resolve(spec["name"]))
            _P[key] = (man, members(man))
        man, st = _P[key]
        return Seat(GA, strategy_to_ints(st[choose(man, k, strategies=st)["member"]]))
    if kind == "az":
        return Seat(AZ_SEAT, g, _params(spec["path"]), sims=spec.get("sims", 0))
    return Seat(NET, g, _params(spec["path"]))


def job(j):
    from rl.core import outcome, play
    out = []
    for gm in j["games"]:
        k = gm["kingdom"]
        me = seat(j["spec"], k)
        for base in ("ga", "ga2"):
            other = seat({"kind": base}, k, gm["pick2"])
            seats = [me, other] if gm["seat"] == 0 else [other, me]
            r = play(seats, k, gm["seed"], rng_seed=gm["seed"] ^ 0x5EED)
            out.append({"label": j["label"], "base": base, "points": (outcome(r, gm["seat"]) + 1) / 2})
    return out


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kingdoms", type=int, default=1000)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "rescore_ga2.json"))
    a = ap.parse_args()
    from rl.core import random_kingdom
    rng = random.Random("GA base v2 rescore, 2026-11-21")
    ks = [(random_kingdom(rng), rng.getrandbits(32)) for _ in range(a.kingdoms)]
    t = time.time()
    with mp.Pool(a.workers) as pool:
        picks = pool.map(_pick_v2, [k for k, _ in ks], chunksize=4)
    print(f"new-base picks for {len(ks)} kingdoms in {time.time() - t:.0f} s", flush=True)
    games = [{"kingdom": k, "seed": s, "seat": seat_, "pick2": p} for (k, s), p in zip(ks, picks) for seat_ in (0, 1)]
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    rows = {r["label"]: r for r in old.get("models", [])}
    for label, spec in MODELS:
        if a.only and a.only not in label:
            continue
        n = 4 if spec["kind"] in ("az", "portfolio") else 40
        jobs = [{"label": label, "spec": spec, "games": games[i:i + n]} for i in range(0, len(games), n)]
        t = time.time()
        with mp.Pool(a.workers) as pool:
            res = [r for part in pool.imap_unordered(job, jobs) for r in part]
        row = {"label": label, "spec": spec, "secs": round(time.time() - t)}
        for base in ("ga", "ga2"):
            p = [r["points"] for r in res if r["base"] == base]
            m = float(np.mean(p))
            row["vs_" + base] = {"score": round(m, 4), "ci": wilson(m, len(p)), "n": len(p)}
        rows[label] = row
        out = {"t": time.strftime("%Y-%m-%d %H:%M"), "kingdoms": a.kingdoms,
               "games": "1,000 fresh kingdoms, both seats; the same games against both bases",
               "old_base": "az-v1 base_genome: portfolio-v2, the best-vs-bots eligible member",
               "new_base": "az-v1 base_genome_v2: portfolio-v3, round robin among the eligible members (100 seeds)",
               "pick_counts": {n_: picks.count(n_) for n_ in sorted(set(picks), key=lambda x: -picks.count(x))},
               "models": [rows[l_] for l_, _ in MODELS if l_ in rows]}
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"{label:45s} vs old base {100 * row['vs_ga']['score']:5.1f}%   vs new base {100 * row['vs_ga2']['score']:5.1f}%"
              f"   ({row['secs']} s)", flush=True)


if __name__ == "__main__":
    main()
