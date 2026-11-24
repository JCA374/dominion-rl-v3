"""rl-v3 training: rl-v1's best network (r2 at iteration 2,219, 81-83% against the GA base) goes on learning with
PPO, now reading rl-v2's helper inputs (what the next hand brings, what one more copy of a card changes), through a
curriculum that ends with beating that same network.

    1 home league   rl-v1 r2's own opponents (GA base, bots, az-v1 net and search, past snapshots)
    2 champion      + rl-v1's frozen champion (net 2,200) as a quarter of the games
    3 open league   + rl-v2's from-zero net; everything earlier

    python train.py --run c1                 # new run in runs/c1/ (starts from opponents/rl_v1_r2_2219_ckpt.pt)
    python train.py --run c1 --continue      # resume from runs/c1/ckpt.pt

Per iteration: `games` games with the learner sampling from its policy, a PPO update, then the fixed evaluation
(argmax play; rl-v1's evaluation games): 400 games against the GA base genome, 300 against the bots, and 400 head to
head against rl-v1's champion (net 2,200).
Writes runs/<run>/stats.csv, events.jsonl, net_<iter>.pt/.npy, ckpt.pt, config.json, curriculum.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np

CFG = {
    "nh": 256,
    "nf": 0,                 # inputs the net reads: 0 = all (with the helper inputs), 298 = rl-v1's only
    "bc_games": 0,           # rl-v3 starts from rl-v1's checkpoint (INIT_CKPT): no imitation, no random start
    "bc_epochs": 6,
    "games": 3000,           # PPO games per iteration
    "focus_share": 0.0,      # share of training games on kingdoms without the focus_exclude cards (kingdom curriculum)
    "focus_exclude": "Witch,Militia,Bandit,Bureaucrat,Gardens",
    "drill_share": 0.0,      # card drill: share of non-bot games on a kingdom with one of the drill cards (see DRILL)
    "drill": "Harbinger,Workshop,Mine,Bureaucrat,Militia,Witch,Vassal,Sentry,Gardens,Library",
    "snapshot_every": 10,
    "snapshots": 8,
    "temp": 1.0,
    "lam": 0.95,             # GAE
    "clip": 0.2,
    "epochs": 3,
    "batch": 4096,
    "lr": 3e-4,
    "vf": 0.5,               # value loss weight
    "ent": 0.01,             # entropy bonus (after the schedule below)
    "ent_start": 0.0,        # >0: the entropy bonus starts here and falls linearly to ent over ent_iters iterations
    "ent_iters": 600,
    "max_kl": 0.03,          # stop the epoch loop early above this approximate KL
    "eval_kingdoms": 100,    # vs GA base: x 2 seeds x 2 seats = 400 games; bots: 25 x 2 x 2 x 3 = 300
    "workers": max(1, os.cpu_count() - 1),
    "iterations": 100000,
    "goal_every": 25,
    "vs_az_every": 100,      # goal checks with the 1,000 head-to-head games against az-v1's best (~3 min)
    "chunk": 50,
    "seed": "",              # training stream seed; "" = the run name (c1, c2). Set it so two runs can share a stream
    "frozen": "",            # frozen opponents and their shares before the stage's league, e.g. "c2:0.1,c1:0.1" (FROZEN)
    "eval_vs": "",           # extra evaluation opponents from FROZEN, e.g. "c2,c1" (100 kingdoms x 2 seats each)
    "exclude": "",           # JSON file of kingdoms never trained on ({"kingdoms": ["Card,Card,..."]}, sorted names)
}
BOTS = ["Big Money", "Smithy-BM", "Witch-BM"]
AZ_NET = "opponents/az_r2_357.npy"
RL_V1 = ["opponents/rl_r2_200.npy", "opponents/rl_r2_2200.npy"]   # rl-v1's released net and its r2 at 2,200
# Card school: cards rl-v1's champion rarely or never buys (card audit, results/card_audit_champion.json). In a
# `school` share of the games the learner's first buys are forced to 2 copies of one (not recorded: it learns to
# play them, its buying stays its own) on a kingdom with that card.
SCHOOL = ["Chapel", "Mine", "Library", "Moneylender", "Council Room", "Smithy", "Village", "Merchant", "Remodel",
          "Throne Room"]
CHAMPION = "opponents/rl_r2_2200.npy"                              # the best model before rl-v3
RL_V2 = "opponents/rl_v2_z1h_1100.npy"                            # rl-v2's from-zero net (75% vs GA at its peak)
INIT_CKPT = "opponents/rl_v1_r2_2219_ckpt.pt"                      # rl-v1 r2's last checkpoint: net + Adam state
FROZEN = {"c2": "opponents/rl3_c2_2654.npy", "c1": "opponents/rl3_c1_2300.npy",   # rl-v3's frozen best two
          "champion": CHAMPION}

# Opponent shares per stage (the rest: the current net, both seats learning). gate: (eval key, level)
# that ends the stage, or ("plateau", n): no new best against the GA base for n iterations (self-play
# runs until it stops improving on its own); budget: iterations after which it ends anyway.
STAGES = [
    {"name": "home league", "shares": {"past": 0.15, "ga": 0.3, "bots": 0.15, "aznet": 0.15, "azsearch": 0.05},
     "gate": ("GA base", 0.84), "budget": 300,
     "expect": "rl-v1's own opponents, so the only change is the new inputs: learns to use them without forgetting."},
    {"name": "champion", "shares": {"past": 0.15, "champion": 0.25, "ga": 0.2, "bots": 0.1, "aznet": 0.05,
                                    "azsearch": 0.05}, "school": 0.15,
     "gate": ("champion", 0.56), "budget": 600,
     "expect": "A quarter of the games against rl-v1's frozen best net: finds and exploits its habits."},
    {"name": "open league", "shares": {"past": 0.2, "champion": 0.15, "rlv1": 0.05, "rlv2": 0.1, "ga": 0.15,
                                       "bots": 0.1, "aznet": 0.05, "azsearch": 0.05}, "school": 0.15,
     "gate": None, "budget": None,
     "expect": "Every earlier model, so beating the champion does not cost strength elsewhere."},
]


def chunks(games, n):
    return [games[i:i + n] for i in range(0, len(games), n)]


def _ppo(job):
    from rl.jobs import ppo_chunk
    return ppo_chunk(job)


def _bc(job):
    from rl.jobs import bc_chunk
    return bc_chunk(job)


def _eval(job):
    from rl.jobs import eval_chunk
    return eval_chunk(job)


def eval_games(cfg):
    from rl.core import kingdom_for_bot, random_kingdom
    rng = random.Random("rl-v1 eval")         # (the champion games come after rl-v1's; GA and bot games unchanged)
    games = []
    for _ in range(cfg["eval_kingdoms"]):
        k = random_kingdom(rng)
        for _ in range(2):
            s = rng.getrandbits(32)
            games += [{"kingdom": k, "seed": s, "seat": seat, "opp": {"kind": "ga", "tag": "GA base"}} for seat in (0, 1)]
    for _ in range(cfg["eval_kingdoms"]):
        k = random_kingdom(rng)
        s = rng.getrandbits(32)
        games += [{"kingdom": k, "seed": s, "seat": seat, "opp": {"kind": "net", "path": CHAMPION, "tag": "champion"}}
                  for seat in (0, 1)]
    for name in filter(None, cfg.get("eval_vs", "").split(",")):     # after the fixed games: they stay unchanged
        r2 = random.Random(f"rl-v3 eval vs {name}")
        for _ in range(cfg["eval_kingdoms"]):
            k = random_kingdom(r2)
            s = r2.getrandbits(32)
            games += [{"kingdom": k, "seed": s, "seat": seat, "opp": {"kind": "net", "path": FROZEN[name],
                                                                      "tag": f"vs {name}"}} for seat in (0, 1)]
    for b in BOTS:
        for _ in range(cfg["eval_kingdoms"] // 4):
            k = kingdom_for_bot(rng, b)
            for _ in range(2):
                s = rng.getrandbits(32)
                games += [{"kingdom": k, "seed": s, "seat": seat, "opp": {"kind": "bot", "name": b, "tag": b}}
                          for seat in (0, 1)]
    return games


def run_eval(pool, cfg, agent):
    jobs = [{"games": c, "agent": agent, "nh": cfg["nh"]} for c in chunks(eval_games(cfg), cfg["chunk"])]
    res = [r for part in pool.imap(_eval, jobs) for r in part]
    out = {"cpu": float(sum(r.get("cpu", 0.0) for r in res)), "errors": int(sum(r.get("error", False) for r in res))}
    for tag in ["GA base", "champion"] + BOTS + [f"vs {n}" for n in filter(None, cfg.get("eval_vs", "").split(","))]:
        pts = [r["points"] for r in res if r["opp"] == tag]
        out[tag] = float(np.mean(pts))
    out["bots"] = float(np.mean([out[b] for b in BOTS]))
    out["turns"] = float(np.mean([r["turns"] for r in res]))
    out["capped"] = float(np.mean([r["capped"] for r in res]))
    return out


def fmt(e):
    return f"GA {e['GA base']:.3f} champ {e['champion']:.3f} BM {e['Big Money']:.3f} Smithy {e['Smithy-BM']:.3f} Witch {e['Witch-BM']:.3f} bots {e['bots']:.3f}"


class Log:
    def __init__(self, d):
        self.d = d

    def event(self, kind, **kw):
        with open(os.path.join(self.d, "events.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **kw}) + "\n")
        print(f"[{kind}]", json.dumps(kw)[:300], flush=True)

    def row(self, row):
        path = os.path.join(self.d, "stats.csv")
        new = not os.path.exists(path)
        fields = list(row)
        if not new:                                  # a resumed run keeps its file's columns (new ones are dropped)
            with open(path) as f:
                fields = next(csv.reader(f))
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def policy_logp(net, X, L, temp, torch):
    """log pi(a|x) for all options at the sampling temperature: the C sampler draws from softmax(logits / temp) and
    records that log probability, so the update must divide by the same temp (else the first ratio is already off)."""
    logits, v = net(X, L)
    return torch.log_softmax(logits / temp, -1), v


def ppo_update(net, opt, data, cfg, torch, seed=None):
    """Clipped PPO on one iteration's decisions. Returns mean metrics. seed: the minibatch shuffle's seed."""
    n = len(data["A"])
    gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    X = torch.from_numpy(data["X"])
    L = torch.from_numpy(data["LEGAL"]).bool()
    A = torch.from_numpy(data["A"]).long()
    OLD = torch.from_numpy(data["LOGP"])
    adv = data["ADV"]
    ADV = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
    RET = torch.from_numpy(data["RET"])
    stats = []
    stop = False
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg["batch"]):
            j = perm[i:i + cfg["batch"]]
            logp_all, v = policy_logp(net, X[j], L[j], cfg["temp"], torch)
            logp = logp_all.gather(1, A[j][:, None]).squeeze(1)
            ratio = torch.exp(logp - OLD[j])
            s1 = ratio * ADV[j]
            s2 = torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * ADV[j]
            pg = -torch.min(s1, s2).mean()
            p = logp_all.exp()
            ent = -(p * torch.where(L[j], logp_all, torch.zeros_like(logp_all))).sum(-1).mean()
            vl = ((v - RET[j]) ** 2).mean()
            loss = pg + cfg["vf"] * vl - cfg["ent_now"] * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                kl = (OLD[j] - logp).mean().item()
                clipfrac = ((ratio - 1).abs() > cfg["clip"]).float().mean().item()
            stats.append({"pg_loss": pg.item(), "value_loss": vl.item(), "entropy": ent.item(), "kl": kl,
                          "clipfrac": clipfrac, "updates": 1})
            if kl > cfg["max_kl"]:
                stop = True
                break
        if stop:
            break
    out = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    out["updates"] = len(stats)
    out["epochs_done"] = ep + 1
    return out


def bc_train(net, data, cfg, torch, log):
    """Imitation of the GA genome: cross-entropy on its choice + 0.5 x value error on the result.
    Batches index the numpy arrays (no copies: the data is ~1 GB)."""
    n = len(data["A"])
    idx = np.arange(n)
    hold = idx[idx % 10 == 0][:200000]
    train_idx = idx[idx % 10 != 0]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def batch(j):
        return (torch.from_numpy(data["X"][j]), torch.from_numpy(data["LEGAL"][j]).bool(),
                torch.from_numpy(data["A"][j]).long(), torch.from_numpy(data["Z"][j]))

    for ep in range(cfg["bc_epochs"]):
        perm = np.random.default_rng(ep).permutation(train_idx)
        for i in range(0, len(perm), cfg["batch"]):
            x, lg, a, z = batch(np.sort(perm[i:i + cfg["batch"]]))
            logits, v = net(x, lg)
            loss = torch.nn.functional.cross_entropy(logits, a) + 0.5 * ((v - z) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            x, lg, a, z = batch(hold)
            logits, v = net(x, lg)
            pick = logits.argmax(1)
            buy = torch.from_numpy(data["KIND"][hold] == 1)
            res = {"acc": (pick == a).float().mean().item(), "acc_buys": (pick == a)[buy].float().mean().item(),
                   "acc_other": (pick == a)[~buy].float().mean().item(),
                   "ce": torch.nn.functional.cross_entropy(logits, a).item(), "value_mse": ((v - z) ** 2).mean().item()}
        log.event("bc_epoch", epoch=ep, samples=n, holdout=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r1")
    ap.add_argument("--continue", dest="cont", action="store_true")
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--stage", type=int, default=None, help="the stage to start in (0-based; required with --init_from)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from rl.core import kingdom_for_bot, random_kingdom
    from rl.core import NF
    from rl.net import Net, load, save

    d = os.path.join("runs", args.run)
    os.makedirs(d, exist_ok=True)
    log = Log(d)
    cfg = dict(CFG)
    if args.cont and os.path.exists(os.path.join(d, "config.json")):
        cfg.update(json.load(open(os.path.join(d, "config.json"))))
    for kv in args.set:
        k, v = kv.split("=")
        cfg[k] = type(CFG[k])(v)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    write_curriculum(d)
    pool = mp.Pool(cfg["workers"])
    t0 = time.time()

    def snap(it):
        return os.path.join(d, f"net_{it}.npy")

    if args.cont:
        ck = torch.load(os.path.join(d, "ckpt.pt"), weights_only=False)
        net = Net(cfg["nh"], ck["net"]["l1.weight"].shape[1])
        net.load_state_dict(ck["net"])
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:                 # the config's learning rate wins over the saved one
            g["lr"] = cfg["lr"]
        it, snaps = ck["iter"] + 1, ck["snaps"]
        stage, stage_start = ck.get("stage", 0), ck.get("stage_start", 1)
        totals = ck.get("totals", {"games": 0, "updates": 0, "cpu_s": 0.0})
        log.event("run_resume", iter=it, stage=STAGES[stage]["name"])
    else:
        log.event("run_start", config=cfg, init_from=args.init_from)
        totals = {"games": 0, "updates": 0, "cpu_s": 0.0}
        if args.init_from:
            if args.stage is None:
                raise SystemExit("--init_from needs an explicit --stage (0 home league, 1 champion, 2 open league)")
            net = load(args.init_from)
        elif os.path.exists(INIT_CKPT):
            net, opt_state = widen_ckpt(INIT_CKPT, cfg, torch)
        elif not cfg["bc_games"]:
            torch.manual_seed(0)
            net = Net(cfg["nh"], cfg["nf"] or NF)
        else:
            net = Net(cfg["nh"], cfg["nf"] or NF)
            rng = random.Random("rl-v2 bc")
            games = []
            for i in range(cfg["bc_games"]):
                if i % 2 == 0:
                    games.append({"kingdom": random_kingdom(rng), "seed": rng.getrandbits(32)})
                else:
                    b = BOTS[i // 2 % 3]
                    games.append({"kingdom": kingdom_for_bot(rng, b), "seed": rng.getrandbits(32), "bot": b,
                                  "seat": i // 6 % 2})
            tb = time.time()
            parts = list(pool.imap_unordered(_bc, [{"games": c} for c in chunks(games, cfg["chunk"])]))
            data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            log.event("bc_data", games=len(games), decisions=int(len(data["A"])), secs=round(time.time() - tb))
            bc_train(net, data, cfg, torch, log)
            del data
        save(net, os.path.join(d, "net_0.pt"))
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        if not args.init_from and os.path.exists(INIT_CKPT):
            opt.load_state_dict(opt_state)
            for g in opt.param_groups:
                g["lr"] = cfg["lr"]
        if args.init_from:                         # step zero: the exported net is the source net, C and torch agree
            log.event("init_check", **init_check(net, args.init_from, snap(0), cfg, pool, torch))
        e = run_eval(pool, cfg, {"kind": "net", "path": snap(0)})
        log.event("init_eval", eval=fmt(e))
        log.row(make_row(0, t0, {}, {}, e))
        it, snaps = 1, [snap(0)]
        stage, stage_start = args.stage or 0, 1
        log.event("stage_start", stage=STAGES[stage]["name"], index=stage, iter=1, expect=STAGES[stage].get("expect", ""))
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "iter": 0, "snaps": snaps,
                    "stage": stage, "stage_start": stage_start, "totals": totals}, os.path.join(d, "ckpt.pt"))

    excluded = load_exclusions(cfg["exclude"])
    if excluded:
        log.event("exclusions", file=cfg["exclude"], kingdoms=len(excluded))
    cpu0 = time.process_time()
    stage_best = [-1.0, it - 1]              # best eval vs GA within the current stage, and its iteration
    best_eval = [-1.0]
    if os.path.exists(os.path.join(d, "stats.csv")):
        with open(os.path.join(d, "stats.csv")) as f:
            best_eval[0] = max(float(r["eval_GA_base"]) for r in csv.DictReader(f))
    while it <= cfg["iterations"]:
        ti = time.time()
        learner = snap(it - 1)
        games = make_games(cfg, stage, it, args.run, snaps, learner, excluded)
        jobs = [{"games": c, "learner": learner, "nh": cfg["nh"], "temp": cfg["temp"], "lam": cfg["lam"]}
                for c in chunks(games, cfg["chunk"])]
        parts = list(pool.imap(_ppo, jobs))                 # in job order: the same stream gives the same records
        res = [r for p in parts for r in p["results"]]
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k not in ("results", "cpu")}
        t_play = time.time() - ti
        sp = {"selfplay_vs_ga": float(np.mean([r["points"] for r in res if r["opp"] == "GA base"] or [float("nan")])),
              "selfplay_vs_bots": float(np.mean([r["points"] for r in res if r["opp"] in BOTS] or [float("nan")])),
              "selfplay_vs_past": float(np.mean([r["points"] for r in res if r["opp"] == "past"] or [float("nan")])),
              "selfplay_vs_aznet": float(np.mean([r["points"] for r in res if r["opp"] == "AZ net"] or [float("nan")])),
              "selfplay_vs_azsearch": float(np.mean([r["points"] for r in res if r["opp"] == "AZ search"] or [float("nan")])),
              "selfplay_vs_rlv1": float(np.mean([r["points"] for r in res if r["opp"] == "rl-v1"] or [float("nan")])),
              "selfplay_vs_champion": float(np.mean([r["points"] for r in res if r["opp"] == "champion"] or [float("nan")])),
              "selfplay_vs_rlv2": float(np.mean([r["points"] for r in res if r["opp"] == "rl-v2"] or [float("nan")])),
              "selfplay_vs_frozen_c2": float(np.mean([r["points"] for r in res if r["opp"] == "frozen c2"] or [float("nan")])),
              "selfplay_vs_frozen_c1": float(np.mean([r["points"] for r in res if r["opp"] == "frozen c1"] or [float("nan")])),
              "school_games": sum(1 for r in res if r.get("school")),
              "school_points": float(np.mean([r["points"] for r in res if r.get("school")] or [float("nan")])),
              "drill_games": sum(1 for r in res if r.get("drill")),
              "drill_points": float(np.mean([r["points"] for r in res if r.get("drill")] or [float("nan")])),
              "decisions": int(len(data["A"])),
              "games": len(res),
              "rejected_error": sum(1 for r in res if r.get("error")),
              "rejected_overflow": sum(1 for r in res if r.get("overflow")),
              "capped_games": sum(1 for r in res if r.get("capped")),
              "points_all": float(np.mean([r["points"] for r in res])),
              "points_uncapped": float(np.mean([r["points"] for r in res if not r.get("capped")] or [float("nan")])),
              "value_mse_fresh": float(((data["V"] - data["Z"]) ** 2).mean()),
              "t_play": t_play}
        tt = time.time()
        net.train()
        cfg["ent_now"] = entropy_bonus(cfg, it)
        if sp["rejected_error"] or sp["rejected_overflow"]:
            log.event("rejected", iter=it, errors=sp["rejected_error"], overflow=sp["rejected_overflow"])
        m = ppo_update(net, opt, data, cfg, torch, seed=stream_seed(cfg, args.run, it))
        m["ent_coef"] = cfg["ent_now"]
        sp["t_train"] = time.time() - tt
        save(net, os.path.join(d, f"net_{it}.pt"))
        e = run_eval(pool, cfg, {"kind": "net", "path": snap(it)})
        sp["t_iter"] = time.time() - ti
        totals["games"] += len(res)
        totals["updates"] += m["updates"]
        totals["cpu_s"] += sum(p["cpu"] for p in parts) + e["cpu"] + (time.process_time() - cpu0)
        cpu0 = time.process_time()
        sp.update(games_total=totals["games"], updates_total=totals["updates"], cpu_hours=totals["cpu_s"] / 3600)
        sp["stage"] = stage + 1
        log.row(make_row(it, t0, m, sp, e))
        print(f"iter {it}: {fmt(e)} | train vs GA {sp['selfplay_vs_ga']:.3f} kl {m['kl']:.4f} ent {m['entropy']:.3f} "
              f"| {sp['t_iter']:.0f}s", flush=True)
        if e["GA base"] > best_eval[0]:
            best_eval[0] = e["GA base"]
            log.event("goal_best", iter=it, check={"score": round(e["GA base"], 4), "n": 4 * cfg["eval_kingdoms"]},
                      champion=snap(it), eval=fmt(e))
        if it % cfg["snapshot_every"] == 0:
            snaps = (snaps + [snap(it)])[-cfg["snapshots"]:]
        for old in range(max(0, it - 30), it - 2):          # keep snapshots, goal nets and the last two
            p = os.path.join(d, f"net_{old}.npy")
            if old % cfg["snapshot_every"] and old % cfg["goal_every"] and os.path.exists(p):
                os.remove(p)
                os.remove(p.replace(".npy", ".pt"))
        st = STAGES[stage]
        if e["GA base"] > stage_best[0]:
            stage_best[:] = [e["GA base"], it]
        if stage + 1 < len(STAGES):
            key, level = st["gate"]
            if key == "plateau":
                passed = it - stage_best[1] >= level
                why = f"no new best against the GA base for {level} iterations (best {stage_best[0]:.3f} at {stage_best[1]})"
            else:
                passed = e[key] >= level
                why = f"eval {key} {e[key]:.3f} >= {level}"
            capped = not passed and it - stage_start + 1 >= st["budget"]
            if passed or capped:
                log.event("stage_pass" if passed else "stage_cap", stage=st["name"], index=stage, iter=it,
                          why=why if passed else f"budget of {st['budget']} iterations used", champion=snap(it),
                          eval=fmt(e))
                stage, stage_start, stage_best[:] = stage + 1, it + 1, [-1.0, it]
                log.event("stage_start", stage=STAGES[stage]["name"], index=stage, iter=it + 1,
                          expect=STAGES[stage].get("expect", ""))
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "iter": it, "snaps": snaps,
                    "stage": stage, "stage_start": stage_start, "totals": totals}, os.path.join(d, "ckpt.pt"))
        if cfg["goal_every"] and it % cfg["goal_every"] == 0:
            import subprocess
            import sys
            subprocess.run([sys.executable, "experiments/goal_check.py", snap(it), "--name", f"{args.run}_iter{it}",
                            "--label", f"{args.run} iteration {it}", "--nh", str(cfg["nh"])]
                           + (["--vs_az"] if it % cfg["vs_az_every"] == 0 else []), check=False)
        it += 1


def stream_seed(cfg, run, it):
    """The training stream's integer seed at iteration it (minibatch shuffles): from cfg seed, else the run name."""
    import zlib
    return zlib.crc32(f"rl-v3 ppo {cfg['seed'] or run} {it}".encode())


def load_exclusions(path):
    """Kingdoms never trained on (sorted card names joined by commas), from a shared exclusion manifest."""
    if not path:
        return set()
    return {",".join(sorted(k.split(","))) for k in json.load(open(path))["kingdoms"]}


def parse_frozen(spec):
    """"c2:0.1,c1:0.1" -> [("c2", 0.1), ("c1", 0.1)]"""
    out = []
    for part in filter(None, spec.split(",")):
        name, share = part.split(":")
        assert name in FROZEN, name
        out.append((name, float(share)))
    return out


def make_games(cfg, stage, it, run, snaps, learner, excluded=frozenset()):
    """One iteration's training games. The stream is seeded by cfg seed (or the run name, as c1 and c2 were), so two
    runs with the same seed and settings play the same kingdoms, seeds, seats and opponents. Frozen opponents
    (cfg frozen) take their shares first; the stage's league shares the rest. Excluded kingdoms are redrawn."""
    from rl.core import kingdom_for_bot, random_kingdom
    rng = random.Random(f"rl-v3 ppo {cfg['seed']} {it}" if cfg["seed"] else f"rl-v3 ppo {run} {it}")
    shares = STAGES[stage]["shares"]
    frozen = parse_frozen(cfg.get("frozen", ""))

    def draw(fn, *a):
        for _ in range(1000):
            k = fn(rng, *a)
            if ",".join(sorted(k)) not in excluded:
                return k
        raise RuntimeError("could not draw a kingdom outside the exclusions")

    games = []
    for i in range(cfg["games"]):
        u, seat, opp = rng.random(), i % 2, None
        if frozen:
            f = rng.random()
            for name, share in frozen:
                if f < share:
                    opp = {"kind": "net", "path": FROZEN[name], "tag": f"frozen {name}"}
                    break
                f -= share
        if opp is None:
            for kind, share in shares.items():
                if u < share:
                    opp = pick_opponent(kind, i, rng, snaps)
                    break
                u -= share
        if opp is None or (opp["kind"] == "net" and opp["path"] is None):
            opp = {"kind": "net", "path": learner, "tag": "self"}
        force = opp_force = drill = None
        if opp["kind"] == "bot":
            k = draw(kingdom_for_bot, opp["name"])
        elif rng.random() < cfg["drill_share"]:
            drill = DRILL_CARDS(cfg)[i % len(DRILL_CARDS(cfg))]
            k = draw(school_kingdom, drill)
            u2 = rng.random()
            if u2 < 0.3:
                force = drill                                  # the learner plays it
            elif u2 < 0.6 and opp["kind"] == "net":
                opp_force = drill                              # the learner plays against it
        elif rng.random() < STAGES[stage].get("school", 0):
            force = SCHOOL[i % len(SCHOOL)]
            k = draw(school_kingdom, force)
        elif rng.random() < cfg["focus_share"]:
            k = draw(focus_kingdom, cfg["focus_exclude"].split(","))
        else:
            k = draw(random_kingdom)
        games.append({"kingdom": k, "seed": rng.getrandbits(32), "seat": seat, "opp": opp, "force": force,
                      "opp_force": opp_force, "drill": drill})
    return games


def ratio_check(net, data, temp, torch):
    """The first PPO ratio pi_torch(a) / pi_C(a) on records the C sampler just made with this net: 1 up to float
    rounding when the update uses the sampler's temperature."""
    with torch.no_grad():
        logp_all, _ = policy_logp(net, torch.from_numpy(data["X"]), torch.from_numpy(data["LEGAL"]).bool(), temp, torch)
        lp = logp_all.gather(1, torch.from_numpy(data["A"]).long()[:, None]).squeeze(1).numpy()
    r = np.exp(lp - data["LOGP"])
    return {"decisions": int(len(r)), "mean_abs_ratio_err": float(np.abs(r - 1).mean()),
            "outside_clip": float((np.abs(r - 1) > 0.2).mean())}


def init_check(net, src, snap0, cfg, pool, torch, games=24):
    """Step-zero checks for a run started from a saved net: the exported weights equal the source's .npy (if any),
    and a short sampled collection gives first PPO ratios of 1 (C sampling and torch agree, temperature included)."""
    out = {}
    npy = src.replace(".pt", ".npy")
    if os.path.exists(npy):
        out["weights_equal_source"] = bool(np.array_equal(np.load(npy), np.load(snap0)))
        if not out["weights_equal_source"]:
            raise SystemExit(f"init check: {snap0} differs from {npy}")
    from rl.core import random_kingdom
    rng = random.Random("rl-v3 init check")
    gs = [{"kingdom": random_kingdom(rng), "seed": rng.getrandbits(32), "seat": i % 2, "opp": {"kind": "ga"}}
          for i in range(games)]
    parts = list(pool.imap(_ppo, [{"games": c, "learner": snap0, "nh": cfg["nh"], "temp": cfg["temp"],
                                    "lam": cfg["lam"]} for c in chunks(gs, 4)]))
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "LEGAL", "A", "LOGP")}
    out.update(ratio_check(net, data, cfg["temp"], torch))
    if out["mean_abs_ratio_err"] > 1e-4:
        raise SystemExit(f"init check: first PPO ratios are off ({out})")
    return out


def write_curriculum(d):
    """runs/<run>/curriculum.json for the site's Curriculum tab, written before the stages run."""
    names = {"past": "own past snapshots", "bots": "the 3 bots", "ga": "GA base genome", "rlv1": "rl-v1's released net",
             "champion": "rl-v1's champion (net 2,200)", "rlv2": "rl-v2's from-zero net",
             "aznet": "az-v1 net", "azsearch": "az-v1 search"}
    stages = []
    for st in STAGES:
        opp = ", ".join(f"{names[k]} {v:.0%}" for k, v in st["shares"].items())
        rest = 1 - sum(st["shares"].values())
        g = st["gate"]
        school = (f"; card school in {st['school']:.0%} of the games (first buys forced to 2 copies of a card the "
                  f"champion rarely buys: {', '.join(SCHOOL)})") if st.get("school") else ""
        stages.append({"name": st["name"], "world": f"opponents: itself {rest:.0%}, {opp}{school}",
                       "reward": "+1 win, 0 shared win, -1 loss, at the end of the game only",
                       "pass": "never (the last stage)" if g is None else
                       f"no new best against the GA base (eval) for {g[1]} iterations" if g[0] == "plateau" else
                       f"eval against the {g[0]} at least {g[1]:.0%}",
                       "cap": f"{st['budget']} iterations" if st["budget"] else "none", "expect": st["expect"]})
    json.dump({"note": "PPO from rl-v1's best checkpoint (iteration 2,219) with rl-v2's helper inputs added at zero "
                       "weight. The evaluation games are only measured, never trained on.", "stages": stages},
              open(os.path.join(d, "curriculum.json"), "w"), indent=1)


def widen_ckpt(path, cfg, torch):
    """rl-v1's checkpoint (298 inputs) as a net reading all NF inputs: the new inputs get zero weights in the first
    layer and zero Adam moments, so the widened net plays exactly as before."""
    from rl.core import NF, NF0
    from rl.net import Net
    ck = torch.load(path, weights_only=False)

    def pad(t):
        assert t.shape[1] == NF0, t.shape
        return torch.cat([t, torch.zeros(t.shape[0], NF - NF0, dtype=t.dtype)], 1)

    keys = list(ck["net"].keys())
    ck["net"]["l1.weight"] = pad(ck["net"]["l1.weight"])
    st = ck["opt"]["state"][keys.index("l1.weight")]
    st["exp_avg"], st["exp_avg_sq"] = pad(st["exp_avg"]), pad(st["exp_avg_sq"])
    net = Net(cfg["nh"], NF)
    net.load_state_dict(ck["net"])
    return net, ck["opt"]


def entropy_bonus(cfg, it):
    """The entropy coefficient at iteration `it`: ent_start falling linearly to ent over ent_iters (or ent)."""
    if cfg["ent_start"] <= 0:
        return cfg["ent"]
    f = max(0.0, 1 - it / cfg["ent_iters"])
    return cfg["ent"] + (cfg["ent_start"] - cfg["ent"]) * f


def DRILL_CARDS(cfg):
    """Card drill (results/card_losses/: cards the model loses against): a kingdom with the card, and in 30% of
    those games the learner's first buys forced to 2 copies, in 30% (net opponents) the opponent's."""
    return cfg["drill"].split(",")


def school_kingdom(rng, card):
    """The card and 9 random others."""
    import rl.core  # noqa: F401
    from dominion_engine.cards import KINGDOM_CARDS
    return sorted([card] + rng.sample([c for c in KINGDOM_CARDS if c != card], 9))


def focus_kingdom(rng, exclude):
    """10 kingdom cards drawn from those not in `exclude`: a kingdom where the excluded strategies are impossible."""
    import rl.core  # noqa: F401  (puts the engine on the path)
    from dominion_engine.cards import KINGDOM_CARDS
    return sorted(rng.sample([c for c in KINGDOM_CARDS if c not in exclude], 10))


def pick_opponent(kind, i, rng, snaps):
    """One opponent of the given league kind ("net" with path None: not available yet, play self)."""
    if kind == "past":
        return {"kind": "net", "path": rng.choice(snaps[:-1]) if len(snaps) > 1 else None, "tag": "past"}
    if kind == "bots":
        b = BOTS[i % 3]
        return {"kind": "bot", "name": b, "tag": b}
    if kind == "ga":
        return {"kind": "ga", "tag": "GA base"}
    if kind == "rlv1":
        return {"kind": "net", "path": rng.choice(RL_V1), "tag": "rl-v1"}
    if kind == "champion":
        return {"kind": "net", "path": CHAMPION, "tag": "champion"}
    if kind == "rlv2":
        return {"kind": "net", "path": RL_V2, "tag": "rl-v2"}
    if kind == "aznet":
        return {"kind": "az", "path": AZ_NET, "sims": 0, "tag": "AZ net"}
    if kind == "azsearch":
        return {"kind": "az", "path": AZ_NET, "sims": 400, "tag": "AZ search"}
    raise ValueError(kind)


def make_row(it, t0, m, sp, e):
    row = {"iter": it, "hours": round((time.time() - t0) / 3600, 4)}
    for k in ("pg_loss", "value_loss", "entropy", "kl", "clipfrac", "epochs_done", "ent_coef", "updates"):
        row[k] = round(m[k], 5) if k in m else ""
    for tag in ["GA base", "champion"] + BOTS + ["bots"] + [t for t in e if t.startswith("vs ")]:
        row[f"eval_{tag.replace(' ', '_').replace('-', '_')}"] = round(e[tag], 4)
    row["eval_turns"] = round(e["turns"], 2)
    row["eval_capped"] = round(e["capped"], 4)
    row["stage"] = sp.get("stage", 1)
    for k in ("selfplay_vs_ga", "selfplay_vs_bots", "selfplay_vs_past", "selfplay_vs_aznet", "selfplay_vs_azsearch",
              "selfplay_vs_rlv1", "selfplay_vs_champion", "selfplay_vs_rlv2", "selfplay_vs_frozen_c2", "selfplay_vs_frozen_c1", "school_games", "school_points", "drill_games", "drill_points", "decisions",
              "value_mse_fresh", "games", "rejected_error", "rejected_overflow", "capped_games", "points_all",
              "points_uncapped", "games_total", "updates_total", "cpu_hours",
              "t_play", "t_train", "t_iter"):
        row[k] = (round(sp[k], 4) if isinstance(sp[k], float) and not math.isnan(sp[k]) else sp[k]) if k in sp else ""
    return row


if __name__ == "__main__":
    main()
