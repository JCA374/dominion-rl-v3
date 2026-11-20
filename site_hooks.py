"""Project hooks for publish.py (ml-project-site): the Play tab's game clips, pictures of the net for
The model tab, and the results file for the custom pages.

Clips use az-v1's drawing (one frame per buy of either player: both decks, the score, what was
bought, and for the network the probabilities it gave the top buys). Drawing lives here only; no
policy imports this file.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os

import numpy as np

from rl import core

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = "c1"                     # the main run (caches live in runs/RUN)
LINEAGE = ["c1"]
EVERY = 25                     # the Play tab's newest net moves every EVERY iterations (clips are committed)
N = core.NUM_CARDS
NAME = core.NAME

_spec = importlib.util.spec_from_file_location("az_site_hooks", os.path.join(ROOT, "az-v1", "site_hooks.py"))
AZH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AZH)          # az-v1's frame(), encode_mp4(), GAMES and the model-tab figure helpers


def _nets(runs=None):
    """The main lineage's nets, oldest first."""
    by_it = {}
    for run in runs or LINEAGE:
        for p in glob.glob(os.path.join(ROOT, "runs", run, "net_*.npy")):
            by_it[int(p.split("_")[-1][:-4])] = os.path.relpath(p, ROOT)
    return [by_it[i] for i in sorted(by_it)]


def models():
    """rl-v3's newest net, its start (rl-v1's checkpoint with the new inputs), rl-v1's champion and rl-v2."""
    nets = _nets()
    its = [int(p.split("_")[-1][:-4]) for p in nets]
    out = []
    newest = [p for p in nets if int(p.split("_")[-1][:-4]) % EVERY == 0 and not p.endswith("net_0.npy")]
    if newest:
        it = int(newest[-1].split("_")[-1][:-4])
        out.append({"id": f"c1-{it}", "label": f"rl-v3 (iteration {it})", "path": newest[-1],
                    "note": "The newest rl-v3 network."})
    if 0 in its:
        out.append({"id": "c1-0", "label": "rl-v3 start (iteration 0)", "path": nets[its.index(0)],
                    "note": "rl-v1's last checkpoint with the helper inputs added at zero weight: plays as rl-v1 did."})
    out.append({"id": "rl-v1", "label": "rl-v1 champion (r2, iteration 2,200)", "path": "opponents/rl_r2_2200.npy",
                "note": "The best model before rl-v3: 82.8% against the GA base, 71.7% against az-v1."})
    out.append({"id": "rl-v2", "label": "rl-v2 from zero (z1h, iteration 1,100)", "path": "opponents/rl_v2_z1h_1100.npy",
                "note": "The same network learned from random weights by self-play: ~74% against the GA base."})
    return out


def games():
    return AZH.games()


def _seat(path, kingdom):
    g = core.base_genome(kingdom)["ints"]
    if path == "ga":
        return core.Seat(core.GA, g), "GA base"
    if path == "az":
        return core.Seat(core.AZ_SEAT, g, np.load(os.path.join(ROOT, "opponents", "az_r2_357.npy")), sims=400), "AZ search"
    return core.Seat(core.NET, g, np.load(os.path.join(ROOT, path))), "RL net"


def render_play(policy_path, game, out_mp4):
    k = game["kingdom"]
    seat, label = _seat(policy_path, k)
    og = core.base_genome(k)["ints"] if game["opponent"] == "GA base" else core.bot(game["opponent"], k)
    seats = [seat, core.Seat(core.GA, og)]
    labels = [label, game["opponent"]]
    r = core.play(seats, k, game["seed"], rng_seed=game["seed"] + 1, trace=True)
    tr = r["trace"]
    frames, last = [], [None, None]
    for t in tr:
        me, coins, choice = int(t[0]), int(t[2]), int(t[3])
        what = "nothing" if choice == core.PASS else NAME[choice]
        line2 = ""
        pi = t[7 + 2 * N:]
        if me == 0 and label != "GA base" and pi.max() < 0.999:
            top = np.argsort(-pi)[:3]
            who = "search" if label == "AZ search" else "net"
            line2 = f"{who}: " + " · ".join(f"{'nothing' if a == core.PASS else NAME[a]} {pi[a] * 100:.0f}%"
                                            for a in top if pi[a] > 0.005)
            if label == "RL net":
                line2 += f"   win estimate {(float(t[6]) + 1) / 2 * 100:.0f}%"
        last[me] = (f"${coins}: buys {what}", line2)
        frames.append((AZH.frame(t, labels, k, last), 11))
    res = core.outcome(r, 0)
    word = "wins" if res > 0 else "loses" if res < 0 else "shares the win"
    sc = r["scores"]
    fin = f"{label} {word} {sc[0]}–{sc[1]} in {max(r['turns'])} turns"
    frames.append((AZH.frame(tr[-1], labels, k, last, final=fin), 75))
    AZH.encode_mp4(frames, out_mp4)
    return {"ok": res > 0, "result": "win" if res > 0 else "loss" if res < 0 else "shared", "detail": fin + "."}


# ── The model tab ───────────────────────────────────────────────────────────────────────────

def cached(out_png, source_path):
    """True when out_png is newer than the model file it is drawn from (so the figure need not be redrawn)."""
    return os.path.exists(out_png) and os.path.exists(source_path) and os.path.getmtime(out_png) >= os.path.getmtime(source_path)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'xtick.labelsize': 11,
                         'ytick.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


# Figures are shown one per row, up to ~900 px wide: draw them 9 inches wide at 120 dpi (1080 px), fonts >= 11 pt.
def fig_arch(layers, out_png, title='Architecture'):
    """layers: [("input", 8), ("dense tanh", 64), ("dense tanh", 64), ("action logits", 4)] -> a box diagram with sizes."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.9 * len(layers)))
    ax.axis('off')
    for i, (name, n) in enumerate(layers):
        y = len(layers) - 1 - i
        w = 2.2 + 5 * (n / max(m for _, m in layers)) ** 0.5
        ax.add_patch(plt.Rectangle((5 - w / 2, y - 0.32), w, 0.64, fc='#dbe4ff', ec='#3b5bdb', lw=1.5))
        ax.text(5, y, f'{name}  ·  {n:,}', ha='center', va='center', fontsize=13)
        if i:
            ax.annotate('', (5, y + 0.34), (5, y + 0.66), arrowprops=dict(arrowstyle='->', color='#6b6b70', lw=1.5))
    ax.set_xlim(0, 10); ax.set_ylim(-0.6, len(layers) - 0.4)
    ax.set_title(title, fontsize=14)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def fig_weight_hist(named_arrays, out_png, title='Weights per layer'):
    """named_arrays: [("layer 1", ndarray), ...] -> one histogram per layer, two per row."""
    plt = _plt()
    import numpy as np
    n = len(named_arrays)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(9, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, a) in zip(axes, named_arrays):
        a = np.asarray(a).ravel()
        lim = float(np.percentile(np.abs(a), 99.5)) or 1
        ax.hist(np.clip(a, -lim, lim), bins=50, color='#3b5bdb')
        ax.set_title(f'{name}: {a.size:,} weights, std {a.std():.3f}')
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(title, fontsize=14); fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def fig_bars(labels, values, out_png, title='', ylabel='', colors=None, fmt='{:.2f}'):
    """Horizontal bars, first label on top, value written at the end of each bar: e.g. action probabilities in a fixed
    situation, or input sensitivity (how much the output moves when each input is nudged by one standard deviation).
    ylabel names the value axis."""
    plt = _plt()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.36 * n))
    ys = list(range(n))[::-1]
    ax.barh(ys, values, color=colors or '#3b5bdb', height=0.7)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    top = max(values) if len(values) else 1
    for y, v in zip(ys, values):
        ax.text(v + top * 0.01, y, fmt.format(v), va='center', fontsize=10, color='#444')
    ax.set_xlim(0, top * 1.12 if top > 0 else 1)
    ax.set_xlabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


KIND_SHOW = ["buy", "play action", "Militia discard", "Workshop gain", "Chapel trash", "Remodel gain"]


def model_viz(out_dir):
    """Architecture, weights, what it chooses per decision kind (vs the GA genome) and input
    sensitivity, for the newest net of RUN. Redrawn only when that net changes."""
    import torch
    from rl.net import load
    nets = _nets()
    if not nets:
        return []
    src = os.path.join(ROOT, nets[-1])
    pt = src.replace(".npy", ".pt")
    it = nets[-1].split("_")[-1][:-4]
    figs = [("arch", "Architecture", f"{RUN} net {it}: layer sizes read from the saved weights. One trunk, two heads: "
                                    "34 logits (a card, or pass) for whatever decision is asked, and a win estimate."),
            ("weights", "Weights per layer", "A wide, centred histogram means the layer is used; a spike at 0 means "
                                             "dead units."),
            ("agreement", "How often it does what the GA genome would do", "On positions from games against the GA "
             "base: the share of decisions, per kind, where the network's top choice equals the genome's. rl-v1 began by "
             "imitating the genome; low agreement shows where training has moved it away."),
            ("buys", "What it buys as training goes on", "For snapshots along the run: what the network buys "
             "(top choice) in 40 games against itself, as a share of all its buys. "),
            ("sensitivity", "What it looks at", "How much its choices move (total variation, averaged over all recorded "
                                                "decisions) when one input is nudged by one standard deviation. The 25 "
                                                "most influential of the 413 inputs.")]
    paths = {k: os.path.join(out_dir, f"{k}.png") for k, _, _ in figs}
    entries = [{"src": f"model/{k}.png", "title": t, "caption": c} for k, t, c in figs]
    try:
        fig_buys(paths["buys"])
    except Exception as ex:                       # a figure must never stop the publisher
        print("buys figure failed:", ex)
    if all(cached(paths[k], src) for k in paths if k != "buys"):
        return entries
    net = load(pt)
    sd = net.state_dict()
    fig_arch([("input: what it knows at a decision", net.nf), ("dense ReLU", net.nh), ("dense ReLU", net.nh),
                  ("34 logits + 1 win estimate", core.NA + 1)], paths["arch"], title=f"Net {it}")
    fig_weight_hist([("W1", sd["l1.weight"].numpy()), ("W2", sd["l2.weight"].numpy()),
                         ("policy", sd["pi.weight"].numpy()), ("value", sd["v.weight"].numpy())], paths["weights"])
    # positions: the GA base genome's decisions (recorded) in the Play games and more
    import random
    rng = random.Random("rl-v2 model viz")
    rows = []
    for i in range(60):
        k = core.random_kingdom(rng)
        g = core.base_genome(k)["ints"]
        r = core.play([core.Seat(core.GA, g, record=True), core.Seat(core.GA, g, record=True)], k, rng.getrandbits(32))
        rows.append(r["rec"])
    X = np.concatenate([r["X"] for r in rows])
    L = np.concatenate([r["LEGAL"] for r in rows])
    A = np.concatenate([r["A"] for r in rows])
    K = np.concatenate([r["KIND"] for r in rows])
    with torch.no_grad():
        lg, _ = net(torch.from_numpy(X), torch.from_numpy(L).bool())
        prob = torch.softmax(lg, -1).numpy()
    pick = prob.argmax(1)
    names, vals = [], []
    for kname in core.KINDS:
        m = K == core.KINDS.index(kname)
        if m.sum() >= 20:
            names.append(f"{kname} ({m.sum()})")
            vals.append(float((pick[m] == A[m]).mean()))
    fig_bars(names, vals, paths["agreement"], title="Top choice = the GA genome's choice, per kind of decision", ylabel="share of decisions (number of decisions in brackets)", fmt="{:.0%}")
    std = X.std(0)
    eff = np.zeros(net.nf)
    with torch.no_grad():
        for j in range(net.nf):
            if std[j] == 0:
                continue
            X2 = X.copy()
            X2[:, j] += std[j]
            lg2, _ = net(torch.from_numpy(X2), torch.from_numpy(L).bool())
            eff[j] = 0.5 * np.abs(torch.softmax(lg2, -1).numpy() - prob).sum(1).mean()
    fn = feature_names()
    top = np.argsort(-eff)[:25]
    fig_bars([fn[j] for j in top], [eff[j] for j in top], paths["sensitivity"],
                 title="How much its choices move when one input rises by 1 std", ylabel="total variation (0 = no change, 1 = a different choice)", fmt="{:.3f}")
    return entries


BUY_GROUPS = [("nothing", "#bbbbbb"), ("Curse", "#7b3fa0"), ("Copper", "#b87333"), ("Estate", "#8fbf6f"),
              ("Silver", "#9aa5b1"), ("Gold", "#e0b000"), ("Duchy", "#4f9a3c"), ("Province", "#1f6a2a"),
              ("action cards", "#3b6fd8")]


def _buy_shares(path, games=40):
    """Shares of the net's buys (top choice, self-play) per group, cached per net file in runs/<run>/buys.json."""
    import random
    rng = random.Random("rl-v2 buys")
    cnt = {g: 0 for g, _ in BUY_GROUPS}
    W = np.load(os.path.join(ROOT, path))
    for _ in range(games):
        k = core.random_kingdom(rng)
        g = core.base_genome(k)["ints"]
        r = core.play([core.Seat(core.NET, g, W, record=True), core.Seat(core.NET, g, W, record=True)], k,
                      rng.getrandbits(32))
        rec = r["rec"]
        for a in rec["A"][rec["KIND"] == core.KINDS.index("buy")]:
            a = int(a)
            name = "nothing" if a == core.PASS else NAME[a]
            cnt[name if name in cnt else "action cards"] += 1
    tot = max(1, sum(cnt.values()))
    return {g: cnt[g] / tot for g in cnt}


def fig_buys(out_png):
    nets = [p for p in _nets() if int(p.split("_")[-1][:-4]) % EVERY == 0 or p.endswith("_10.npy")
            or p.endswith("_5.npy")]
    if not nets:
        return
    cache_p = os.path.join(ROOT, "runs", RUN, "buys.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    changed = False
    for p in nets:
        if p not in cache:
            cache[p] = _buy_shares(p)
            changed = True
    if changed or not os.path.exists(out_png):
        json.dump(cache, open(cache_p, "w"))
        keep = sorted((p for p in cache if os.path.exists(os.path.join(ROOT, p))),
                      key=lambda p: int(p.split("_")[-1][:-4]))
        if len(keep) > 14:                         # at most 14 rows: first, last, evenly spaced between
            idx = np.linspace(0, len(keep) - 1, 14).round().astype(int)
            keep = [keep[i] for i in sorted(set(idx))]
        plt = _plt()
        fig, ax = plt.subplots(figsize=(9, 1.2 + 0.42 * len(keep)))
        labels = [f"iteration {int(p.split('_')[-1][:-4])}" for p in keep]
        left = np.zeros(len(keep))
        for g, col in BUY_GROUPS:
            v = np.array([cache[p][g] for p in keep])
            ax.barh(labels, v, left=left, color=col, label=g, edgecolor="white", linewidth=0.5)
            for i, (x, w) in enumerate(zip(left, v)):
                if w >= 0.07:
                    ax.text(x + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=10,
                            color="white" if g not in ("nothing", "Estate", "Silver", "Gold") else "black")
            left += v
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of all its buys in 40 self-play games")
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        fig.suptitle("What it buys, iteration by iteration", y=0.995)
        ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10)
        fig.set_size_inches(9, 2.2 + 0.42 * len(keep))
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)


def feature_names():
    names = AZH.feature_names()
    names += [f"hand: {NAME[c]}" for c in range(N)] + [f"in play: {NAME[c]}" for c in range(N)]
    names += ["actions left", "my turn"] + [f"kind: {k}" for k in core.KINDS] + [f"about: {NAME[c]}" for c in range(N)]
    names += ["next hand: expected coins", "next hand: chance of $3+", "next hand: chance of $5+",
              "next hand: chance of $6+", "next hand: chance of $8+", "next hand: expected action cards",
              "next hand: expected +Actions", "next hand: expected +Buys", "next hand: expected +Cards",
              "next turn: expected VP gain", "next turn: expected VP"]
    names += [f"deck per 5 cards: {w}" for w in ("coins", "action cards", "+Actions", "+Buys", "+Cards")]
    names += [f"if I gain {NAME[c]}: {w}" for w in ("coins per 5", "+Actions per 5", "+Buys per 5") for c in range(N)]
    assert len(names) == core.NF
    return names


BASE_SHOW = ["Province", "Gold", "Silver", "Duchy", "Estate", "Copper", "Curse"]


def favorites(data_dir, games=300):
    """What the newest net buys right now, from real games: it (argmax) against the GA base genome on `games` fresh
    kingdoms, both seats. Per kingdom card: how often it buys one when the card is on the table, and its score then;
    the pairs it builds together most; its openings. Written to data/favorites.json, redrawn when the net changes."""
    import itertools
    import random
    nets = [p for p in _nets() if int(p.split("_")[-1][:-4]) % EVERY == 0]
    if not nets:
        return
    src = os.path.join(ROOT, nets[-1])
    out_p = os.path.join(data_dir, "favorites.json")
    if cached(out_p, src) and json.load(open(out_p)).get("net") == nets[-1]:
        return
    W = np.load(src)
    rng = random.Random("rl-v2 favorites")
    buy = core.KINDS.index("buy")
    rows = []
    for _ in range(games):
        k, seed = core.random_kingdom(rng), rng.getrandbits(32)
        g = core.base_genome(k)["ints"]
        for seat in (0, 1):
            me = core.Seat(core.NET, g, W, record=True)
            seats = [me, core.Seat(core.GA, g)] if seat == 0 else [core.Seat(core.GA, g), me]
            r = core.play(seats, k, seed, rng_seed=seed ^ 0xFA7)
            rec = r["rec"]
            m = (rec["KIND"] == buy) & (rec["A"] != core.PASS)
            bought = [NAME[int(a)] for a in rec["A"][m]]
            turns = rec["TURN"][m]
            opening = sorted(NAME[int(a)] for a, t in zip(rec["A"][m], turns) if t <= 2)
            rows.append({"kingdom": list(k), "bought": bought, "points": (core.outcome(r, seat) + 1) / 2,
                         "opening": " / ".join(opening) if opening else "nothing"})
    score = float(np.mean([r["points"] for r in rows]))
    cards = []
    for c in sorted({c for r in rows for c in r["kingdom"]}):
        on = [r for r in rows if c in r["kingdom"]]
        took = [r for r in on if c in r["bought"]]
        left = [r for r in on if c not in r["bought"]]
        cards.append({"card": c, "games": len(on), "take": len(took) / len(on),
                      "copies": float(np.mean([r["bought"].count(c) for r in took])) if took else 0.0,
                      "score_took": float(np.mean([r["points"] for r in took])) if took else None,
                      "score_left": float(np.mean([r["points"] for r in left])) if left else None})
    cards.sort(key=lambda x: -x["take"])
    take = {x["card"]: x["take"] for x in cards}
    base = [{"card": c, "per_game": float(np.mean([r["bought"].count(c) for r in rows]))} for c in BASE_SHOW]
    combos = []
    for a, b in itertools.combinations(sorted(take), 2):
        on = [r for r in rows if a in r["kingdom"] and b in r["kingdom"]]
        if len(on) < 16:
            continue
        both = [r for r in on if a in r["bought"] and b in r["bought"]]
        if len(both) < 6:
            continue
        p = len(both) / len(on)
        combos.append({"cards": [a, b], "games": len(on), "both": p, "lift": p / max(1e-9, take[a] * take[b]),
                       "score": float(np.mean([r["points"] for r in both])),
                       "copies": [float(np.mean([r["bought"].count(a) for r in both])),
                                  float(np.mean([r["bought"].count(b) for r in both]))]})
    combos.sort(key=lambda x: -(x["both"] * min(x["lift"], 3)))
    openings = {}
    for r in rows:
        openings[r["opening"]] = openings.get(r["opening"], 0) + 1
    op = sorted(openings.items(), key=lambda kv: -kv[1])[:8]
    it = int(nets[-1].split("_")[-1][:-4])
    turns = None
    try:
        turns = turn_usage_table(W)
    except Exception as ex:
        print("turn usage failed:", ex)
    json.dump({"net": nets[-1], "iter": it, "games": len(rows), "score": score, "cards": cards, "base": base,
               "combos": combos[:12], "openings": [{"opening": o, "share": n / len(rows)} for o, n in op],
               "turns": turns}, open(out_p, "w"), separators=(",", ":"))


def turn_usage_table(W, n_kingdoms=150):
    """How it uses +Actions, +Cards and +Buys per turn (experiments/turn_usage.py), next to the GA genome and rl-v1
    on the same games (those two are cached: they never change)."""
    import random
    spec = importlib.util.spec_from_file_location("turn_usage", os.path.join(ROOT, "experiments", "turn_usage.py"))
    tu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tu)
    rng = random.Random("rl-v2 turn usage site")
    want = {"Village", "Festival", "Market", "Laboratory", "Council Room"}
    games = []
    while len(games) < 2 * n_kingdoms:
        k = core.random_kingdom(rng)
        if want & set(k):
            sd = rng.getrandbits(32)
            games += [(k, sd, 0), (k, sd, 1)]
    cache_p = os.path.join(ROOT, "runs", RUN, "turn_usage_ref.json")
    ref = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    if "GA genome" not in ref:
        ref = {"GA genome": tu.usage(None, games),
               "rl-v1": tu.usage(np.load(os.path.join(ROOT, "opponents", "rl_r2_2200.npy")), games)}
        json.dump(ref, open(cache_p, "w"))
    return {"this net": tu.usage(W, games), **ref}


def playvs_files(docs_dir):
    """The Play-against-it page's files: the Python engine, play/rlplay.py + session.py, and the networks."""
    import shutil
    out = os.path.join(docs_dir, "playvs")
    os.makedirs(os.path.join(out, "py", "dominion_engine"), exist_ok=True)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    for f in ("__init__.py", "cards.py", "rules.py", "rng.py", "game.py", "bots.py"):
        shutil.copyfile(os.path.join(ROOT, "engine", "dominion_engine", f), os.path.join(out, "py", "dominion_engine", f))
    for f in ("rlplay.py", "session.py"):
        shutil.copyfile(os.path.join(ROOT, "play", f), os.path.join(out, "py", f))
    shutil.copyfile(os.path.join(ROOT, "play", "spec.json"), os.path.join(out, "spec.json"))   # the game page's settings
    nets = _nets()
    its = {int(p.split("_")[-1][:-4]): p for p in nets}
    newest = max((i for i in its if i % 100 == 0), default=None)   # moves every 100 iterations (files are committed)
    pick = []
    if newest:
        pick.append((f"rl-v3 (iteration {newest})", its[newest], "The newest rl-v3 network: rl-v1's champion trained on, with the helper inputs and the card school. The strongest."))
    pick.append(("rl-v1 champion", os.path.join("opponents", "rl_r2_2200.npy"),
                 "rl-v1 r2 at iteration 2,200: started as a copy of the GA genome, then PPO. 82.8% against the GA base. Hard."))
    pick.append(("rl-v2 from zero", os.path.join("opponents", "rl_v2_z1h_1100.npy"),
                 "Learned from random weights by self-play only: money, Witch and Militia; few engines. Medium."))
    models, keep = [], set()
    for label, path, note in pick:
        f = os.path.basename(path) if path.startswith("opponents") else f"{path.split(os.sep)[1]}_{os.path.basename(path)}"
        dst = os.path.join(out, "models", f)
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(os.path.join(ROOT, path)):
            shutil.copyfile(os.path.join(ROOT, path), dst)
        keep.add(f)
        models.append({"label": label, "name": label.split(" (")[0] if "rl-v1" in label else label, "file": f, "note": note})
    models.append({"label": "Big Money (a simple bot)", "name": "Big Money", "file": None,
                   "note": "Buys Province at $8, Gold at $6, Silver at $3, nothing else. A good first opponent."})
    for f in os.listdir(os.path.join(out, "models")):
        if f not in keep:
            os.remove(os.path.join(out, "models", f))
    json.dump(models, open(os.path.join(out, "models.json"), "w"), indent=1)


def audits():
    """Chain exam and card audit for rl-v3's newest net at every 100 iterations (each ~10 s), once."""
    import subprocess
    import sys
    nets = [p for p in _nets() if int(p.split("_")[-1][:-4]) % 100 == 0 and not p.endswith("net_0.npy")]
    if not nets:
        return
    p = nets[-1]
    it = int(p.split("_")[-1][:-4])
    name = f"{RUN}_{it}"
    ex = os.path.join(ROOT, "results", "chain_exam.json")
    have = json.load(open(ex))["models"] if os.path.exists(ex) else {}
    if f"rl-v3 (iteration {it})" not in have:
        subprocess.run([sys.executable, "experiments/chain_exam.py", "score", f"rl-v3 (iteration {it})={p}"],
                       cwd=ROOT, check=False)
    if not os.path.exists(os.path.join(ROOT, "results", f"card_audit_{name}.json")):
        subprocess.run([sys.executable, "experiments/card_audit.py", p, "--name", name, "--workers", "2"],
                       cwd=ROOT, check=False)


def extra(data_dir):
    try:
        audits()
    except Exception as ex:                       # never stops the publisher
        print("audits failed:", ex)
    try:
        favorites(data_dir)
    except Exception as ex:                       # never stops the publisher
        print("favorites failed:", ex)
    try:
        playvs_files(os.path.dirname(data_dir))
    except Exception as ex:
        print("playvs failed:", ex)
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        out[os.path.basename(p)[:-5]] = json.load(open(p))
    json.dump(out, open(os.path.join(data_dir, "results.json"), "w"), separators=(",", ":"))
