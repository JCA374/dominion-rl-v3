"""Card losses report: which kingdom cards does a model lose against, per opponent, and where is it weaker than
rl-v1's champion on the very same games? Reads results/card_losses/<name>/games.jsonl.gz (experiments/card_losses.py)
for the model and for a reference run with the champion in the model's seat (same kingdoms, seeds, seats, opponents).

A card's effect = its coefficient in a linear model of the game points on the opponent plus the 26 card-in-kingdom
dummies (centred): the change in points when the card is in the kingdom, against the average card. Kingdoms are
random, so these are causal effects of the card being available; buy rates are descriptive only.

    python experiments/card_losses_report.py c1_2363 --ref champion_ref     # -> results/card_losses/c1_2363/report.html
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import rl.core  # noqa: E402,F401  (puts the engine on the path)
from dominion_engine.cards import KINGDOM_CARDS as K  # noqa: E402
from far import style  # noqa: E402
from far.report import CSS, privacy_scan  # noqa: E402

style.apply()
import matplotlib.pyplot as plt  # noqa: E402

BOTS = ("Big Money", "Smithy-BM", "Witch-BM")
B = 200                                   # bootstrap resamples


def load(name):
    rows = [json.loads(l) for l in gzip.open(os.path.join(ROOT, "results", "card_losses", name, "games.jsonl.gz"), "rt")]
    return rows[0]["header"], rows[1:]


def key(r):
    return r["opp"], r["seed"], r["seat"]


def design(L):
    opps = sorted({r["opp"] for r in L})
    X = np.array([[c in r["kingdom"] for c in K] for r in L], float)
    X -= X.mean(0)
    O = np.array([[r["opp"] == o for o in opps] for r in L], float)
    return np.hstack([O, X]), len(opps)


def effects(L, y, rng):
    """Card effects (points) and bootstrap SE: least squares on opponent + card dummies."""
    M, no = design(L)
    b = np.linalg.lstsq(M, y, rcond=None)[0][no:]
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        bs.append(np.linalg.lstsq(M[i], y[i], rcond=None)[0][no:])
    return b, np.std(bs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--ref", default="champion_ref")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    hdr, A = load(a.name)
    _, R = load(a.ref)
    rm = {key(r): r for r in R}
    opps = hdr["opponents"]
    out = os.path.join(ROOT, "results", "card_losses", a.name)

    # win rate per opponent, model and champion
    wr = {o: float(np.mean([r["points"] for r in A if r["opp"] == o])) for o in opps}
    wr_ref = {o: float(np.mean([r["points"] for r in R if r["opp"] == o])) for o in opps}

    # card effects: model, champion (both without the champion-vs-champion games), paired difference, h2h
    noc = [r for r in A if r["opp"] != "champion"]
    noc_ref = [rm[key(r)] for r in noc]
    e_m, s_m = effects(noc, np.array([r["points"] for r in noc]), rng)
    e_r, s_r = effects(noc_ref, np.array([r["points"] for r in noc_ref]), rng)
    e_d, s_d = effects(noc, np.array([r["points"] - rm[key(r)]["points"] for r in noc]), rng)
    h2h = [r for r in A if r["opp"] == "champion"]
    e_h, s_h = effects(h2h, np.array([r["points"] for r in h2h]), rng)

    # per opponent effects (model) for the heatmap
    heat = np.full((len(K), len(opps)), np.nan)
    for j, o in enumerate(opps):
        g = [r for r in A if r["opp"] == o]
        X = np.array([[c in r["kingdom"] for c in K] for r in g], float)
        keep = X.std(0) > 0.05                          # bots: their own card is always in the kingdom
        Xc = X[:, keep] - X[:, keep].mean(0)
        M = np.hstack([np.ones((len(g), 1)), Xc])
        b = np.linalg.lstsq(M, np.array([r["points"] for r in g]), rcond=None)[0][1:]
        heat[np.where(keep)[0], j] = b

    # buy rates (net opponents only): share of games with the card in the kingdom where the player owns >= 1
    def buys(L):
        out_ = {}
        for c in K:
            g = [r for r in L if c in r["kingdom"] and r["opp"] not in BOTS]
            out_[c] = float(np.mean([r["me"].get(c, 0) > 0 for r in g]))
        return out_
    by_m, by_r = buys(A), buys(R)

    order = np.argsort(e_d)                             # weakest vs the champion first
    figs, secs = {}, []

    # 1. win rates
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(opps))
    b1 = ax.bar(x - 0.2, [wr[o] for o in opps], 0.4, color=style.BLUE, label=f"model ({os.path.basename(hdr['net'])})")
    b2 = ax.bar(x + 0.2, [wr_ref[o] for o in opps], 0.4, color=style.INK3, label="rl-v1 champion (reference)")
    style.label_bars(ax, b1, [wr[o] for o in opps], "{:.0%}", fontsize=7.5)
    style.label_bars(ax, b2, [wr_ref[o] for o in opps], "{:.0%}", fontsize=7.5)
    ax.set_xticks(x, opps, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("points per game")
    ax.legend(loc="lower left")
    figs["wins"] = style.png(fig)
    secs.append(("wins", "Who does it lose to?",
                 f"Points per game (win 1, tie 0.5) against each opponent, {len(A) // len(opps):,} games each, both "
                 "seats. Grey: rl-v1's champion on exactly the same games."))

    # 2. heatmap per opponent
    fig, ax = plt.subplots(figsize=(8, 7.5))
    idx = np.argsort(np.nanmean(heat, 1))
    lim = 0.10
    im = ax.imshow(heat[idx] * 100, cmap="RdBu", vmin=-lim * 100, vmax=lim * 100, aspect="auto")
    ax.set_yticks(range(len(K)), [K[i] for i in idx])
    ax.set_xticks(range(len(opps)), opps, rotation=30, ha="right")
    ax.grid(False)
    for i, ki in enumerate(idx):
        for j in range(len(opps)):
            v = heat[ki, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v * 100:+.0f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.06 else style.INK)
    fig.colorbar(im, ax=ax, shrink=0.6, label="points per 100 games")
    figs["heat"] = style.png(fig)
    secs.append(("heat", "Which cards cost it games, per opponent?",
                 "Change in the model's points (per 100 games) when the card is in the kingdom, against the average "
                 "card, per opponent (joint model of all 26 cards). Red = it scores less. Empty: the bot's own card, "
                 "always in its kingdoms. Each cell is about ±3 points (2 SE)."))

    # 3. model vs champion card effects
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(K))
    o2 = np.argsort(e_m)
    ax.errorbar(e_r[o2] * 100, y + 0.15, xerr=2 * s_r[o2] * 100, fmt="o", color=style.INK3, ms=4, label="rl-v1 champion")
    ax.errorbar(e_m[o2] * 100, y - 0.15, xerr=2 * s_m[o2] * 100, fmt="o", color=style.BLUE, ms=4, label="model")
    ax.axvline(0, color=style.INK3, lw=1)
    ax.set_yticks(y, [K[i] for i in o2])
    ax.set_xlabel("points per 100 games with the card in the kingdom (vs the average card)")
    ax.legend(loc="lower right")
    figs["both"] = style.png(fig)
    secs.append(("both", "Is it the card, or the model?",
                 "Card effects against the seven other opponents, model (blue) and the champion (grey) on the same "
                 "games, ±2 SE (bootstrap). Where both dots sit together the card lowers any strong player's edge (the "
                 "game); where blue is left of grey the model handles it worse than its predecessor."))

    # 4. paired difference + h2h
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.errorbar(e_d[order] * 100, y + 0.15, xerr=2 * s_d[order] * 100, fmt="o", color=style.ORANGE, ms=4,
                label="model minus champion, same games (7 opponents)")
    ax.errorbar(e_h[order] * 100, y - 0.15, xerr=2 * s_h[order] * 100, fmt="s", color=style.PURPLE, ms=4,
                label="head to head: model vs champion")
    ax.axvline(0, color=style.INK3, lw=1)
    ax.set_yticks(y, [K[i] for i in order])
    ax.set_xlabel("points per 100 games, relative to the average card")
    ax.legend(loc="lower right")
    figs["pair"] = style.png(fig)
    secs.append(("pair", "Where is it weaker than the champion?",
                 "Orange: the per-game difference between the model and the champion on identical games, explained by "
                 "the kingdom's cards. Purple: the cards in the head-to-head games against the champion. Left = the "
                 "champion does better with this card; right = the model's gain (the card school's cards are there)."))

    # 5. buy rates
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xb = np.arange(len(K))
    ob = np.argsort([by_m[c] for c in K])
    ax.bar(xb - 0.2, [by_m[K[i]] for i in ob], 0.4, color=style.BLUE, label="model")
    ax.bar(xb + 0.2, [by_r[K[i]] for i in ob], 0.4, color=style.INK3, label="rl-v1 champion")
    ax.set_xticks(xb, [K[i] for i in ob], rotation=60, ha="right")
    ax.set_ylabel("buys ≥ 1 copy")
    ax.legend(loc="upper left")
    figs["buys"] = style.png(fig)
    secs.append(("buys", "Which cards does it buy?",
                 "Share of games (net opponents) with the card in the kingdom where the player ends with at least one "
                 "copy. Descriptive: buying follows the game situation."))

    weak = [K[i] for i in order if e_d[i] < -2 * s_d[i] or e_h[i] < -2 * s_h[i]]
    weak = [c for c in weak if e_d[K.index(c)] < 0]
    s = {"title": f"Card losses analysis: {os.path.basename(hdr['net'])}", "net": hdr["net"],
         "games": len(A), "opponents": opps, "win_rate": wr, "win_rate_champion": wr_ref,
         "cards": {c: {"effect": round(float(e_m[i]), 4), "effect_se": round(float(s_m[i]), 4),
                       "champion_effect": round(float(e_r[i]), 4), "champion_effect_se": round(float(s_r[i]), 4),
                       "vs_champion": round(float(e_d[i]), 4), "vs_champion_se": round(float(s_d[i]), 4),
                       "h2h": round(float(e_h[i]), 4), "h2h_se": round(float(s_h[i]), 4),
                       "buys": round(by_m[c], 3), "champion_buys": round(by_r[c], 3),
                       "per_opponent": {o: (None if np.isnan(heat[i, j]) else round(float(heat[i, j]), 4))
                                        for j, o in enumerate(opps)}} for i, c in enumerate(K)},
         "weaker_than_champion": weak}
    json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=1)

    notes = {}
    np_ = os.path.join(out, "notes.json")
    if os.path.exists(np_):
        notes = json.load(open(np_))
    esc = html.escape
    rows = "".join(
        f"<tr><td>{esc(K[i])}</td><td>{e_m[i] * 100:+.1f}</td><td>{e_r[i] * 100:+.1f}</td>"
        f"<td><b>{e_d[i] * 100:+.1f}</b> ±{2 * s_d[i] * 100:.1f}</td><td>{e_h[i] * 100:+.1f} ±{2 * s_h[i] * 100:.1f}</td>"
        f"<td>{by_m[K[i]]:.0%}</td><td>{by_r[K[i]]:.0%}</td></tr>" for i in order)
    table = ("<div class='tablewrap'><table><tr><th>Card</th><th>Model effect</th><th>Champion effect</th>"
             "<th>Model − champion</th><th>Head to head</th><th>Model buys</th><th>Champion buys</th></tr>" + rows +
             "</table></div><p class='legend'>Points per 100 games with the card in the kingdom, against the average "
             "card; ± = 2 SE (bootstrap, 200 resamples). Sorted by model − champion.</p>")
    tiles = [(f"{len(A):,}", "games"), (f"{np.mean([r['points'] for r in A]):.1%}", "points, all opponents"),
             (f"{wr['champion']:.1%}", "vs rl-v1 champion"), (str(len(weak)), "cards weaker than the champion")]
    tiles_html = "".join(f"<div class='tile'><b>{esc(v)}</b><span>{esc(k)}</span></div>" for v, k in tiles)
    findings = "".join(f"<li>{f}</li>" for f in notes.get("findings", []))
    nxt = notes.get("next_step")
    nxt_html = (f"<div class='next'><b>Next step: {esc(nxt['title'])}</b><pre>{esc(nxt['text'])}</pre></div>"
                if nxt else "")
    caps = notes.get("captions", {})
    body = "".join(f"<h2>{esc(t)}</h2><p class='caption'>{caps.get(k, c)}</p><div class='fig'><img alt='{esc(t)}' "
                   f"src='data:image/png;base64,{figs[k]}'></div>" for k, t, c in secs)
    sub = (f"{os.path.basename(hdr['net'])} against {len(opps)} opponents on random kingdoms (bots: kingdoms with their "
           f"card), {len(A) // len(opps):,} games each, both seats, seeds never used in training. Reference: rl-v1's "
           "champion (net 2,200) in the model's seat on the same games.")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(s["title"])}</title><style>{CSS}</style></head><body><main>
<h1>{esc(s['title'])}</h1><p class="sub">{esc(sub)}</p><div class='tiles'>{tiles_html}</div>
{f"<div class='findings'><h2 style='margin-top:12px'>Key findings</h2><ol>{findings}</ol></div>" if findings else ""}
{nxt_html}<h2>Summary</h2>{table}{body}
<p class="foot">Kingdoms are random, so a card's effect is causal for the card being available; why (buying, playing,
or defending) needs the drill's before/after. Numbers come from summary.json next to this page.</p></main></body></html>"""
    open(os.path.join(out, "report.html"), "w").write(page)
    for h in privacy_scan(out):
        print("PRIVACY", h)
    print("weaker than the champion:", weak)
    for i in order:
        print(f"{K[i]:14s} model {e_m[i] * 100:+5.1f}  champ {e_r[i] * 100:+5.1f}  diff {e_d[i] * 100:+5.1f}±{2 * s_d[i] * 100:.1f}"
              f"  h2h {e_h[i] * 100:+5.1f}±{2 * s_h[i] * 100:.1f}  buys {by_m[K[i]]:.0%}/{by_r[K[i]]:.0%}")


if __name__ == "__main__":
    main()
