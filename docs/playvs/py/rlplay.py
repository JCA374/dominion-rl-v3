"""The rl-v2 network as a Python agent for the Python reference engine: c/rl.c's 298 features and forward pass in
numpy, with no C. Runs in the browser (Pyodide) for the Play-against-it page; tests/test_play_parity.py checks
that the features equal the C ones on every decision of real games, so this agent plays exactly like the C net
(argmax).
"""
from __future__ import annotations

import numpy as np

from dominion_engine import rules
from dominion_engine.cards import CARDS, DUCHY, ESTATE, NUM_CARDS, PROVINCE
from dominion_engine.game import NUM_KINDS

N = NUM_CARDS
AZ_NF = 5 * N + 10
NF0 = AZ_NF + 2 * N + 2 + NUM_KINDS + N         # 298: rl-v1's inputs
NH_NEXT, NH_DECK = 11, 5
NF = NF0 + NH_NEXT + NH_DECK + 3 * N            # 413: + the helper inputs (c/rl.c rl_helpers)
NA = N + 1                                       # a card id (or Sentry code), then PASS
PASS = N


# c/rl.c is built with -ffast-math, which turns x / 10.0f into x * (1 / 10.0f): the same here, to the bit.
R2, R3, R8, R10, R20 = (np.float32(1) / np.float32(d) for d in (2, 3, 8, 10, 20))


def _counts(cards):
    c = np.zeros(N, np.float32)
    for x in cards:
        c[x] += 1
    return c


def _choose(n, k):
    if k < 0 or k > n:
        return 0.0
    r = 1.0
    for i in range(k):
        r = r * (n - i) / (i + 1)
    return r


HK = 16                     # c/rl.c: most cards drawn, and coin sums 0..HS-1
HS = 3 * HK + 1


def _draw_dist(cnt, k):
    """Distribution of the coins of k cards drawn from counts `cnt` (c/rl.c draw_dist, same order of operations)."""
    k = min(k, HK - 1)
    nv, n = [0, 0, 0, 0], 0
    for c in range(N):
        nv[CARDS[c].coins] += cnt[c]
        n += cnt[c]
    f = [[0.0] * HS for _ in range(HK)]
    f[0][0] = 1.0
    for v in range(4):
        g = [[0.0] * HS for _ in range(HK)]
        for j in range(k + 1):
            for s in range(HS):
                if f[j][s] == 0:
                    continue
                t = 0
                while t <= nv[v] and j + t <= k:
                    g[j + t][s + v * t] += f[j][s] * _choose(nv[v], t)
                    t += 1
        f = g
    tot = _choose(n, k)
    return [f[k][s] / tot if tot > 0 else float(s == 0) for s in range(HS)]


def helpers(g, me):
    """c/rl.c rl_helpers: the next hand's expected coins, chances of $3/$5/$6/$8, expected actions, +Actions, +Buys,
    +Cards, VP gain; the deck per 5 cards; what one more copy of each card changes (coins, +Actions, +Buys)."""
    p = g.players[me]
    h = np.zeros(NF - NF0, np.float32)
    deck, pool, own = [0] * N, [0] * N, [0] * N
    np_ = 0
    if g.current != me:
        for c in p.hand:
            deck[c] += 1
        nd = len(p.hand)
        a, b = nd, 0
    else:
        for c in p.deck:
            deck[c] += 1
        for z in (p.discard, p.hand, p.play):
            for c in z:
                pool[c] += 1
        nd, np_ = len(p.deck), len(p.discard) + len(p.hand) + len(p.play)
        a = min(nd, 5)
        b = min(5 - a, np_)
    Pa, Pb = _draw_dist(deck, a), _draw_dist(pool, b)
    P = [0.0] * HS
    for i in range(HS):
        for j in range(HS - i):
            P[i + j] += Pa[i] * Pb[j]
    ec = p3 = p5 = p6 = p8 = evp = 0.0
    prov, duchy, est = g.supply[PROVINCE] > 0, g.supply[DUCHY] > 0, g.supply[ESTATE] > 0
    for s in range(HS):
        ec += s * P[s]
        if s >= 3:
            p3 += P[s]
        if s >= 5:
            p5 += P[s]
        if s >= 6:
            p6 += P[s]
        if s >= 8:
            p8 += P[s]
        v = 6 if s >= 8 and prov else 3 if s >= 5 and duchy else 1 if s >= 2 and est else 0
        evp += v * P[s]
    ea = eact = ebuy = ecard = 0.0
    for c in range(N):
        w = (a * deck[c] / nd if nd else 0.0) + (b * pool[c] / np_ if np_ else 0.0)
        if CARDS[c].is_action:
            ea += w
        eact += w * CARDS[c].actions
        ebuy += w * CARDS[c].buys
        ecard += w * CARDS[c].cards
    vp = rules.score(p.all_cards())
    h[:NH_NEXT] = [ec / 8, p3, p5, p6, p8, ea / 3, eact / 3, ebuy / 2, ecard / 5, evp / 6, (vp + evp) / 20]
    for c in p.all_cards():
        own[c] += 1
    n = sum(own)
    S = [0.0] * 5
    for c in range(N):
        S[0] += own[c] * CARDS[c].coins
        S[1] += own[c] * int(CARDS[c].is_action)
        S[2] += own[c] * CARDS[c].actions
        S[3] += own[c] * CARDS[c].buys
        S[4] += own[c] * CARDS[c].cards
    scale = (8, 3, 3, 2, 5)
    for k in range(5):
        h[NH_NEXT + k] = 5 * S[k] / n / scale[k] if n else 0
    o = NH_NEXT + NH_DECK
    for c in range(N):
        if g.supply[c] < 0 or not n:
            continue
        x = (CARDS[c].coins, CARDS[c].actions, CARDS[c].buys)
        base = (S[0], S[2], S[3])
        for k in range(3):
            h[o + k * N + c] = 2 * 5 * ((base[k] + x[k]) / (n + 1) - base[k] / n)
    return h


def features(g, me: int, kind: int, subject: int, helpers_too: bool = True) -> np.ndarray:
    p, o = g.players[me], g.players[1 - me]
    x = np.zeros(NF, np.float32)
    sup = np.array(g.supply, np.float32)
    x[0:N] = _counts(p.all_cards()) * R10
    x[N:2 * N] = _counts(p.deck) * R10
    x[2 * N:3 * N] = _counts(o.all_cards()) * R10
    x[3 * N:4 * N] = np.where(sup > 0, sup * R10, 0)
    x[4 * N:5 * N] = sup >= 0
    e = 5 * N
    vp, ovp = rules.score(p.all_cards()), rules.score(o.all_cards())
    x[e + 0] = g.coins * R8
    x[e + 1] = g.buys * R2
    x[e + 2] = p.turns * R20
    x[e + 3] = o.turns * R20
    x[e + 4] = vp * R20
    x[e + 5] = ovp * R20
    x[e + 6] = (vp - ovp) * R10
    x[e + 7] = max(g.supply[PROVINCE], 0) * R8
    x[e + 8] = rules.empty_piles(g.supply) * R3
    x[e + 9] = me == 0
    f = AZ_NF
    for c in p.hand:
        x[f + c] += np.float32(0.2)
    for c in p.play:
        x[f + N + c] += np.float32(0.2)
    x[f + 2 * N] = g.actions * R3 if g.current == me else 0
    x[f + 2 * N + 1] = g.current == me
    x[f + 2 * N + 2 + kind] = 1
    if 0 <= subject < N:
        x[f + 2 * N + 2 + NUM_KINDS + subject] = 1
    if helpers_too:
        x[NF0:] = helpers(g, me)
    return x


class Net:
    """Flat float32 weights in c/rl.c's layout: W1 b1 W2 b2 Wp bp Wv bv."""

    def __init__(self, flat: np.ndarray):
        for nf in (NF, NF0):                      # with or without the helper inputs
            b = nf + NA + 4
            nh = int(round((-b + (b * b + 4 * (flat.size - NA - 1)) ** 0.5) / 2))
            if nh * nf + nh + nh * nh + nh + NA * nh + NA + nh + 1 == flat.size:
                break
        else:
            raise ValueError(flat.size)
        self.nf = nf
        i = 0

        def take(n, shape):
            nonlocal i
            a = flat[i:i + n].reshape(shape)
            i += n
            return a
        self.w1, self.b1 = take(nh * nf, (nh, nf)), take(nh, (nh,))
        self.w2, self.b2 = take(nh * nh, (nh, nh)), take(nh, (nh,))
        self.wp, self.bp = take(NA * nh, (NA, nh)), take(NA, (NA,))
        self.wv, self.bv = take(nh, (nh,)), take(1, (1,))
        assert i == flat.size
        self.nh = nh

    def forward(self, x):
        h = np.maximum(self.w1 @ x[:self.nf] + self.b1, 0)
        h = np.maximum(self.w2 @ h + self.b2, 0)
        return self.wp @ h + self.bp, float(np.tanh(self.wv @ h + self.bv[0]))


class NetAgent:
    """Argmax over the legal answers, as the C NET seat in every evaluation. `last` keeps the probabilities of the
    last decision (for the page: what the network considered)."""

    def __init__(self, flat):
        self.net = Net(np.asarray(flat, np.float32))
        self.last = None

    def decide(self, game, d) -> int:
        opts = list(d.options) + ([PASS] if d.may_pass else [])
        if len(opts) == 1:
            return d.options[0] if d.options else -1
        lg, v = self.net.forward(features(game, d.player, d.kind, d.subject, self.net.nf > NF0))
        legal = np.array(opts)
        z = lg[legal] - lg[legal].max()
        pr = np.exp(z) / np.exp(z).sum()
        a = int(legal[int(np.argmax(lg[legal]))])
        self.last = {"options": [int(o) for o in legal], "probs": [float(q) for q in pr], "value": v, "kind": d.kind}
        return -1 if a == PASS else a
