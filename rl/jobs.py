"""Game jobs for a process pool (one chunk of games per job; rl.so is loaded once per worker).

    bc_chunk     the GA base genome against itself or a bot: its decisions, for behaviour cloning
    ppo_chunk    the learner (sampling) against an opponent from the league: decisions + GAE targets
    eval_chunk   an agent (argmax) against an opponent; scores only
"""
from __future__ import annotations

import numpy as np

from .core import AZ_SEAT, GA, NET, Seat, base_genome, bot, outcome, play

_cache: dict = {}


def _params(path):
    if path is None:
        return None
    if path not in _cache:
        if len(_cache) > 12:
            _cache.clear()
        _cache[path] = np.load(path)
    return _cache[path]


def _seat(spec: dict, kingdom, sample=False, record=False):
    """spec: {"kind": "ga"} | {"kind": "bot", "name"} | {"kind": "net", "path"} | {"kind": "az", "path", "sims"}"""
    g = base_genome(kingdom)["ints"]
    k = spec["kind"]
    if k == "ga":
        return Seat(GA, g, record=record)
    if k == "bot":
        return Seat(GA, bot(spec["name"], kingdom))
    if k == "az":
        return Seat(AZ_SEAT, g, _params(spec["path"]), sims=spec.get("sims", 0))
    return Seat(NET, g, _params(spec["path"]), sample=sample, record=record)


def gae(v: np.ndarray, z: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """One player's decisions in order, value estimates v, final result z (gamma = 1)."""
    n = len(v)
    adv = np.zeros(n, np.float32)
    nxt, last = z, 0.0                       # after the last decision: the result, no bootstrap
    for t in range(n - 1, -1, -1):
        delta = nxt - v[t]
        last = delta + lam * last
        adv[t] = last
        nxt = v[t]
    return adv, adv + v


def _concat(parts, keys):
    return {k: np.concatenate([p[k] for p in parts]) if parts else np.zeros(0) for k in keys}


KEYS = ["X", "LEGAL", "A", "LOGP", "V", "KIND", "TURN", "ADV", "RET", "Z"]


def ppo_chunk(job: dict) -> dict:
    """job: games [{kingdom, seed, seat, opp}], learner path, nh, temp, lam."""
    parts, results = [], []
    for gm in job["games"]:
        k, seat = gm["kingdom"], gm["seat"]
        me = _seat({"kind": "net", "path": job["learner"]}, k, sample=True, record=True)
        self_play = gm["opp"]["kind"] == "net" and gm["opp"]["path"] == job["learner"]
        other = _seat(gm["opp"], k, sample=True, record=self_play)
        seats = [me, other] if seat == 0 else [other, me]
        force = (gm.get("force"), None) if seat == 0 else (None, gm.get("force"))
        r = play(seats, k, gm["seed"], rng_seed=gm["seed"] * 31 + 7, nh=job["nh"], temp=job["temp"], force=force)
        rec = r["rec"]
        for s in ([0, 1] if self_play else [seat]):
            m = rec["SEAT"] == s
            if not m.any():
                continue
            z = outcome(r, s)
            adv, ret = gae(rec["V"][m], z, job["lam"])
            parts.append({"X": rec["X"][m], "LEGAL": rec["LEGAL"][m], "A": rec["A"][m], "LOGP": rec["LOGP"][m],
                          "V": rec["V"][m], "KIND": rec["KIND"][m], "TURN": rec["TURN"][m], "ADV": adv, "RET": ret,
                          "Z": np.full(m.sum(), z, np.float32)})
        results.append({"opp": gm["opp"].get("tag", gm["opp"]["kind"]), "points": (outcome(r, seat) + 1) / 2,
                        "school": gm.get("force"),
                        "turns": max(r["turns"]), "capped": r["capped"]})
    out = _concat(parts, KEYS)
    out["results"] = results
    return out


def bc_chunk(job: dict) -> dict:
    """The base genome (recorded) against itself (both recorded) or a bot."""
    parts = []
    for gm in job["games"]:
        k = gm["kingdom"]
        if gm.get("bot"):
            me = _seat({"kind": "ga"}, k, record=True)
            other = _seat({"kind": "bot", "name": gm["bot"]}, k)
            seats = [me, other] if gm["seat"] == 0 else [other, me]
            rec_seats = [gm["seat"]]
        else:
            seats = [_seat({"kind": "ga"}, k, record=True), _seat({"kind": "ga"}, k, record=True)]
            rec_seats = [0, 1]
        r = play(seats, k, gm["seed"])
        rec = r["rec"]
        for s in rec_seats:
            m = rec["SEAT"] == s
            parts.append({"X": rec["X"][m], "LEGAL": rec["LEGAL"][m], "A": rec["A"][m], "KIND": rec["KIND"][m],
                          "Z": np.full(m.sum(), outcome(r, s), np.float32)})
    return _concat(parts, ["X", "LEGAL", "A", "KIND", "Z"])


def eval_chunk(job: dict) -> list[dict]:
    """agent spec vs opponent spec per game, the agent playing its argmax."""
    out = []
    for gm in job["games"]:
        k, seat = gm["kingdom"], gm["seat"]
        me = _seat(job["agent"], k)
        other = _seat(gm["opp"], k)
        seats = [me, other] if seat == 0 else [other, me]
        r = play(seats, k, gm["seed"], rng_seed=gm["seed"] ^ 0x5EED, nh=job["nh"])
        out.append({"opp": gm["opp"].get("tag", gm["opp"]["kind"]), "points": (outcome(r, seat) + 1) / 2,
                    "margin": r["scores"][seat] - r["scores"][1 - seat], "turns": max(r["turns"]),
                    "capped": r["capped"]})
    return out
