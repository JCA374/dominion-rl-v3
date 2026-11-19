"""The C side through ctypes: engine/c/engine.c + c/rl.c (which includes az-v1's az.c and, through
it, dominion-v1's ga_agent.c) built into rl.so.

Seats: GA (a dominion-v1 genome plays every decision), AZ (az-v1's agent: GA genome for the
non-buy decisions, its net + IS-MCTS for the buys) and NET (this project's network plays every
decision). The GA genomes, the bots and the per-kingdom base genome come from az-v1's Python
package (submodule az-v1/), so the comparisons are exactly az-v1's.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AZ = os.path.join(ROOT, "az-v1")
ENGINE_C = os.path.join(ROOT, "engine", "c")
if AZ not in sys.path:
    sys.path.insert(0, AZ)

from az.base import base_genome, kingdom_for_bot, random_kingdom  # noqa: E402,F401
from az.core import bot  # noqa: E402,F401
from dominion_engine.cards import ID, NAME, NUM_CARDS  # noqa: E402,F401

SO = os.path.join(ROOT, "rl.so")
SOURCES = [os.path.join(ENGINE_C, "engine.c"), os.path.join(ROOT, "c", "rl.c")]
DEPS = SOURCES + [os.path.join(ENGINE_C, "engine.h"), os.path.join(AZ, "c", "az.c"),
                  os.path.join(AZ, "dominion-v1", "c", "ga_agent.c")]


def build() -> str:
    if not os.path.exists(SO) or max(os.path.getmtime(d) for d in DEPS) > os.path.getmtime(SO):
        tmp = f"{SO}.{os.getpid()}"
        subprocess.check_call(["gcc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC",
                               "-Wall", "-Wextra", "-Wno-unused-parameter", "-Wno-unused-function",
                               "-I", ENGINE_C, "-o", tmp, *SOURCES, "-lm"])
        os.replace(tmp, SO)
    return SO


lib = ctypes.CDLL(build())
NF = lib.rl_num_features()              # 413: rl-v1's 298 inputs + the helper inputs (rl_helpers in c/rl.c)
NF0 = lib.rl_num_features_base()         # 298: what a net without the helper inputs reads
NA = lib.rl_num_actions()
PASS = NA - 1
_I, _F = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
_U8 = ctypes.POINTER(ctypes.c_ubyte)
lib.rl_num_params.argtypes = [ctypes.c_int, ctypes.c_int]
lib.rl_forward.restype = ctypes.c_float
lib.rl_forward.argtypes = [_F, ctypes.c_int, ctypes.c_int, _F, _F]
lib.rl_play.restype = ctypes.c_int
lib.rl_play.argtypes = [_I, _F, _F, _I, _I, _I, _I, _I, _I, ctypes.c_float, _I, ctypes.c_float,
                        _I, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
                        _F, _U8, _I, _F, _F, _I, _I, _I, ctypes.c_int, _I]
lib.az_set_prior.argtypes = [ctypes.c_float, ctypes.c_float]
lib.az_set_rollout.argtypes = [ctypes.c_int]
lib.az_set_trace.argtypes = [_F, ctypes.c_int]
lib.rl_set_plan.argtypes = [ctypes.c_int] * 4
lib.rl_set_plan_buf.argtypes = [_I, _F]
lib.rl_set_planner.argtypes = [_F, ctypes.c_int, ctypes.c_int]
lib.rl_plan_rollouts.restype = ctypes.c_long
lib.rl_set_plan_leaf.argtypes = [ctypes.c_int]
lib.rl_set_force.argtypes = [ctypes.c_int] * 4
lib.rl_set_chain_rule.argtypes = [ctypes.c_int, ctypes.c_int]
TRACE_W = lib.az_trace_width()          # az-v1's clip trace: one row per buy of either player

GA, AZ_SEAT, NET = 0, 1, 2
MAX_REC = 1500
KINDS = ["play action", "buy", "Throne Room", "Chapel trash", "Moneylender trash", "Mine trash", "Mine gain",
         "Remodel trash", "Remodel gain", "Workshop gain", "Artisan gain", "Artisan topdeck", "Cellar discard",
         "Harbinger topdeck", "Vassal play", "Library skip", "Sentry card", "Sentry top", "Poacher discard",
         "Militia discard", "Bureaucrat topdeck", "Bandit trash"]


def _ints(xs):
    return (ctypes.c_int * len(xs))(*xs)


def _fptr(a):
    return None if a is None else a.ctypes.data_as(_F)


class Seat:
    """mode GA (genome plays all), AZ (az-v1 agent; params = its net, sims 0 = net argmax) or NET
    (params = this project's net; sample = draw from the policy, record = keep the decisions)."""

    def __init__(self, mode, genome, params=None, sims=0, sample=False, record=False):
        self.mode, self.genome, self.params = mode, genome, params
        self.sims, self.sample, self.record = sims, sample, record


def net_shape(params) -> tuple[int, int]:
    """(hidden size, inputs read) of a flat RL net: solves nh^2 + (nf + NA + 4) nh + NA + 1 = size for nf in
    (NF, NF0)."""
    for nf in (NF, NF0):
        b = nf + NA + 4
        nh = int(round((-b + (b * b + 4 * (params.size - NA - 1)) ** 0.5) / 2))
        if nh > 0 and lib.rl_num_params(nh, nf) == params.size:
            return nh, nf
    raise ValueError(f"not an RL net: {params.size} parameters")


def nh_of(params) -> int:
    return net_shape(params)[0]


_planner = [None]


def set_planner(params) -> None:
    """The net the turn planner's rollouts and leaves use (None: each seat's own net)."""
    _planner[0] = params
    if params is None:
        lib.rl_set_planner(None, 0, 0)
    else:
        nh, nf = net_shape(params)
        lib.rl_set_planner(_fptr(params), nh, nf)


def play(seats, kingdom, seed, rng_seed=1, nh=256, temp=1.0, az_lam=0.5, trace=False, plan=(0, 0),
         plan_play=(0, 0), force=(None, None)) -> dict:
    """nh is only a fallback: each NET seat's hidden size is read from its weights.
    plan: per seat, K determinisations for the turn planner at action decisions (0 = off); plan_play: per seat,
    1 = play the planner's choice, 0 = only label recorded decisions (rec PLAN, PVAL).
    force: per seat, a card name for the card school (its first buys forced to 2 copies, not recorded) or None."""
    T = np.zeros((2000, TRACE_W), np.float32) if trace else None
    lib.az_set_trace(_fptr(T), 2000 if trace else 0)
    k = _ints([ID[c] for c in kingdom])
    X = np.zeros((MAX_REC, NF), np.float32)
    LEGAL = np.zeros((MAX_REC, NA), np.uint8)
    A = np.zeros(MAX_REC, np.int32)
    LOGP = np.zeros(MAX_REC, np.float32)
    V = np.zeros(MAX_REC, np.float32)
    SEAT = np.zeros(MAX_REC, np.int32)
    TURN = np.zeros(MAX_REC, np.int32)
    KIND = np.zeros(MAX_REC, np.int32)
    PLAN = np.full(MAX_REC, -1, np.int32)
    PVAL = np.full((MAX_REC, NA), np.nan, np.float32)
    lib.rl_set_plan(int(plan[0]), int(plan[1]), int(plan_play[0]), int(plan_play[1]))
    lib.rl_set_plan_buf(PLAN.ctypes.data_as(_I), _fptr(PVAL))
    lib.rl_set_force(ID[force[0]] if force[0] else -1, ID[force[1]] if force[1] else -1, 2, 6)
    out = (ctypes.c_int * 6)()
    p = [s.params for s in seats]
    for s in seats:
        assert s.params is None or s.params.dtype == np.float32
    shapes = [net_shape(s.params) if s.mode == NET and s.params is not None else (nh, NF) for s in seats]
    n = lib.rl_play(_ints([s.mode for s in seats]), _fptr(p[0]), _fptr(p[1]), _ints([a for a, _ in shapes]),
                    _ints([b for _, b in shapes]),
                    seats[0].genome, seats[1].genome, _ints([int(s.sample) for s in seats]),
                    _ints([int(s.record) for s in seats]), temp, _ints([s.sims for s in seats]), az_lam,
                    k, len(kingdom), seed & (2**64 - 1), rng_seed & (2**64 - 1),
                    _fptr(X), LEGAL.ctypes.data_as(_U8), A.ctypes.data_as(_I), _fptr(LOGP), _fptr(V),
                    SEAT.ctypes.data_as(_I), TURN.ctypes.data_as(_I), KIND.ctypes.data_as(_I), MAX_REC, out)
    lib.rl_set_plan(0, 0, 0, 0)
    lib.rl_set_plan_buf(None, None)
    lib.rl_set_force(-1, -1, 2, 6)
    mask = out[0]
    tr = T[:lib.az_trace_count()] if trace else None
    lib.az_set_trace(None, 0)
    return {"trace": tr, "winners": [i for i in range(2) if mask >> i & 1], "scores": [out[1], out[2]],
            "turns": [out[3], out[4]], "capped": bool(out[5] & 1), "error": bool(out[5] & 2),
            "rec": {"X": X[:n], "LEGAL": LEGAL[:n], "A": A[:n], "LOGP": LOGP[:n], "V": V[:n],
                    "SEAT": SEAT[:n], "TURN": TURN[:n], "KIND": KIND[:n],
                    "PLAN": PLAN[:n], "PVAL": PVAL[:n]}}


def outcome(result: dict, seat: int) -> float:
    """+1 win, 0 shared, -1 loss."""
    w = result["winners"]
    return 0.0 if len(w) == 2 else (1.0 if seat in w else -1.0)
