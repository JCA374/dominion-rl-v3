"""PPO correctness fixes from the c2 search-training handoff (2026-11-23):

- the update reads the sampler's temperature (the first PPO ratio is 1, also at temp 0.7);
- errored games and games whose records overflowed give no training records (flagged, counted);
- the training stream is seeded by cfg seed, not the run name, with frozen c2/c1 and kingdom exclusions;
- two runs with the same seed and settings write identical nets (ordered workers, seeded minibatch shuffles).
"""
import json
import os
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402
from rl import jobs  # noqa: E402
from rl.core import GA, NET, Seat, base_genome, play, random_kingdom  # noqa: E402
from rl.net import load  # noqa: E402

C2_PT = os.path.join(ROOT, "runs", "c2", "net_2654.pt")
C2 = os.path.join(ROOT, "opponents", "rl3_c2_2654.npy")


def _sampled(temp, n=6):
    p = np.load(C2)
    rng = random.Random(11)
    parts = []
    for i in range(n):
        k = random_kingdom(rng)
        g = base_genome(k)["ints"]
        r = play([Seat(NET, g, p, sample=True, record=True), Seat(GA, g)], k, 100 + i, rng_seed=7 + i, temp=temp)
        parts.append(r["rec"])
    return {x: np.concatenate([q[x] for q in parts]) for x in ("X", "LEGAL", "A", "LOGP")}


def _c2_net():
    if os.path.exists(C2_PT):
        return load(C2_PT)
    from rl.net import Net                                   # (the run folder is not in git): rebuild from the .npy
    flat = torch.from_numpy(np.load(C2))
    net = Net(256)
    o = 0
    for prm in [net.l1.weight, net.l1.bias, net.l2.weight, net.l2.bias, net.pi.weight, net.pi.bias, net.v.weight,
                net.v.bias]:
        prm.data.copy_(flat[o:o + prm.numel()].reshape(prm.shape))
        o += prm.numel()
    return net


def test_concat_free_equals_concatenate():
    rng = np.random.default_rng(0)
    parts = [{"X": rng.normal(size=(n, 3)).astype(np.float32), "A": np.arange(n)} for n in (4, 0, 7)]
    want = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "A")}
    got = train.concat_free(parts, ["X", "A"])
    assert all(np.array_equal(want[k], got[k]) and want[k].dtype == got[k].dtype for k in want)


def test_net_from_npy_is_exact():
    from rl.net import from_flat
    flat = np.load(C2)
    assert np.array_equal(from_flat(flat).flat(), flat)


def test_first_ratio_matches_sampler_temperature():
    net = _c2_net()
    for temp in (0.7, 1.0):
        d = _sampled(temp)
        r = train.ratio_check(net, d, temp, torch)
        assert r["decisions"] > 300 and r["mean_abs_ratio_err"] < 1e-5 and r["outside_clip"] == 0.0, r
    wrong = train.ratio_check(net, _sampled(0.7), 1.0, torch)       # the old update (unscaled logits) at temp 0.7
    assert wrong["mean_abs_ratio_err"] > 0.01, wrong


def test_overflow_is_reported():
    p = np.load(C2)
    k = random_kingdom(random.Random(2))
    g = base_genome(k)["ints"]
    r = play([Seat(NET, g, p, sample=True, record=True), Seat(GA, g)], k, 5, max_rec=10)
    assert r["overflow"] and len(r["rec"]["A"]) == 10
    r = play([Seat(NET, g, p, sample=True, record=True), Seat(GA, g)], k, 5)
    assert not r["overflow"]


def _chunk(monkeypatch, patch):
    real = jobs.play
    monkeypatch.setattr(jobs, "play", lambda *a, **kw: patch(real(*a, **kw)) if patch else real(*a, **kw))
    k = random_kingdom(random.Random(3))
    gm = [{"kingdom": k, "seed": 9, "seat": 0, "opp": {"kind": "ga"}},
          {"kingdom": k, "seed": 9, "seat": 0, "opp": {"kind": "net", "path": C2}}]
    return jobs.ppo_chunk({"games": gm, "learner": C2, "nh": 256, "temp": 1.0, "lam": 0.95})


def test_errored_and_overflowed_games_give_no_records(monkeypatch):
    ok = _chunk(monkeypatch, None)
    assert len(ok["A"]) > 0 and not any(r["error"] or r["overflow"] for r in ok["results"])
    err = _chunk(monkeypatch, lambda r: {**r, "error": True})
    assert len(err["A"]) == 0 and all(r["error"] for r in err["results"]) and len(err["results"]) == 2
    ovf = _chunk(monkeypatch, lambda r: {**r, "overflow": True})
    assert len(ovf["A"]) == 0 and all(r["overflow"] for r in ovf["results"])


def test_stream_is_seeded_not_named_with_frozen_and_exclusions(tmp_path):
    cfg = dict(train.CFG, games=400, seed="s7", frozen="c2:0.15,c1:0.15", drill_share=0.3)
    a = train.make_games(cfg, 2, 5, "runA", ["x.npy", "y.npy"], "learner.npy")
    b = train.make_games(cfg, 2, 5, "runB", ["x.npy", "y.npy"], "learner.npy")
    assert a == b
    tags = [g["opp"].get("tag") for g in a]
    assert 30 < tags.count("frozen c2") < 100 and 30 < tags.count("frozen c1") < 100
    legacy = train.make_games(dict(cfg, seed=""), 2, 5, "runA", ["x.npy", "y.npy"], "learner.npy")
    assert legacy != a
    ex = {",".join(sorted(g["kingdom"])) for g in a[:50]}
    f = tmp_path / "ex.json"
    f.write_text(json.dumps({"kingdoms": sorted(ex)}))
    c = train.make_games(cfg, 2, 5, "runA", ["x.npy", "y.npy"], "learner.npy", train.load_exclusions(str(f)))
    assert not ex & {",".join(sorted(g["kingdom"])) for g in c}


@pytest.mark.skipif(not os.path.exists(C2_PT), reason="needs runs/c2/net_2654.pt")
def test_two_runs_with_one_seed_write_identical_nets():
    names = ["_test_seed_a", "_test_seed_b"]
    try:
        for n in names:
            subprocess.run([sys.executable, "train.py", "--run", n, "--init_from", C2_PT, "--stage", "2", "--set", "games=40", "eval_kingdoms=2", "workers=2",
                            "iterations=4", "goal_every=0", "snapshot_every=2", "seed=t", "frozen=c2:0.2", "chunk=10", "temp=0.8"],
                           cwd=ROOT, check=True, capture_output=True)
        a, b = (np.load(os.path.join(ROOT, "runs", n, "net_4.npy")) for n in names)
        assert np.array_equal(a, b)
        ev = [json.loads(x) for x in open(os.path.join(ROOT, "runs", names[0], "events.jsonl"))]
        chk = [e for e in ev if e["kind"] == "init_check"][0]
        assert chk["weights_equal_source"] and chk["mean_abs_ratio_err"] < 1e-5
    finally:
        for n in names:
            shutil.rmtree(os.path.join(ROOT, "runs", n), ignore_errors=True)
