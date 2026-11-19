"""play/rlplay.py (numpy, for the browser) computes exactly c/rl.c's features and picks exactly the C net's move:
C games are replayed in the Python reference engine and compared at every recorded decision."""
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "play"))

from rl.core import GA, NET, PASS, Seat, base_genome, play, random_kingdom  # noqa: E402


def _replay(k, seed, rec, check):
    import rlplay
    from dominion_engine.game import Game
    g = Game(k, seed)
    gen = g.run()
    j = 0
    try:
        d = next(gen)
        while True:
            nopt = len(d.options) + bool(d.may_pass)
            if nopt == 1:
                ans = d.options[0] if d.options else -1
            else:
                assert rec["KIND"][j] == d.kind and rec["SEAT"][j] == d.player, (j, d)
                check(j, g, d, rlplay)
                a = int(rec["A"][j])
                ans = -1 if a == PASS else a
                j += 1
            d = gen.send(ans)
    except StopIteration:
        pass
    assert j == len(rec["A"])


def test_features_equal_c():
    rng = random.Random(11)
    for i in range(12):
        k = random_kingdom(rng)
        g = base_genome(k)["ints"]
        r = play([Seat(GA, g, record=True), Seat(GA, g, record=True)], k, 1000 + i, rng_seed=7)
        rec = r["rec"]

        def check(j, game, d, rlplay):
            x = rlplay.features(game, d.player, d.kind, d.subject)
            n0 = rlplay.NF0                       # rl-v1's inputs: bit-exact; the helper inputs (double maths): 1e-6
            assert np.array_equal(x[:n0], rec["X"][j][:n0]), (j, d, np.nonzero(x[:n0] != rec["X"][j][:n0]))
            assert np.allclose(x[n0:], rec["X"][j][n0:], atol=1e-6), (j, d, np.nonzero(~np.isclose(x[n0:], rec["X"][j][n0:], atol=1e-6)))
        _replay(k, 1000 + i, rec, check)


def test_net_agent_plays_like_c():
    W = np.load(os.path.join(ROOT, "opponents", "rl_r2_2200.npy"))
    rng = random.Random(12)
    for i in range(6):
        k = random_kingdom(rng)
        g = base_genome(k)["ints"]
        r = play([Seat(NET, g, W, record=True), Seat(GA, g, record=True)], k, 2000 + i, rng_seed=9)
        rec = r["rec"]

        def check(j, game, d, rlplay):
            if d.player == 0:
                a = rlplay.NetAgent(W).decide(game, d)
                assert (PASS if a == -1 else a) == rec["A"][j], (j, d)
        _replay(k, 2000 + i, rec, check)


def test_online_game_smithy_draws_three_and_logs_them():
    """The online game's session: playing Smithy draws exactly 3 cards, and the log says which."""
    sys.path.insert(0, os.path.join(ROOT, "engine"))
    import session as S
    from dominion_engine.cards import ID, NAME
    k = ["Smithy", "Village", "Market", "Laboratory", "Festival", "Moat", "Cellar", "Chapel", "Gardens", "Mine"]
    seen = 0
    for seed in range(200):
        s = S.Session(k, seed, "Big Money", None, human=0)
        while s.d is not None and seen < 8:
            d = s.d
            if d.kind == 0 and ID["Smithy"] in d.options:
                p = s.g.players[0]
                hand, deck_disc = len(p.hand), len(p.deck) + len(p.discard)
                s.answer(ID["Smithy"])
                drew = min(3, deck_disc)
                if s.d is not None and s.d.kind in (0, 1):      # still in the same turn, before the treasures
                    if s.d.kind == 0:
                        assert len(p.hand) == hand - 1 + drew
                line = [l for l in s.log if l["text"].startswith("play action: Smithy")][-1]["text"]
                assert line.count(",") == drew - 1 and "draws" in line, line
                seen += 1
                continue
            opts = list(d.options) + ([-1] if d.may_pass else [])
            s.answer(ID["Smithy"] if ID["Smithy"] in opts and d.kind == 1 and seed % 2 else opts[-1] if d.kind != 1 else (ID["Smithy"] if ID["Smithy"] in opts else -1))
        if seen >= 8:
            break
    assert seen >= 8


def test_helpers_with_a_big_hand_off_turn():
    """Off turn the next hand is the current hand, which can hold more than 5 cards (it once overflowed C's tables).
    Council Room / Library kingdoms, many games: C must not crash, and Python must agree."""
    from rl.core import NET
    from rl.net import Net
    rng = random.Random(21)
    W = Net(32).flat()
    k = ["Council Room", "Library", "Smithy", "Village", "Laboratory", "Market", "Festival", "Militia", "Moat", "Cellar"]
    g = base_genome(k)["ints"]
    for i in range(60):
        r = play([Seat(NET, g, W, sample=True, record=True), Seat(NET, g, W, sample=True, record=True)], k, 3000 + i,
                 rng_seed=rng.getrandbits(32))
        assert not r["error"]
    r = play([Seat(GA, g, record=True), Seat(GA, g, record=True)], k, 4242, rng_seed=7)

    def check(j, game, d, rlplay):
        x = rlplay.features(game, d.player, d.kind, d.subject)
        assert np.allclose(x, r["rec"]["X"][j], atol=1e-6)
    _replay(k, 4242, r["rec"], check)
