# dominion-rl-v3: chains of action cards, and the best Dominion model yet

**🌐 Live site: https://jca374.github.io/dominion-rl-v3/**

The fifth method of the Dominion project. It started from a question: can one network learn to play its action cards in
the right order (Village before Smithy, Festival before the draw, what to Throne), or does it need a statistical model that
calculates chains?

**The answer, measured.** A turn planner (every option played to the end of the turn on 128 shuffles of the unseen cards)
graded 14,243 action decisions from engine kingdoms. rl-v1's champion network picks the coin-best card 91.6% of the time
(76% with Throne Room in hand) and loses $0.09 per decision. But with the planner, or a hand-written chain rule, choosing its
actions, it **does not win more games** (48–50% head to head, 800–2,000 games per test). So the player stays one network,
with no chain module.

**What rl-v3 does instead.** It trains on from rl-v1's best checkpoint (iteration 2,219) and adds rl-v2's 115 helper inputs
at zero weight. Its curriculum ends with a quarter of its games against rl-v1's frozen champion, and a **card school**
forces its opening buys to cards the champion never tried (Chapel, Mine, Library, …). The card audit found those cards; the
engine got whole-game tests for all 26 cards. Goal: ≥ 55% head to head against rl-v1's champion on fresh games.

```
git clone --recursive https://github.com/JCA374/dominion-rl-v3 && cd dominion-rl-v3
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy pytest pillow matplotlib
.venv/bin/python -m pytest -q tests                        # builds rl.so; 20 passed
.venv/bin/python experiments/chain_exam.py build && .venv/bin/python experiments/chain_exam.py score
.venv/bin/python experiments/plan_vs.py opponents/rl_r2_2200.npy --k 32 --leaf 1 --engine
.venv/bin/python experiments/card_audit.py opponents/rl_r2_2200.npy --name champion
.venv/bin/python -u train.py --run c1                      # stages in STAGES at the top of train.py
```

| Folder | What |
|---|---|
| `train.py` | stages, card school, PPO, evaluation (incl. head to head vs the champion); writes `runs/<run>/` |
| `c/rl.c`, `rl/` | the agent in C (features, net, turn planner, chain rule, forced buys), the ctypes bridge, the PyTorch net, game jobs and GAE |
| `experiments/` | chain exam, planner/rule head to head, card audit, goal checks, the once-only confirmation, the journal |
| `opponents/` | rl-v1's checkpoint and champion, rl-v2's net, az-v1's best net |
| `play/` | the net in numpy for playing against it in the browser |
| `engine/`, `az-v1/` | submodules: the rules (with `de_resume_action`), and az-v1 (with dominion-v1's genomes and bots) |
