# Glossary

One term per line, shown on the Study tab:

**Dominion** — a deck-building card game: each turn you play action cards from your hand, then buy one card for your deck; most victory points when the game ends wins
**decision** — one question the engine asks a player: which action to play, what to buy, what to trash, discard or gain; about 22 kinds
**kingdom** — the 10 action cards (out of the base set's 26) dealt for one game
**GA genome** — dominion-v1's evolved priority lists: a fixed rule for every kind of decision, found by a genetic algorithm
**GA base** — the best portfolio-v2 genome whose required cards are in the kingdom; the reference opponent of every measure
**az-v1 / AZ best** — the AlphaZero-light project: the GA genome's action play with buys chosen by IS-MCTS (400 simulations) guided by a network; 70% vs the GA base
**reinforcement learning (RL)** — learning to act from the results of one's own actions (here: won or lost games), not from correct answers
**behaviour cloning (BC)** — supervised learning of an expert's choices; rl-v1 started this way (imitating the GA genome)
**self-play** — the network plays against a copy of itself; the opponent always matches the learner's strength
**tabula rasa** — "blank slate": training from random weights with no examples or rules
**stage / curriculum** — training in steps with different opponents: rl-v1's league, then + the champion, then everyone
**plateau gate** — stage 1 ends when the score against the GA genome has not set a new best for 60 iterations
**rl-v1** — this project's parent: the same network and PPO, started as a copy of the GA genome; 82.8% vs the GA base
**policy** — the network's probabilities over the legal answers to a decision
**value / win estimate** — the network's guess of the final result from here, in [−1, +1]
**PPO** — proximal policy optimisation: a policy-gradient method that limits how far one update can change the policy (the clip)
**policy gradient** — make choices from games that went better than expected more likely, and the others less likely
**advantage** — how much better a decision turned out than the network expected; positive → reinforce, negative → discourage
**GAE** — generalised advantage estimation: advantages built from the chain of win estimates along a game, with a mixing factor λ
**entropy** — how spread out the policy's probabilities are; high = undecided / exploring, low = certain
**KL divergence** — how far one update moved the policy; PPO stops an update early when it gets too large
**league** — the mix of opponents in training; in stage 3: the GA base, the bots, az-v1, rl-v1, past snapshots and the network itself
**snapshot** — a saved copy of the network at some iteration, used later as an opponent
**argmax play** — taking the highest-probability answer; used in every evaluation (training games sample instead)
**goal check** — az-v1's 1,000 goal games against the GA base plus 1,000 head-to-head games against az-v1's best agent
**Wilson interval** — the 95% confidence interval of a win rate; about ±3 points at 1,000 games
**winner's curse** — the best of many noisy evaluations is flattered by luck
**chain** — the order in which a turn's action cards are played: +Actions cards (Village, Festival) first keep the actions for terminal cards (Smithy, Witch) later
**terminal** — an action card that gives no +Actions: playing it with your last action ends the action phase
**terminal collision** — two terminal cards in hand with one action: only one of them can be played
**turn planner** — tries every option of an action decision on K shuffles of the unseen cards, plays each to the end of the turn and scores it (`rl_plan` in c/rl.c)
**coins judge / value judge** — the planner's two scores: coins at the first buy, or the network's win estimate after the turn
**chain exam** — 14,243 action decisions from engine kingdoms, graded by the planner (experiments/chain_exam.py)
**determinisation** — one guess of the hidden cards (a shuffle of the draw pile, a redraw of the opponent's hand) to plan on
**common random numbers** — every option is tried on the same shuffles, so luck cancels out of the comparison
**champion** — rl-v1's best network (r2 at iteration 2,200), the model rl-v3 has to beat
**card school** — curriculum: in 15% of games the learner's first buys are forced to a card it never tried; those buys are not trained on
**card audit** — per card: how often the model buys and plays it and answers every question it asks (experiments/card_audit.py)
**widening** — adding inputs to a trained network with zero weights, so it plays exactly as before
