"""Dominion base set (2nd edition) with Steam Dominion's rules, for training any kind of model.

    from dominion_engine import Game, play, cards
    result = play(["Smithy", ...], [agent0, agent1], seed=1)

`game` is the Python reference engine (step-wise, any agent); `fast` runs the identical C engine.
"""
from . import cards, rules
from .game import Agent, Decision, Game, PlayerState, Result, play

__all__ = ["Agent", "Decision", "Game", "PlayerState", "Result", "cards", "play", "rules"]
