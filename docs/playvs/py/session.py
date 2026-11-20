"""One game of a person against a model, for the Play-against-it page (runs in the browser with Pyodide).

The Python reference engine runs the game; the model's decisions are answered here, and the game stops at each of
the person's decisions. `view()` is everything the person may see (never the opponent's hand or deck order).
"""
from __future__ import annotations

import json
import random

from dominion_engine.bots import BigMoney
from dominion_engine.cards import CARDS, NAME, NUM_CARDS
from dominion_engine.game import (KIND_NAME, SENTRY_CARD, SENTRY_DISCARD, SENTRY_KEEP, SENTRY_TRASH, Game)
from dominion_engine import rules

import numpy as np

import rlplay

SENTRY_LABEL = {SENTRY_TRASH: "Trash it", SENTRY_DISCARD: "Discard it", SENTRY_KEEP: "Keep it on top"}


def kingdom_random(seed: int) -> list[str]:
    from dominion_engine.cards import KINGDOM_CARDS
    return sorted(random.Random(seed).sample(list(KINGDOM_CARDS), 10))


class Session:
    def __init__(self, kingdom, seed, opponent, flat=None, human=0, advisor=None):
        """advisor: flat weights of a net that judges the person's position (win probability) and suggests a move
        at each of their decisions; it never plays. answers: every answer the person gave (take back = replay)."""
        self._init = (list(kingdom), seed, opponent, flat, human)
        self.adv = rlplay.Net(np.asarray(advisor, np.float32)) if advisor is not None else None
        self._adv_flat = advisor
        self.answers = []
        self.review = []                              # per decision of the person: what the advisor thought
        self.wp = []                                  # [turn, win probability] at each decision of the person
        self._advice = None
        self.k = list(kingdom)
        self.human = human
        self.g = Game(self.k, seed)
        self.gen = self.g.run()
        if opponent == "Big Money":
            self.agent = BigMoney()
        else:
            self.agent = rlplay.NetAgent(flat)
        self.opponent = opponent
        self.log = []
        self.result = None
        self.d = None
        self._draws = []                              # (player, cards) for every draw, as the engine makes it
        self._open = None                             # the action whose effects are being collected
        engine_draw = self.g._draw

        def draw(p, n):
            before = len(p.hand)
            engine_draw(p, n)
            self._draws.append((self.g.players.index(p), p.hand[before:]))
        self.g._draw = draw
        engine_gain = self.g._gain

        def gain(p, card, to="discard"):
            ok = engine_gain(p, card, to)
            if ok:
                self._draws.append((self.g.players.index(p), ("gain", card)))
            return ok
        self.g._gain = gain
        self._advance(None, first=True)

    def _say(self, who, d, ans):
        p = "You" if who == self.human else "Opponent"
        kind = KIND_NAME[d.kind].replace("_", " ")
        if d.kind == SENTRY_CARD:
            what = SENTRY_LABEL[ans].lower()
        else:
            what = {0: "ends actions", 1: "ends turn"}.get(d.kind, "no") if ans < 0 else NAME[ans]
        about = f" ({NAME[d.subject]})" if 0 <= d.subject < NUM_CARDS and d.kind in (6, 8, 14, 15, 16) else ""
        entry = {"turn": self.g.players[who].turns + 1, "who": p, "text": f"{kind}{about}: {what}"}
        self.log.append(entry)
        if d.kind in (0, 2, 14) and ans >= 0:         # play action, Throne Room's choice, Vassal's card
            self._open = {"entry": entry, "card": ans, "who": who, "start": len(self._draws), "times": 2 if d.kind == 2 else 1}

    def _close(self):
        """Append to the open action's log line what it did: the cards it drew, +Actions, +Buys, +$."""
        o, self._open = self._open, None
        if o is None:
            return
        c, t = CARDS[o["card"]], o["times"]
        drawn = {0: [], 1: []}
        gained = {0: [], 1: []}
        for who, x in self._draws[o["start"]:]:
            if isinstance(x, tuple):
                gained[who].append(x[1])
            else:
                drawn[who].extend(x)
        me, other = o["who"], 1 - o["who"]
        parts = []
        for who in (me, other):
            mine = who == me
            if drawn[who]:
                if who == self.human:
                    parts.append(("draws " if mine else "you draw ") + ", ".join(NAME[x] for x in drawn[who]))
                else:
                    n = len(drawn[who])
                    parts.append(("draws" if mine else "opponent draws") + f" {n} card{'s' if n > 1 else ''}")
            if gained[who]:
                parts.append(("gains " if mine else ("you gain " if who == self.human else "opponent gains "))
                             + ", ".join(NAME[x] for x in gained[who]))
        for n, word in ((c.actions, "action"), (c.buys, "buy")):
            if n:
                parts.append(f"+{n * t} {word}{'s' if n * t > 1 else ''}")
        if c.coins:
            parts.append(f"+${c.coins * t}")
        if t == 2:
            parts.insert(0, "played twice")
        if parts:
            o["entry"]["text"] += " → " + "; ".join(parts)

    def _advance(self, ans, first=False):
        try:
            d = next(self.gen) if first else self.gen.send(ans)
            while True:
                self._close()
                forced = len(d.options) + bool(d.may_pass) == 1
                if d.player == self.human and not forced:
                    self.d = d
                    return
                a = (d.options[0] if d.options else -1) if forced else self.agent.decide(self.g, d)
                if d.player != self.human and not forced:
                    self._say(d.player, d, a)
                d = self.gen.send(a)
        except StopIteration as stop:
            self._close()
            self.result = stop.value
            self.d = None

    def answer(self, a: int):
        assert self.d is not None
        d = self.d
        assert a in d.options or (a == -1 and d.may_pass), (a, d)
        adv = self.advice()
        if adv:
            p = dict((o, q) for o, q in adv["probs"])
            self.review.append({"turn": self.g.players[self.human].turns + 1, "kind": KIND_NAME[d.kind],
                                "subject": NAME[d.subject] if 0 <= d.subject < NUM_CARDS else None,
                                "chose": self._label(d, a), "p_chose": p.get(a, 0.0),
                                "best": self._label(d, adv["probs"][0][0]), "p_best": adv["probs"][0][1],
                                "win": adv["win"]})
        self.answers.append(a)
        self._say(self.human, d, a)
        self.d = None
        self._advice = None
        self._advance(a)
        return self.view()

    def _label(self, d, o):
        if o == -1:
            return "pass"
        return SENTRY_LABEL[o] if d.kind == SENTRY_CARD else NAME[o]

    def advice(self):
        """The advisor on the person's current decision: its win probability for them and its probability for every
        answer (best first; -1 = pass). None without an advisor or a decision. Cached until the next answer."""
        if self.adv is None or self.d is None:
            return None
        if self._advice is None:
            d = self.d
            lg, v = self.adv.forward(rlplay.features(self.g, self.human, d.kind, d.subject, self.adv.nf > rlplay.NF0))
            opts = list(d.options) + ([rlplay.PASS] if d.may_pass else [])
            legal = np.array(opts)
            z = lg[legal] - lg[legal].max()
            pr = np.exp(z) / np.exp(z).sum()
            probs = sorted(((-1 if o == rlplay.PASS else int(o), float(q)) for o, q in zip(opts, pr)), key=lambda t: -t[1])
            win = (v + 1) / 2
            self._advice = {"win": win, "probs": probs}
            turn = self.g.players[self.human].turns + 1
            if not self.wp or self.wp[-1][0] != turn:
                self.wp.append([turn, win])
        return self._advice

    def undo(self, n: int = 1):
        """Take back the person's last n answers: replay the game from the start (engine and nets are deterministic)."""
        keep = self.answers[:max(0, len(self.answers) - n)]
        k, seed, opponent, flat, human = self._init
        fresh = Session(k, seed, opponent, flat, human, self._adv_flat)
        for a in keep:
            fresh.advice()
            fresh.answer(a)
        self.__dict__.update(fresh.__dict__)
        return self.view()

    def _thinking(self):
        """The model's last non-forced decision: its top options with probabilities, and its win estimate."""
        last = getattr(self.agent, "last", None)
        if not last:
            return None

        def label(o):
            if o == rlplay.PASS:
                return "pass"
            return SENTRY_LABEL[o] if last["kind"] == SENTRY_CARD else NAME[o]
        top = sorted(zip(last["options"], last["probs"]), key=lambda t: -t[1])[:4]
        return {"value": last["value"], "kind": KIND_NAME[last["kind"]].replace("_", " "),
                "top": [[label(o), p] for o, p in top]}

    def view(self) -> str:
        g, me, op = self.g, self.g.players[self.human], self.g.players[1 - self.human]

        def counts(cards):
            out = {}
            for c in cards:
                out[NAME[c]] = out.get(NAME[c], 0) + 1
            return dict(sorted(out.items(), key=lambda kv: (CARDS[[i for i in range(NUM_CARDS) if NAME[i] == kv[0]][0]].cost, kv[0])))
        supply = [{"card": NAME[c], "id": c, "cost": CARDS[c].cost, "left": g.supply[c],
                   "kingdom": c in self.g.kingdom}
                  for c in range(NUM_CARDS) if g.supply[c] >= 0]
        supply.sort(key=lambda s: (s["kingdom"], s["cost"], s["card"]))
        q = None
        if self.d is not None:
            d = self.d
            opts = [{"id": o, "label": SENTRY_LABEL[o] if d.kind == SENTRY_CARD else NAME[o]} for o in d.options]
            q = {"kind": KIND_NAME[d.kind], "kind_id": d.kind, "subject": d.subject,
                 "subject_name": NAME[d.subject] if 0 <= d.subject < NUM_CARDS else None,
                 "options": opts, "may_pass": d.may_pass}
        res = None
        if self.result is not None:
            r = self.result
            res = {"scores": [r.scores[self.human], r.scores[1 - self.human]],
                   "won": self.human in r.winners and len(r.winners) == 1, "shared": len(r.winners) > 1,
                   "turns": r.turns, "opp_deck": counts(r.decks[1 - self.human])}
        mine = g.current == self.human
        return json.dumps({
            "kingdom": self.k, "supply": supply, "question": q, "result": res, "log": self.log[-60:],
            "me": {"hand": sorted(NAME[c] for c in me.hand), "play": [NAME[c] for c in me.play],
                   "deck": len(me.deck), "discard": len(me.discard),
                   "discard_top": NAME[me.discard[-1]] if me.discard else None,
                   "owned": counts(me.all_cards()), "vp": rules.score(me.all_cards()), "turn": me.turns + 1},
            "opp": {"name": self.opponent, "hand": len(op.hand), "deck": len(op.deck), "discard": len(op.discard),
                    "play": [NAME[c] for c in op.play], "owned": counts(op.all_cards()),
                    "vp": rules.score(op.all_cards()), "turn": op.turns + 1},
            "coins": g.coins if mine else None, "actions": g.actions if mine else None,
            "buys": g.buys if mine else None, "trash": counts(g.trash),
            "thinking": self._thinking(),
            "advice": self.advice(), "wp": self.wp, "review": self.review if self.result is not None else None,
            "answers": len(self.answers),
        })
