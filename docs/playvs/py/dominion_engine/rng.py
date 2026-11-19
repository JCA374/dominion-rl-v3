"""xorshift64, bit-identical to the C engine, so both engines shuffle the same way."""
from __future__ import annotations

MASK = (1 << 64) - 1


class Rng:
    __slots__ = ("state",)

    def __init__(self, seed: int):
        self.state = (seed & MASK) or 1  # xorshift can't hold 0

    def next(self) -> int:
        x = self.state
        x ^= (x << 13) & MASK
        x ^= x >> 7
        x ^= (x << 17) & MASK
        self.state = x
        return x

    def below(self, n: int) -> int:
        """Uniform-ish int in [0, n)."""
        return self.next() % n

    def shuffle(self, lst: list) -> None:
        """Fisher-Yates from the back, same draw order as the C engine."""
        for i in range(len(lst) - 1, 0, -1):
            j = self.next() % (i + 1)
            lst[i], lst[j] = lst[j], lst[i]
