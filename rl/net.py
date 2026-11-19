"""The policy/value network in PyTorch, and its flat float32 export in c/rl.c's layout.

x[298] -> relu(W1 x + b1) -> relu(W2 h1 + b2) -> logits Wp h2 + bp (34, masked to the legal options)
                                              -> value tanh(Wv h2 + bv)
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .core import NA, NF, NF0, lib


class Net(nn.Module):
    def __init__(self, nh: int = 256, nf: int = NF):
        super().__init__()
        self.nh, self.nf = nh, nf
        self.l1 = nn.Linear(nf, nh)
        self.l2 = nn.Linear(nh, nh)
        self.pi = nn.Linear(nh, NA)
        self.v = nn.Linear(nh, 1)
        nn.init.normal_(self.pi.weight, 0, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.normal_(self.v.weight, 0, 0.01)

    def forward(self, x, legal=None):
        x = x[..., :self.nf]                     # a net without the helper inputs reads the first NF0
        h = torch.relu(self.l2(torch.relu(self.l1(x))))
        logits = self.pi(h)
        if legal is not None:
            logits = logits.masked_fill(legal == 0, -1e9)
        return logits, torch.tanh(self.v(h)).squeeze(-1)

    def flat(self) -> np.ndarray:
        """Weights in c/rl.c's order: W1 b1 W2 b2 Wp bp Wv bv (row-major, out x in)."""
        parts = [self.l1.weight, self.l1.bias, self.l2.weight, self.l2.bias, self.pi.weight, self.pi.bias,
                 self.v.weight.reshape(-1), self.v.bias]
        out = torch.cat([p.detach().reshape(-1) for p in parts]).numpy().astype(np.float32)
        assert out.size == lib.rl_num_params(self.nh, self.nf)
        return out


def load(path: str) -> Net:
    d = torch.load(path, weights_only=True)
    n = Net(d["nh"], d.get("nf", NF0))          # nets saved before the helper inputs read NF0
    n.load_state_dict(d["state"])
    return n


def save(net: Net, path: str) -> None:
    torch.save({"nh": net.nh, "nf": net.nf, "state": net.state_dict()}, path)
    np.save(path.replace(".pt", ".npy"), net.flat())
