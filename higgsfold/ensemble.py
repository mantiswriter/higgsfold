"""Perestroika statistics over a conformational ensemble (MD trajectory or NMR models).

The paper samples a thermal ensemble with Monte Carlo; here any ensemble can be supplied, e.g.
an all-atom MD trajectory. For each frame the link indicator b_i is computed, and the
order parameter Q = C(0) is tracked with its Binder cumulant B_Q = <Q^4>/<Q^2>^2 - 3 and
susceptibility S_Q = <Q^2> - <Q>^2. Per-link perestroika frequency = fraction of frames in
which b_i differs from the reference (first) frame - a map of where local topology changes.
"""
from __future__ import annotations
import numpy as np
from .geometry import kappa_tau
from .topology import crossings, correlation

def analyze_ensemble(frames, kmax=20, reference=0):
    frames = np.asarray(frames, float)
    B = []; C = []
    for x in frames:
        _, t = kappa_tau(x)
        b, _ = crossings(t)
        B.append(b); C.append(correlation(b[1:-3], kmax))
    B = np.array(B); C = np.array(C)
    Q = C[:, 0]
    q2 = np.mean(Q ** 2)
    return {
        "Q": Q, "C_mean": C.mean(0), "C_std": C.std(0),
        "binder": float(np.mean(Q ** 4) / q2 ** 2 - 3) if q2 > 0 else float("nan"),
        "susceptibility": float(np.var(Q)),
        "perestroika_freq": np.mean(B != B[reference][None, :], axis=0),
        "b": B,
    }
