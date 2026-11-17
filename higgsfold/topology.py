"""Local topology of the backbone: flattening points, branch points and perestroikas.

Phys. Rev. E 111, 024406 (2025):
  * a flattening point is a link where the torsion passes through 0 (consecutive binormals parallel),
  * a branch point is a link where the torsion passes through +-pi (binormals antiparallel),
  * the link indicator b_i = 1 if tau_i * tau_{i+1} <= 0 marks either crossing,
  * C(k) = <sum_i b_i b_{i+k}> / (N - k) is the correlation function used to track perestroikas
    (C(0) ~ 0.13 for folded myoglobin, ~0.44 for a self-avoiding random walk, 0.5 for an ideal random walk),
  * Gamma(n1, n2) = sum of wrapped torsion increments; the folding index Ind_f = Gamma / pi.
"""
from __future__ import annotations
import numpy as np
from .geometry import wrap

def crossings(tau):
    """Per-link crossing type between consecutive torsions: 'flattening' (through 0),
    'branch' (through +-pi) or '' (no sign change). Returns (indicator b, types) of length N."""
    tau = np.asarray(tau, float); n = len(tau)
    b = np.zeros(n, int); kind = np.array([""] * n, dtype=object)
    for i in range(n - 1):
        t0, t1 = tau[i], tau[i + 1]
        if np.isnan(t0) or np.isnan(t1):
            continue
        if t0 * t1 <= 0:
            b[i] = 1
            kind[i] = "flattening" if abs(t0) + abs(t1) < np.pi else "branch"
    return b, kind

def correlation(b, kmax=20):
    b = np.asarray(b, float); n = len(b)
    return np.array([np.sum(b[: n - k] * b[k:]) / (n - k) for k in range(min(kmax, n - 1) + 1)])

def folding_index(tau, n1=None, n2=None):
    t = np.asarray(tau, float); t = t[~np.isnan(t)]
    t = t[n1:n2]
    gamma = float(np.sum(wrap(np.diff(t))))
    return gamma, gamma / np.pi

def summary(kappa, tau):
    b, kind = crossings(tau)
    valid = np.sum(~np.isnan(tau[:-1]) & ~np.isnan(tau[1:]))
    gamma, ind = folding_index(tau)
    return {
        "n_flattening": int(np.sum(kind == "flattening")),
        "n_branch": int(np.sum(kind == "branch")),
        "C0": float(b.sum() / max(valid, 1)),
        "Gamma": gamma,
        "folding_index": ind,
        "n_inflection_like": int(np.sum(np.asarray(kappa) < 0.5)),
    }
