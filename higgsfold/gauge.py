"""Z2 gauge fixing of the (kappa, tau) description.

The discrete Frenet description has a Z2 redundancy (Niemi et al.): at any link j one may
shift the torsion tau_j -> tau_j - pi and flip the sign of every subsequent bond angle,
kappa_l -> -kappa_l for l >= j+1, without changing the 3D chain. Following Phys. Rev. E 111,
024406 (2025), the gauge is fixed by minimizing the torsion-variation functional

    delta_tau = sum_i | wrap(tau'_{i+1} - tau'_i) |^2,

which is the discrete analogue of E_tau = int ds (d_s tau)^2. Because tau'_i depends only on the
binary choice f_i at link i and the cost couples neighbouring links, the global minimum is found
exactly with a two-state Viterbi / dynamic-programming pass (the paper does not specify an
algorithm; this exact solver is our implementation choice).

After gauge fixing, sign changes ("domain walls") of the signed bond angle kappa' mark soliton
centres.
"""
from __future__ import annotations
import numpy as np
from .geometry import wrap

def apply_flips(kappa, tau, flips):
    """Apply Z2 flips (bool array over links, same indexing as tau) -> signed kappa', tau'."""
    kappa = np.array(kappa, float); tau = np.array(tau, float)
    flips = np.asarray(flips, bool)
    tau_p = tau.copy()
    tau_p[flips] = wrap(tau[flips] - np.pi)
    # parity of flips on links strictly before vertex l  (link j affects vertices >= j+1)
    parity = np.concatenate([[0], np.cumsum(flips.astype(int))[:-1]]) % 2
    kappa_p = np.where(parity == 1, -kappa, kappa)
    return kappa_p, tau_p

def gauge_fix(kappa, tau, wall_penalty=1.0):
    """Return (kappa_signed, tau_gauged, flips) minimizing

        sum_i wrap(tau'_{i+1} - tau'_i)^2  +  wall_penalty * (number of flips).

    Each flip creates one sign domain wall in kappa', so `wall_penalty` plays the role of a
    domain-wall tension. With wall_penalty=0 the criterion is exactly the paper's torsion
    variation, which is degenerate inside beta-strands (tau ~ pi, flipping costs ~nothing) and
    produces spurious walls there; a small positive value breaks the tie. Default 1.0 rad^2.
    """
    tau = np.asarray(tau, float)
    idx = np.where(~np.isnan(tau))[0]
    flips = np.zeros(len(tau), bool)
    if len(idx) == 0:
        return np.array(kappa, float), tau.copy(), flips
    T = tau[idx]
    cand = np.stack([T, wrap(T - np.pi)], axis=1)          # state 0 = keep, 1 = flip
    n = len(T)
    cost = np.zeros((n, 2)); back = np.zeros((n, 2), int)
    cost[0, 1] = wall_penalty
    pen = np.array([0.0, wall_penalty])
    for i in range(1, n):
        d = wrap(cand[i][None, :] - cand[i - 1][:, None]) ** 2   # [prev, cur]
        tot = cost[i - 1][:, None] + d + pen[None, :]
        back[i] = np.argmin(tot, axis=0)
        cost[i] = np.min(tot, axis=0)
    s = np.zeros(n, int); s[-1] = int(np.argmin(cost[-1]))
    for i in range(n - 1, 0, -1):
        s[i - 1] = back[i, s[i]]
    flips[idx] = s.astype(bool)
    kp, tp = apply_flips(kappa, tau, flips)
    return kp, tp, flips

def torsion_variation(tau):
    t = np.asarray(tau, float); t = t[~np.isnan(t)]
    return float(np.sum(wrap(np.diff(t)) ** 2))

def domain_walls(kappa_signed, min_separation=1):
    """Indices w such that sign(kappa'_w) != sign(kappa'_{w+1}); a wall sits between w and w+1."""
    k = np.asarray(kappa_signed, float)
    ok = ~np.isnan(k)
    s = np.sign(k)
    walls = [i for i in range(len(k) - 1) if ok[i] and ok[i + 1] and s[i] * s[i + 1] < 0]
    return np.array(walls, int)

def gauge_fix_smooth(kappa, tau, wall_penalty=0.0, w_kappa=1.0, w_tau=1.0):
    """Z2 gauge that makes the *signed* bond-angle profile and the torsion profile smooth:

        min  w_kappa * sum_j (kappa'_{j+1} - kappa'_j)^2 + w_tau * sum_i wrap(tau'_{i+1} - tau'_i)^2
             + wall_penalty * (#walls)

    A wall placed where |kappa| is large costs ~(2 kappa)^2, so walls are pushed to the
    low-curvature sites where a kink (dark soliton) passes most naturally - the configuration the
    soliton model describes. Solved exactly by dynamic programming over the parity state of two
    consecutive vertices (4 states). Returns (kappa_signed, tau_gauged, flips).
    """
    kappa = np.asarray(kappa, float); tau = np.asarray(tau, float)
    n = len(kappa)
    v = np.where(~np.isnan(kappa))[0]                       # vertices 1..N-2
    links = np.where(~np.isnan(tau))[0]                     # links 1..N-3
    if len(links) == 0:
        return kappa.copy(), tau.copy(), np.zeros(n, bool)
    # parity P_j for vertices v[0]..v[-1]; flip on link i  <=>  P_{i+1} != P_i
    js = list(v)
    K = kappa
    def tau_p(i, pi, pi1):
        return wrap(tau[i] - np.pi * (pi ^ pi1))
    # DP over consecutive vertex pairs (P_j, P_{j+1}); first vertex parity fixed to 0 (global Z2)
    INF = 1e18
    # state after processing vertex j as pair (P_{j-1}, P_j)
    j0 = js[0]
    cost = {(0, 0): 0.0, (0, 1): INF}   # dummy predecessor parity 0
    cost = {}
    # initialise with vertex j0 and j0+1
    first = {}
    for p1 in (0, 1):
        kk0 = K[j0]; kk1 = (-1) ** p1 * K[j0 + 1]
        c = w_kappa * (kk1 - kk0) ** 2 + (wall_penalty if p1 else 0.0)
        first[(0, p1)] = (c, None)
    layers = [first]
    for j in js[1:-1]:                 # transition to vertex j+1, state (P_j, P_{j+1})
        nxt = {}
        for (pa, pb), (c, _) in layers[-1].items():
            for pc in (0, 1):
                kb = (-1) ** pb * K[j]; kc = (-1) ** pc * K[j + 1]
                cc = c + w_kappa * (kc - kb) ** 2 + (wall_penalty if pb != pc else 0.0)
                i0, i1 = j - 1, j            # links j-1 (between P_{j-1},P_j) and j (P_j,P_{j+1})
                if not np.isnan(tau[i0]) and not np.isnan(tau[i1]):
                    cc += w_tau * wrap(tau_p(i1, pb, pc) - tau_p(i0, pa, pb)) ** 2
                key = (pb, pc)
                if key not in nxt or cc < nxt[key][0]:
                    nxt[key] = (cc, (pa, pb))
        layers.append(nxt)
    # backtrack
    state = min(layers[-1], key=lambda s: layers[-1][s][0])
    P = {js[-1]: state[1], js[-2]: state[0]}
    for li in range(len(layers) - 1, 0, -1):
        prev = layers[li][state][1]
        P[js[li - 1]] = prev[0]
        state = prev
    flips = np.zeros(n, bool)
    for j in js[:-1]:
        flips[j] = P[j] != P[j + 1]
    flips &= ~np.isnan(tau)
    kp, tp = apply_flips(kappa, tau, flips)
    return kp, tp, flips

def gauge_match(kappa, tau, kappa_ref, tau_ref):
    """Choose the Z2 gauge of (kappa, tau) that best matches a reference gauge-fixed profile.

    Used to compare two structures (e.g. the rebuilt soliton model and the native chain) in a
    common gauge: gauge-equivalent descriptions of the same 3D chain can otherwise look different.
    Exact two-state dynamic programming over the parity of the bond-angle sign.
    """
    kappa = np.asarray(kappa, float); tau = np.asarray(tau, float)
    n = len(kappa)
    js = [j for j in range(n) if not np.isnan(kappa[j])]
    INF = 1e18
    cost = {0: 0.0, 1: 0.0}; back = []
    def kc(j, p):
        r = kappa_ref[j]
        return 0.0 if np.isnan(r) else ((-1) ** p * kappa[j] - r) ** 2
    cost = {p: kc(js[0], p) for p in (0, 1)}
    for j in js[1:]:
        i = j - 1                                   # link between j-1 and j
        new = {}; bk = {}
        for p in (0, 1):
            best, arg = INF, 0
            for q in (0, 1):
                c = cost[q] + kc(j, p)
                if not np.isnan(tau[i]) and not np.isnan(tau_ref[i]):
                    c += wrap(wrap(tau[i] - np.pi * (p ^ q)) - tau_ref[i]) ** 2
                if c < best:
                    best, arg = c, q
            new[p], bk[p] = best, arg
        cost = new; back.append(bk)
    P = {js[-1]: min(cost, key=cost.get)}
    for idx in range(len(js) - 1, 0, -1):
        P[js[idx - 1]] = back[idx - 1][P[js[idx]]]
    flips = np.zeros(n, bool)
    for a, b in zip(js[:-1], js[1:]):
        flips[a] = P[a] != P[b]
    flips &= ~np.isnan(tau)
    kp, tp = apply_flips(kappa, tau, flips)
    if P[js[0]] == 1:                               # global sign of the first vertex
        kp = -kp
    return kp, tp
