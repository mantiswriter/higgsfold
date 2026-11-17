"""Thermal Monte Carlo of the discretised NLS free energy (reconstruction of PRE 111, 024406, Sec. on thermal unfolding).

Free energy (paper Eq. 35, signed/gauge-fixed kappa):

  F = -sum_i 2 k_{i+1} k_i + sum_i [ 2 k_i^2 + lam (k_i^2 - m^2)^2 + d/2 k_i^2 t_i^2
                                      - b k_i^2 t_i - a_i t_i + c/2 t_i^2 + h_i k_i ]

plus the hard-core excluded volume of Eq. 43 (|r_i - r_j| <= Delta rejected for |i-j| >= 2).

What the paper does NOT give, and what is chosen here (all disclosed in the README):
  * numerical parameters: lam, m, d, b, c are global constants chosen below; the per-site
    linear terms a_i (torsion) and h_i (bond angle, an addition to Eq. 35) are solved so that the
    native gauge-fixed (kappa, tau) is an exact stationary point of F. The Hessian is checked.
  * the Monte Carlo move: single-variable Gaussian perturbation of one kappa_i or tau_i, accepted
    with the Glauber probability 1/(1+exp(dF/T)) (Eq. 41).
  * temperature units: T is in the units of F here, so the absolute scale is not comparable with
    the paper's log10 T axis.
"""
from __future__ import annotations
import numpy as np
from .geometry import kappa_tau, reconstruct_fast, kabsch_rmsd
from .gauge import gauge_fix_smooth

try:
    from numba import njit
except Exception:                    # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]

PARAMS = dict(lam=1.0, m=0.0, d=1.0, b=0.0, c=1.0, g=2.0)   # g: coupling -g k_i k_{i+1} (paper: 2)


def native_model(ca, wall_penalty=1.0, **p):
    """Return model dict with native gauge-fixed angles and per-site fields making them stationary."""
    P = {**PARAMS, **p}
    n = len(ca)
    k, t = kappa_tau(ca)
    kg, tg, _ = gauge_fix_smooth(k, t, wall_penalty)
    kap = np.nan_to_num(kg); tau = np.nan_to_num(tg)
    kap[0] = kap[-1] = 0.0; tau[0] = 0.0; tau[-2:] = 0.0      # tau defined on 1..n-3 only
    kp = np.concatenate([[0.0], kap, [0.0]])                   # kappa_{-1}.. padded for neighbours
    nb = kp[:-2] + kp[2:]                                      # kappa_{i-1} + kappa_{i+1}
    lam, m, d, b, c, g = (P[x] for x in ("lam", "m", "d", "b", "c", "g"))
    dFdk = -g * nb + 4 * kap + 4 * lam * kap * (kap ** 2 - m ** 2) + d * kap * tau ** 2 - 2 * b * kap * tau
    h = -dFdk
    a = (d * kap ** 2 + c) * tau - b * kap ** 2
    bonds = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    return dict(kappa=kap, tau=tau, h=h, a=a, bonds=bonds, **P, n=n)


def hessian_min_eig(M):
    """Smallest eigenvalue of the Hessian of F at the native state over the free variables."""
    n = M["n"]; lam, m, d, b, c = (M[x] for x in ("lam", "m", "d", "b", "c"))
    kv = np.arange(1, n - 1); tv = np.arange(1, n - 2)
    nk, nt = len(kv), len(tv)
    H = np.zeros((nk + nt, nk + nt)); k = M["kappa"]; t = M["tau"]
    for r, i in enumerate(kv):
        H[r, r] = 4 + 4 * lam * (3 * k[i] ** 2 - m ** 2) + d * t[i] ** 2 - 2 * b * t[i]
        if r + 1 < nk: H[r, r + 1] = H[r + 1, r] = -M["g"]
        if i in tv:
            q = nk + int(np.where(tv == i)[0][0])
            H[r, q] = H[q, r] = 2 * d * k[i] * t[i] - 2 * b * k[i]
    for q, i in enumerate(tv):
        H[nk + q, nk + q] = d * k[i] ** 2 + c
    return float(np.linalg.eigvalsh(H)[0])


@njit(cache=True)
def _local_k(kap, tau, h, i, kv, lam, m, d, b, g, n):
    left = kap[i - 1] if i - 1 >= 1 else 0.0
    right = kap[i + 1] if i + 1 <= n - 2 else 0.0
    return (-g * kv * (left + right) + 2 * kv * kv + lam * (kv * kv - m * m) ** 2
            + 0.5 * d * kv * kv * tau[i] ** 2 - b * kv * kv * tau[i] + h[i] * kv)


@njit(cache=True)
def _local_t(kap, a, i, tv, d, b, c):
    return 0.5 * d * kap[i] ** 2 * tv * tv - b * kap[i] ** 2 * tv - a[i] * tv + 0.5 * c * tv * tv


@njit(cache=True)
def _clash(r, n, d2):
    for i in range(n):
        for j in range(i + 2, n):
            dx = r[i, 0] - r[j, 0]; dy = r[i, 1] - r[j, 1]; dz = r[i, 2] - r[j, 2]
            if dx * dx + dy * dy + dz * dz <= d2:
                return True
    return False


@njit(cache=True)
def _build(kappa, tau, bond, r):
    N = kappa.shape[0]
    t0, t1, t2 = 0.0, 0.0, 1.0; n0, n1, n2 = 1.0, 0.0, 0.0; b0, b1, b2 = 0.0, 1.0, 0.0
    r[0, 0] = 0.0; r[0, 1] = 0.0; r[0, 2] = 0.0
    r[1, 0] = bond[0] * t0; r[1, 1] = bond[0] * t1; r[1, 2] = bond[0] * t2
    for i in range(1, N - 1):
        if i >= 2:
            ta = tau[i - 1]; c = np.cos(ta); s = np.sin(ta)
            m0, m1, m2 = c * n0 + s * b0, c * n1 + s * b1, c * n2 + s * b2
            b0, b1, b2 = -s * n0 + c * b0, -s * n1 + c * b1, -s * n2 + c * b2
            n0, n1, n2 = m0, m1, m2
        k = kappa[i]; c = np.cos(k); s = np.sin(k)
        u0, u1, u2 = c * t0 + s * n0, c * t1 + s * n1, c * t2 + s * n2
        n0, n1, n2 = -s * t0 + c * n0, -s * t1 + c * n1, -s * t2 + c * n2
        t0, t1, t2 = u0, u1, u2
        r[i + 1, 0] = r[i, 0] + bond[i] * t0
        r[i + 1, 1] = r[i, 1] + bond[i] * t1
        r[i + 1, 2] = r[i, 2] + bond[i] * t2


@njit(cache=True)
def run_chain(kap, tau, h, a, bond, lam, m, d, b, c, g, T, nsweeps, nburn, every, delta, steps, seed):
    """Glauber dynamics. Modifies kap, tau in place. Returns (coords of sampled frames, acceptance rate)."""
    np.random.seed(seed)
    n = kap.shape[0]; d2 = delta * delta
    r = np.zeros((n, 3)); rn = np.zeros((n, 3))
    nfr = (nsweeps - nburn) // every
    frames = np.zeros((nfr, n, 3))
    acc = 0; tot = 0; fr = 0
    nk = n - 2; nt = n - 3
    for sw in range(nsweeps):
        for _ in range(nk + nt):
            st = steps[np.random.randint(steps.shape[0])]
            if np.random.randint(nk + nt) < nk:
                i = 1 + np.random.randint(nk)
                old = kap[i]; new = old + st * np.random.normal()
                dF = _local_k(kap, tau, h, i, new, lam, m, d, b, g, n) - _local_k(kap, tau, h, i, old, lam, m, d, b, g, n)
                tot += 1
                if np.random.random() < 1.0 / (1.0 + np.exp(min(dF / T, 700.0))):
                    kap[i] = new; _build(kap, tau, bond, rn)
                    if _clash(rn, n, d2):
                        kap[i] = old
                    else:
                        acc += 1
            else:
                i = 1 + np.random.randint(nt)
                old = tau[i]; new = old + st * np.random.normal()
                dF = _local_t(kap, a, i, new, d, b, c) - _local_t(kap, a, i, old, d, b, c)
                tot += 1
                if np.random.random() < 1.0 / (1.0 + np.exp(min(dF / T, 700.0))):
                    tau[i] = new; _build(kap, tau, bond, rn)
                    if _clash(rn, n, d2):
                        tau[i] = old
                    else:
                        acc += 1
        if sw >= nburn and (sw - nburn) % every == 0 and fr < nfr:
            _build(kap, tau, bond, r)
            frames[fr] = r; fr += 1
    return frames, acc / max(tot, 1)


def observables(frames, kmax=3):
    """Rg, helicity H, C(0) per frame (paper: Eq. 5, helicity box, Eq. 46-47)."""
    from .topology import crossings
    Rg = np.sqrt(((frames - frames.mean(1, keepdims=True)) ** 2).sum(2).mean(1))
    Hh = []; C0 = []
    for x in frames:
        k, t = kappa_tau(x)
        ok = ~np.isnan(k) & ~np.isnan(t)
        Hh.append(np.mean((k[ok] >= 1.29) & (k[ok] <= 1.78) & (t[ok] >= 0.5) & (t[ok] <= 1.27)))
        b, _ = crossings(t)
        C0.append(np.mean(b[1:-3]))
    return Rg, np.array(Hh), np.array(C0)


def binder(x):
    q2 = np.mean(x ** 2)
    return float(np.mean(x ** 4) / q2 ** 2 - 3) if q2 > 0 else float("nan")


def harmonic_control(M):
    """Control energy: independent harmonic restraints on each angle at the native value, with the same
    site-by-site stiffness as the Eq. 35 Hessian diagonal. No neighbour coupling, no kink structure."""
    k = M["kappa"]; t = M["tau"]; lam, m, d, b, c = (M[x] for x in ("lam", "m", "d", "b", "c"))
    Kk = 4 + 4 * lam * (3 * k ** 2 - m ** 2) + d * t ** 2 - 2 * b * t
    Kt = d * k ** 2 + c
    return dict(k0=k.copy(), t0=t.copy(), Kk=Kk, Kt=Kt)


@njit(cache=True)
def run_chain_harm(kap, tau, k0, t0, Kk, Kt, bond, T, nsweeps, nburn, every, delta, steps, seed):
    np.random.seed(seed)
    n = kap.shape[0]; d2 = delta * delta
    r = np.zeros((n, 3)); rn = np.zeros((n, 3))
    nfr = (nsweeps - nburn) // every
    frames = np.zeros((nfr, n, 3)); acc = 0; tot = 0; fr = 0
    nk = n - 2; nt = n - 3
    for sw in range(nsweeps):
        for _ in range(nk + nt):
            st = steps[np.random.randint(steps.shape[0])]
            if np.random.randint(nk + nt) < nk:
                i = 1 + np.random.randint(nk)
                old = kap[i]; new = old + st * np.random.normal()
                dF = 0.5 * Kk[i] * ((new - k0[i]) ** 2 - (old - k0[i]) ** 2)
                tot += 1
                if np.random.random() < 1.0 / (1.0 + np.exp(min(dF / T, 700.0))):
                    kap[i] = new; _build(kap, tau, bond, rn)
                    if _clash(rn, n, d2): kap[i] = old
                    else: acc += 1
            else:
                i = 1 + np.random.randint(nt)
                old = tau[i]; new = old + st * np.random.normal()
                dF = 0.5 * Kt[i] * ((new - t0[i]) ** 2 - (old - t0[i]) ** 2)
                tot += 1
                if np.random.random() < 1.0 / (1.0 + np.exp(min(dF / T, 700.0))):
                    tau[i] = new; _build(kap, tau, bond, rn)
                    if _clash(rn, n, d2): tau[i] = old
                    else: acc += 1
        if sw >= nburn and (sw - nburn) % every == 0 and fr < nfr:
            _build(kap, tau, bond, r); frames[fr] = r; fr += 1
    return frames, acc / max(tot, 1)
