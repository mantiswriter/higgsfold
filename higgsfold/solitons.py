"""Soliton (kink) decomposition of a C-alpha backbone.

Loops are described as kinks (dark solitons) of the discretized nonlinear Schroedinger /
Abelian Higgs-type model of Niemi and co-workers. Each soliton k, centred near site s, owns a
contiguous window of the chain. Two profile models are available:

  model="paper"      bond angle kink + torsion law (Krokhotin, Niemi & Peng, PRE 85, 031906 (2012);
                     Eq. 37 of PRE 111, 024406 (2025)):
        kappa_i = (m1 e^{c1(i-s)} - m2 e^{-c2(i-s)}) / (e^{c1(i-s)} + e^{-c2(i-s)})
        tau_i   = (a + b kappa_i^2) / (1 + d kappa_i^2)                       8 parameters

  model="two-field"  (default; an extension of this package) the torsion carries its own kink,
                     i.e. the phase of the gauge-invariant field phi = kappa exp(i int tau) has an
                     independent domain wall:
        kappa_i = kink(i; s,  m1, m2, c1, c2)
        tau_i   = kink(i; s', t1, t2, e1, e2)                                10 parameters
                     Motivation: inside loops tau varies while kappa is nearly constant, which no
                     function tau(kappa) can follow.

Fitting pipeline (fit_solitons):
  1. one soliton per Z2 domain wall (gauge from gauge.gauge_fix_smooth),
  2. adaptive placement: while some window is reproduced worse than `target` Angstrom (backbone
     RMSD of the window +-2 flanking sites after superposition), insert a soliton where the
     profile fit is worst; kinks need not change sign, so new solitons describe junctions
     without a domain wall,
  3. optional global refinement of all parameters against the full backbone, which removes the
     error that otherwise accumulates along the chain.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares
from .geometry import wrap, reconstruct_fast, kabsch_rmsd, superposed_residual

PI = np.pi

def kink(i, s, m1, m2, c1, c2):
    x = np.asarray(i, float) - s
    e1 = np.exp(np.clip(c1 * x, -50, 50)); e2 = np.exp(np.clip(-c2 * x, -50, 50))
    return (m1 * e1 - m2 * e2) / (e1 + e2)

def torsion_law(kappa, a, b, d):
    k2 = np.asarray(kappa) ** 2
    return (a + b * k2) / (1 + d * k2)

NPAR = {"paper": 8, "two-field": 10}
NAMES = {"paper": ["s", "m1", "m2", "c1", "c2", "a", "b", "d"],
         "two-field": ["s", "m1", "m2", "c1", "c2", "s_tau", "t1", "t2", "e1", "e2"]}

@dataclass
class Soliton:
    index: int
    start: int
    end: int
    model: str = "two-field"
    params: np.ndarray = field(default=None)
    rmsd_local: float = np.nan
    from_wall: bool = True

    @property
    def center(self):
        return float(self.params[0]) if self.params is not None else (self.start + self.end) / 2

    def profile(self, i):
        p = self.params
        k = kink(i, *p[:5])
        t = torsion_law(k, *p[5:8]) if self.model == "paper" else kink(i, *p[5:10])
        return k, wrap(t)

    def bounds(self):
        lo_s, hi_s = self.start - 0.5, self.end + 0.5
        if self.model == "paper":
            return ([lo_s, -PI, -PI, 0.02, 0.02, -10, -10, 0], [hi_s, PI, PI, 20, 20, 10, 10, 50])
        return ([lo_s, -PI, -PI, 0.02, 0.02, lo_s - 2, -2 * PI, -2 * PI, 0.02, 0.02],
                [hi_s, PI, PI, 20, 20, hi_s + 2, 2 * PI, 2 * PI, 20, 20])

    def asdict(self):
        d = {"index": self.index, "start": int(self.start), "end": int(self.end), "model": self.model,
             "from_domain_wall": bool(self.from_wall), "rmsd_local": float(self.rmsd_local)}
        d.update({k: float(v) for k, v in zip(NAMES[self.model], self.params)})
        return d

def tile(centers, n, model, wall_flags=None):
    centers = list(centers)
    order = np.argsort(centers); centers = [centers[k] for k in order]
    flags = [True] * len(centers) if wall_flags is None else [wall_flags[k] for k in order]
    sols = []
    for k, c in enumerate(centers):
        lo = 1 if k == 0 else int(np.floor((centers[k - 1] + c) / 2)) + 1
        hi = n - 2 if k == len(centers) - 1 else int(np.floor((c + centers[k + 1]) / 2))
        s = Soliton(index=k, start=lo, end=max(lo, hi), model=model, from_wall=flags[k])
        s._c0 = c
        sols.append(s)
    return sols

def model_profiles(sols, n):
    kap = np.full(n, np.nan); tau = np.full(n, np.nan)
    for s in sols:
        i = np.arange(s.start, s.end + 1)
        kap[i], tau[i] = s.profile(i)
    return kap, tau

def _starts(sol, kg, tg):
    """Initial parameter guesses from the gauge-fixed profiles."""
    c = sol._c0; i0, i1 = sol.start, sol.end
    left = slice(i0, max(i0 + 1, int(np.floor(c)) + 1)); right = slice(min(i1, int(np.ceil(c))), i1 + 1)
    kl, kr = np.nanmedian(kg[left]), np.nanmedian(kg[right])
    with np.errstate(all="ignore"):
        import warnings; warnings.simplefilter("ignore", RuntimeWarning)
        tl, tr = np.nanmedian(tg[left]), np.nanmedian(tg[right])
    kl = 1.2 if np.isnan(kl) else kl; kr = 1.2 if np.isnan(kr) else kr
    tl = 0.9 if np.isnan(tl) else tl; tr = 0.9 if np.isnan(tr) else tr
    out = []
    for cc in (0.4, 1.5, 5.0):
        if sol.model == "paper":
            for ab in ((tr, 0.0, 0.0), (0.0, 1.0, 0.5), (PI, 0.0, 0.0)):
                out.append([c, kr, -kl, cc, cc, *ab])
        else:
            for sh in (0.0, -1.5, 1.5):
                out.append([c, kr, -kl, cc, cc, c + sh, tr, -tl, cc, cc])
    return out

def _fit_window(sol, ca, bonds, kg, tg, n, max_nfev, flank=2):
    a, b = max(1, sol.start - flank), min(n - 2, sol.end + flank)
    idx = np.arange(a, b + 1); target = ca[a - 1:b + 2]; bl = bonds[a - 1:b + 1]
    kk = np.zeros(len(target)); tt = np.zeros(len(target))
    lo, hi = map(np.array, sol.bounds())
    def res(p):
        sol.params = p
        kk[1:-1], tt[1:-1] = sol.profile(idx)
        return superposed_residual(reconstruct_fast(kk, tt, bl), target)
    best = None
    for q in _starts(sol, kg, tg):
        q = np.clip(np.nan_to_num(np.array(q, float), nan=0.5), lo + 1e-6, hi - 1e-6)
        o = least_squares(res, q, bounds=(lo, hi), max_nfev=max_nfev)
        if best is None or o.cost < best.cost:
            best = o
    sol.params = best.x
    sol.rmsd_local = float(np.sqrt(2 * best.cost / len(target)))
    return sol

def _worst_site(sol, kg, tg, min_window):
    i = np.arange(sol.start, sol.end + 1)
    k, t = sol.profile(i)
    e = np.abs(np.nan_to_num(k - kg[i])) + np.abs(np.nan_to_num(wrap(t - tg[i])))
    e = np.convolve(e, np.ones(3) / 3, mode="same")
    ok = (i - sol.start >= min_window - 1) & (sol.end - i >= min_window - 1)
    return None if not ok.any() else float(i[ok][np.argmax(e[ok])]) + 0.5

def fit_solitons(ca, kappa_g, tau_g, walls, model="two-field", adaptive=True, target=1.0,
                 min_window=3, max_solitons=None, refine_global=True, max_nfev=200,
                 global_nfev=300, verbose=False):
    """Fit the multi-soliton model. Returns (solitons, model_kappa, model_tau, model_ca, rmsd)."""
    ca = np.asarray(ca, float); n = len(ca)
    bonds = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    max_solitons = max_solitons or max(2, n // 4)
    centers = [w + 0.5 for w in walls] or [n / 2]
    flags = [True] * len(centers) if len(walls) else [False]
    cache = {}
    while True:
        sols = tile(centers, n, model, flags)
        for s in sols:
            key = (round(s._c0, 3), s.start, s.end)
            if key in cache:
                s.params, s.rmsd_local = cache[key][0].copy(), cache[key][1]
            else:
                _fit_window(s, ca, bonds, kappa_g, tau_g, n, max_nfev)
                cache[key] = (s.params.copy(), s.rmsd_local)
        worst = sorted(sols, key=lambda s: -s.rmsd_local)
        if verbose:
            print(f"{len(sols)} solitons, worst window {worst[0].rmsd_local:.2f} A")
        if not adaptive or worst[0].rmsd_local <= target or len(sols) >= max_solitons:
            break
        site = None
        for s in worst:
            if s.rmsd_local <= target:
                break
            site = _worst_site(s, kappa_g, tau_g, min_window)
            if site is not None and min(abs(site - c) for c in centers) >= min_window - 0.5:
                break
            site = None
        if site is None:
            break
        centers.append(site); flags.append(False)
    for k, s in enumerate(sols):
        s.index = k

    def build(sols_):
        return reconstruct_fast(*model_profiles(sols_, n), bonds)

    if refine_global:
        npar = NPAR[model]
        lo = np.concatenate([s.bounds()[0] for s in sols]); hi = np.concatenate([s.bounds()[1] for s in sols])
        p0 = np.clip(np.concatenate([s.params for s in sols]), lo + 1e-6, hi - 1e-6)
        def res(p):
            for s, q in zip(sols, p.reshape(-1, npar)):
                s.params = q
            return superposed_residual(build(sols), ca)
        out = least_squares(res, p0, bounds=(lo, hi), max_nfev=global_nfev, x_scale="jac")
        for s, q in zip(sols, out.x.reshape(-1, npar)):
            s.params = q
    mca = build(sols)
    for s in sols:
        a, b = max(0, s.start - 2), min(n, s.end + 3)
        s.rmsd_local = kabsch_rmsd(mca[a:b], ca[a:b])
    mk, mt = model_profiles(sols, n)
    return sols, mk, mt, mca, kabsch_rmsd(mca, ca)
