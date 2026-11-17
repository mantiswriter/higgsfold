"""Generic baseline for the soliton fit: same number of parameters, no soliton ansatz.

The soliton model is only interesting if it describes a backbone better than a generic
smooth basis with the *same number of free parameters*. This module fits the gauge-fixed
profiles kappa'(i) (signed) and tau'(i) (unwrapped) with a cosine basis, then refines all
coefficients against the whole Calpha backbone with the same least-squares objective and
iteration budget as the soliton fit, so the two RMSD values are directly comparable.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from .geometry import reconstruct_fast, kabsch_rmsd, superposed_residual


def _basis(n_pts, m):
    x = np.linspace(0.0, 1.0, n_pts)
    return np.cos(np.pi * np.outer(x, np.arange(m)))


def fit_baseline(ca, kappa_g, tau_g, n_params, global_nfev=1500):
    """Cosine-basis fit with `n_params` coefficients in total (split between kappa and tau).

    Returns dict(rmsd, rmsd_before_refine, n_params, model_ca).
    """
    ca = np.asarray(ca, float)
    n = len(ca)
    bonds = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    m = max(2, int(n_params) // 2)
    iv = np.arange(1, n - 1)                       # vertices carrying kappa
    kap = np.asarray(kappa_g, float)[iv]
    tau_full = np.asarray(tau_g, float)[iv]        # NaN at the last vertex
    ok = ~np.isnan(tau_full)
    tau_u = np.full(len(iv), np.nan)
    tau_u[ok] = np.unwrap(tau_full[ok])
    tau_u[~ok] = tau_u[ok][-1] if ok.any() else 0.0
    B = _basis(len(iv), m)
    ck = np.linalg.lstsq(B, np.nan_to_num(kap), rcond=None)[0]
    ct = np.linalg.lstsq(B, tau_u, rcond=None)[0]

    def build(p):
        k = np.zeros(n); t = np.zeros(n)
        k[iv] = B @ p[:m]; t[iv] = B @ p[m:]
        return reconstruct_fast(k, t, bonds)

    p0 = np.concatenate([ck, ct])
    before = kabsch_rmsd(build(p0), ca)
    out = least_squares(lambda p: superposed_residual(build(p), ca), p0,
                        max_nfev=global_nfev, x_scale="jac")
    mca = build(out.x)
    return {"rmsd": kabsch_rmsd(mca, ca), "rmsd_before_refine": before,
            "n_params": 2 * m, "model_ca": mca}
