"""Discrete Frenet frame geometry of a C-alpha backbone.

Conventions follow Begun, Chernodub, Molochkov & Niemi, Phys. Rev. E 111, 024406 (2025)
and earlier work by Niemi and co-workers:

    t_i = (r_{i+1} - r_i) / |r_{i+1} - r_i|                     (tangent, i = 0..N-2)
    b_i = t_{i-1} x t_i / |t_{i-1} x t_i|                         (binormal at vertex i = 1..N-2)
    n_i = b_i x t_i
    kappa_i = arccos(t_{i-1} . t_i)                               (bond angle at vertex i)
    tau_i   = sign((b_i x b_{i+1}) . t_i) * arccos(b_i . b_{i+1}) (torsion on link i -> i+1)

kappa_i is stored at vertex index i (1..N-2) and tau_i on link index i (1..N-3); arrays are
padded with NaN at the chain ends so every array has length N and is indexed by residue.
"""
from __future__ import annotations
import numpy as np

def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)

def frenet_frames(ca: np.ndarray):
    """Return tangents (N-1,3), binormals (N,3), normals (N,3) with NaN where undefined."""
    ca = np.asarray(ca, float)
    N = len(ca)
    t = _unit(np.diff(ca, axis=0))
    b = np.full((N, 3), np.nan)
    n = np.full((N, 3), np.nan)
    b[1:N-1] = _unit(np.cross(t[:-1], t[1:]))
    n[1:N-1] = np.cross(b[1:N-1], t[1:])
    return t, b, n

def kappa_tau(ca: np.ndarray):
    """Bond angles kappa (rad, [0, pi]) and torsions tau (rad, [-pi, pi)), length-N arrays."""
    ca = np.asarray(ca, float)
    N = len(ca)
    if N < 4:
        raise ValueError("need at least 4 C-alpha atoms")
    t, b, _ = frenet_frames(ca)
    kappa = np.full(N, np.nan)
    tau = np.full(N, np.nan)
    kappa[1:N-1] = np.arccos(np.clip(np.einsum("ij,ij->i", t[:-1], t[1:]), -1, 1))
    b0, b1 = b[1:N-2], b[2:N-1]
    cosang = np.clip(np.einsum("ij,ij->i", b0, b1), -1, 1)
    sgn = np.sign(np.einsum("ij,ij->i", np.cross(b0, b1), t[1:N-2]))
    sgn[sgn == 0] = 1.0
    tau[1:N-2] = sgn * np.arccos(cosang)
    tau = wrap(tau)
    return kappa, tau

def wrap(x):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi

def reconstruct(kappa, tau, bond=3.8, frame0=None):
    """Rebuild C-alpha coordinates from (kappa, tau) with the discrete Frenet transfer matrix.

    Works with signed kappa (Z2 gauge-transformed values), which is how the gauge invariance of
    the 3D chain is checked. `bond` may be a scalar or an (N-1,) array of virtual bond lengths.
    """
    kappa = np.asarray(kappa, float)
    tau = np.asarray(tau, float)
    N = len(kappa)
    bond = np.broadcast_to(np.asarray(bond, float), (N - 1,))
    if frame0 is None:
        t = np.array([0.0, 0.0, 1.0]); n = np.array([1.0, 0.0, 0.0]); b = np.cross(t, n)
    else:
        t, n, b = frame0
    r = np.zeros((N, 3))
    r[1] = r[0] + bond[0] * t
    # frame at vertex 1 is defined by (t_0 -> t_1) bending; use the transfer rule
    #   t_i = cos k_i t_{i-1} + sin k_i n'   with n' the normal of the incoming frame,
    # where the incoming frame is first rotated about t_{i-1} by the torsion of the previous link.
    for i in range(1, N - 1):
        if i >= 2:
            ta = tau[i - 1]
            n, b = np.cos(ta) * n + np.sin(ta) * b, -np.sin(ta) * n + np.cos(ta) * b
        k = kappa[i]
        t_new = np.cos(k) * t + np.sin(k) * n
        n_new = -np.sin(k) * t + np.cos(k) * n
        t, n = t_new, n_new
        r[i + 1] = r[i] + bond[i] * t
    return r

def kabsch_rmsd(a, b):
    """RMSD after optimal superposition (Kabsch)."""
    a = np.asarray(a, float) - np.mean(a, axis=0)
    b = np.asarray(b, float) - np.mean(b, axis=0)
    h = a.T @ b
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    m = np.diag([1, 1, d])
    rot = vt.T @ m @ u.T
    return float(np.sqrt(np.mean(np.sum((a @ rot.T - b) ** 2, axis=1))))

def stereographic(kappa, tau):
    """Map (kappa, tau) to the plane: x + i y = tan(kappa/2) exp(-i tau)."""
    z = np.tan(np.asarray(kappa) / 2) * np.exp(-1j * np.asarray(tau))
    return z.real, z.imag


try:                                    # optional acceleration
    from numba import njit as _njit
except Exception:                       # pragma: no cover
    def _njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]

@_njit(cache=True)
def _reconstruct_core(kappa, tau, bond):
    N = kappa.shape[0]
    r = np.zeros((N, 3))
    t0, t1, t2 = 0.0, 0.0, 1.0
    n0, n1, n2 = 1.0, 0.0, 0.0
    b0, b1, b2 = 0.0, 1.0, 0.0
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
    return r

def reconstruct_fast(kappa, tau, bond=3.8):
    kappa = np.ascontiguousarray(np.nan_to_num(np.asarray(kappa, float)))
    tau = np.ascontiguousarray(np.nan_to_num(np.asarray(tau, float)))
    bond = np.ascontiguousarray(np.broadcast_to(np.asarray(bond, float), (len(kappa) - 1,)), dtype=float)
    return _reconstruct_core(kappa, tau, bond)

def superposed_residual(model, target):
    """Coordinate residual vector after optimal superposition of model onto target."""
    A = model - model.mean(0); B = target - target.mean(0)
    u, _, vt = np.linalg.svd(A.T @ B); d = np.sign(np.linalg.det(vt.T @ u.T))
    return (A @ (vt.T @ np.diag([1.0, 1.0, d]) @ u.T).T - B).ravel()
