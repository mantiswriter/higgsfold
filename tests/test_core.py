import numpy as np
from higgsfold.geometry import kappa_tau, reconstruct, kabsch_rmsd
from higgsfold.gauge import apply_flips, gauge_fix, domain_walls
from higgsfold.topology import crossings, correlation

def ideal_helix(n=30, r=2.3, rise=1.5, turn=np.deg2rad(100)):
    i = np.arange(n)
    return np.stack([r * np.cos(i * turn), r * np.sin(i * turn), rise * i], 1)

def random_chain(n=120, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n - 1, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.vstack([[0, 0, 0], np.cumsum(3.8 * v, axis=0)])

def test_reconstruction_exact():
    ca = random_chain()
    k, t = kappa_tau(ca)
    b = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    assert kabsch_rmsd(reconstruct(np.nan_to_num(k), np.nan_to_num(t), b), ca) < 1e-8

def test_z2_gauge_invariance():
    ca = random_chain(seed=1); k, t = kappa_tau(ca)
    b = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    rng = np.random.default_rng(2); fl = (rng.random(len(t)) < 0.4) & ~np.isnan(t)
    kp, tp = apply_flips(k, t, fl)
    assert kabsch_rmsd(reconstruct(np.nan_to_num(kp), np.nan_to_num(tp), b), ca) < 1e-8
    kg, tg, _ = gauge_fix(k, t)
    assert kabsch_rmsd(reconstruct(np.nan_to_num(kg), np.nan_to_num(tg), b), ca) < 1e-8

def test_helix_values_and_no_walls():
    k, t = kappa_tau(ideal_helix())
    assert 1.29 < np.nanmean(k) < 1.78 and 0.5 < np.nanmean(t) < 1.27   # helix region of the paper
    kg, tg, _ = gauge_fix(k, t)
    assert len(domain_walls(kg)) == 0

def test_random_walk_correlation():
    _, t = kappa_tau(random_chain(4000, seed=3))
    b, _ = crossings(t)
    C = correlation(b[1:-3], 3)
    assert abs(C[0] - 0.5) < 0.05 and abs(C[2] - 0.25) < 0.05   # ideal random walk limits

def test_smooth_gauge_invariance_and_fast_reconstruct():
    from higgsfold.gauge import gauge_fix_smooth
    from higgsfold.geometry import reconstruct_fast
    ca = random_chain(seed=5); k, t = kappa_tau(ca)
    b = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    kg, tg, _ = gauge_fix_smooth(k, t, 1.0)
    assert kabsch_rmsd(reconstruct_fast(kg, tg, b), ca) < 1e-8
    assert np.allclose(reconstruct_fast(k, t, b), reconstruct(np.nan_to_num(k), np.nan_to_num(t), b))

def test_soliton_fit_recovers_synthetic_kink_chain():
    from higgsfold.solitons import kink
    from higgsfold.geometry import reconstruct_fast
    from higgsfold.gauge import gauge_fix_smooth, domain_walls
    from higgsfold.solitons import fit_solitons
    i = np.arange(40)
    k = kink(i, 20, 1.55, -1.1, 1.2, 0.8); t = kink(i, 20.5, 0.9, -2.8, 1.0, 1.0)
    ca = reconstruct_fast(k, t, 3.8)
    kk, tt = kappa_tau(ca); kg, tg, _ = gauge_fix_smooth(kk, tt, 1.0)
    sols, *_, rmsd = fit_solitons(ca, kg, tg, domain_walls(kg), global_nfev=300)
    assert rmsd < 0.5
