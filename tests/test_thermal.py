import os
import numpy as np
from higgsfold.io import load_ca
from higgsfold.thermal import (native_model, hessian_min_eig, harmonic_control, run_chain, run_chain_harm,
                               observables, _build, _clash)

EX = os.path.join(os.path.dirname(__file__), "..", "examples")
STEPS = np.array([0.02, 0.1, 0.4])

def _setup():
    ca = load_ca(os.path.join(EX, "1ABS.cif"))["ca"]
    return ca, native_model(ca)

def test_native_state_is_a_minimum_of_the_energy():
    ca, M = _setup()
    assert hessian_min_eig(M) > 0.1
    r = np.zeros_like(ca); _build(M["kappa"], M["tau"], M["bonds"], r)
    assert not _clash(r, len(r), 3.7 ** 2)

def test_low_temperature_stays_native_and_respects_hard_core():
    ca, M = _setup()
    kap, tau = M["kappa"].copy(), M["tau"].copy()
    fr, acc = run_chain(kap, tau, M["h"], M["a"], M["bonds"], M["lam"], M["m"], M["d"], M["b"], M["c"], M["g"],
                        0.002, 120, 20, 10, 3.7, STEPS, 1)
    Rg, H, C0 = observables(fr)
    assert 0.05 < acc < 0.6 and Rg.mean() < 17 and H.mean() > 0.7 and 0.10 < C0.mean() < 0.15
    assert not any(_clash(f, len(f), 3.7 ** 2) for f in fr)

def test_high_temperature_expands_and_loses_helicity():
    ca, M = _setup()
    kap, tau = M["kappa"].copy(), M["tau"].copy()
    fr, _ = run_chain(kap, tau, M["h"], M["a"], M["bonds"], M["lam"], M["m"], M["d"], M["b"], M["c"], M["g"],
                      5.0, 200, 100, 10, 3.7, STEPS, 2)
    Rg, H, C0 = observables(fr)
    assert Rg.mean() > 30 and H.mean() < 0.15 and C0.mean() > 0.3

def test_harmonic_control_runs():
    ca, M = _setup(); HC = harmonic_control(M)
    kap, tau = M["kappa"].copy(), M["tau"].copy()
    fr, acc = run_chain_harm(kap, tau, HC["k0"], HC["t0"], HC["Kk"], HC["Kt"], M["bonds"], 0.002, 60, 20, 10, 3.7, STEPS, 3)
    assert observables(fr)[0].mean() < 17
