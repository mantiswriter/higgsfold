"""Tests on real structures (shipped in examples/), chain breaks, the baseline and the CLI."""
import os
import numpy as np
import pytest
from higgsfold.io import load_ca
from higgsfold.analyze import analyze, analyze_loaded, split_segments
from higgsfold.geometry import kappa_tau
from higgsfold.topology import crossings

EX = os.path.join(os.path.dirname(__file__), "..", "examples")
UBQ = os.path.join(EX, "1UBQ.cif")
MBN = os.path.join(EX, "1MBN.cif")

def test_ubiquitin_topology_without_fit():
    r = analyze(UBQ, fit=False)
    assert r["n_residues"] == 76 and r["n_segments"] == 1
    assert r["reconstruction_check_rmsd"] < 1e-8
    assert r["n_domain_walls"] == 7
    walls = np.where(r["_per_residue"]["domain_wall"] == 1)[0]
    # the alpha-helix (residues 23-34, indices 22-33) has no walls inside it
    assert not np.any((walls >= 22) & (walls <= 32))

def test_myoglobin_C0_matches_paper():
    """Phys. Rev. E 111, 024406 quotes C(0) ~ 0.13 for folded myoglobin."""
    r = analyze(MBN, fit=False)
    assert r["n_residues"] == 153
    assert 0.11 < r["topology"]["C0"] < 0.15

def test_crossings_are_stable_under_coordinate_noise():
    ca = load_ca(UBQ)["ca"]
    b0, _ = crossings(kappa_tau(ca)[1])
    rng = np.random.default_rng(0)
    flipped = np.mean([np.mean(crossings(kappa_tau(ca + rng.normal(0, 0.1, ca.shape))[1])[0] != b0)
                       for _ in range(10)])
    assert flipped < 0.08          # about 3% observed at 0.1 A

def _with_gap():
    s = load_ca(UBQ)
    keep = np.r_[0:30, 40:76]                      # delete residues 31-40
    t = dict(s)
    t["ca"] = s["ca"][keep]; t["resnum"] = s["resnum"][keep]
    t["sequence"] = "".join(np.array(list(s["sequence"]))[keep])
    d = np.linalg.norm(np.diff(t["ca"], axis=0), axis=1)
    t["breaks"] = [int(i) for i in np.where(d > 4.2)[0]]
    return t

def test_chain_break_is_never_spanned():
    t = _with_gap()
    assert t["breaks"] == [29]
    assert split_segments(len(t["ca"]), t["breaks"]) == [(0, 30), (30, 66)]
    r = analyze_loaded(t, fit=False)
    p = r["_per_residue"]
    assert r["n_segments"] == 2 and r["reconstruction_check_rmsd"] < 1e-8
    # no kappa on the two atoms that border the gap, no tau on the links touching it
    assert np.isnan(p["kappa"][29]) and np.isnan(p["kappa"][30])
    assert np.all(np.isnan(p["tau"][28:31]))
    assert not np.any((np.where(p["domain_wall"] == 1)[0] >= 28) & (np.where(p["domain_wall"] == 1)[0] <= 30))

def test_short_fragments_are_skipped_not_fatal():
    t = _with_gap(); t["breaks"] = t["breaks"] + [len(t["ca"]) - 3]
    r = analyze_loaded(t, fit=False)
    assert r["skipped_segments"] == [[len(t["ca"]) - 2, len(t["ca"]) - 1]]

def test_baseline_returns_comparable_rmsd():
    from higgsfold.baseline import fit_baseline
    from higgsfold.gauge import gauge_fix_smooth
    ca = load_ca(UBQ)["ca"]; k, t = kappa_tau(ca); kg, tg, _ = gauge_fix_smooth(k, t, 1.0)
    lo = fit_baseline(ca, kg, tg, 40, global_nfev=100)["rmsd"]
    hi = fit_baseline(ca, kg, tg, 120, global_nfev=100)["rmsd"]
    assert hi < lo < 8.0           # more parameters, better fit

def test_cli_writes_outputs(tmp_path):
    from higgsfold.cli import main
    main(["analyze", UBQ, "--no-fit", "--out", str(tmp_path)])
    names = os.listdir(tmp_path)
    assert any(n.endswith("_per_residue.csv") for n in names) and any(n.endswith("_summary.json") for n in names)

@pytest.mark.slow
def test_ubiquitin_fast_fit():
    r = analyze(UBQ, global_nfev=300, baseline=True)
    assert r["soliton_model_rmsd"] < 3.0
    assert r["n_soliton_parameters"] == 10 * r["n_solitons"]
    assert 0 <= r["params_at_bounds"]["fraction"] <= 1 and r["baseline"]["rmsd"] > 0
