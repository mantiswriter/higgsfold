"""High-level analysis of one protein chain.

A chain with breaks (consecutive Calpha atoms further apart than ``max_gap`` in
:func:`higgsfold.io.load_ca`) is split into unbroken segments. Each segment is analysed on its
own (kappa/tau, gauge, solitons, baseline) and the results are merged, so no angle or soliton
window ever spans a gap. Segments shorter than 4 residues cannot define a torsion and are
skipped (their per-residue rows are blank and they are listed in ``skipped_segments``).
"""
from __future__ import annotations
import json, os
import numpy as np
from .io import load_ca
from .geometry import kappa_tau, reconstruct, kabsch_rmsd
from .gauge import gauge_fix, gauge_fix_smooth, domain_walls, gauge_match
from .solitons import fit_solitons, NPAR, NAMES
from .topology import crossings, correlation, folding_index

HELIX = ((1.29, 1.78), (0.5, 1.27))      # kappa, tau ranges from Phys. Rev. E 111, 024406
BOUND_TOL = 0.01                         # a parameter within 1% of its range of a bound counts as "at bound"
KMAX = 20

def landscape_label(k, t):
    """Coarse (kappa, tau) region label: H helix-like, E strand-like (tau ~ +-pi), '-' other."""
    if np.isnan(k) or np.isnan(t):
        return "-"
    if HELIX[0][0] <= k <= HELIX[0][1] and HELIX[1][0] <= t <= HELIX[1][1]:
        return "H"
    if abs(abs(t) - np.pi) < 0.8 and 0.7 <= k <= 1.4:
        return "E"
    return "-"

def split_segments(n, breaks):
    """(start, stop) pairs of unbroken stretches; a break index i means a gap between i and i+1."""
    edges = [0] + [int(b) + 1 for b in breaks] + [n]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])]

def _params_at_bounds(sols):
    """Count fitted soliton parameters sitting on a bound (position parameters excluded)."""
    hit = tot = 0
    for s in sols:
        lo, hi = map(np.array, s.bounds())
        pos = {0} if s.model == "paper" else {0, 5}
        for j, (p, a, b) in enumerate(zip(s.params, lo, hi)):
            if j in pos:
                continue
            tot += 1
            hit += int(p - a < BOUND_TOL * (b - a) or b - p < BOUND_TOL * (b - a))
    return hit, tot

def _segment(ca, wall_penalty, gauge, fit, model, target, refine_global, global_nfev, baseline, verbose):
    n = len(ca)
    kappa, tau = kappa_tau(ca)
    bonds = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    check = kabsch_rmsd(reconstruct(np.nan_to_num(kappa), np.nan_to_num(tau), bonds), ca)
    gfun = gauge_fix_smooth if gauge == "smooth" else gauge_fix
    kg, tg, flips = gfun(kappa, tau, wall_penalty)
    walls = domain_walls(kg)
    b, kind = crossings(tau)
    g, _ = folding_index(tau)
    out = {"n": n, "kappa": kappa, "tau": tau, "kg": kg, "tg": tg, "flips": flips, "walls": walls,
           "b": b, "kind": kind, "check": check,
           "valid": int(np.sum(~np.isnan(tau[:-1]) & ~np.isnan(tau[1:]))),
           "Gamma": g, "n_infl": int(np.sum(kappa < 0.5))}
    if fit:
        sols, mk, mt, mca, rmsd = fit_solitons(ca, kg, tg, walls, model=model, target=target,
                                               refine_global=refine_global, global_nfev=global_nfev,
                                               verbose=verbose)
        km, tm = kappa_tau(mca)
        out["kappa_model"], out["tau_model"] = gauge_match(km, tm, kg, tg)
        out.update(sols=sols, model_ca=mca, rmsd=rmsd)
        out["bound_hit"], out["bound_tot"] = _params_at_bounds(sols)
        if baseline:
            from .baseline import fit_baseline
            out["baseline"] = fit_baseline(ca, kg, tg, NPAR[model] * len(sols), global_nfev=global_nfev)
    return out

def analyze_loaded(s, wall_penalty=1.0, gauge="smooth", fit=True, model="two-field", target=1.0,
                   refine_global=True, global_nfev=1500, baseline=False, verbose=False):
    """Analyse a structure dict as returned by :func:`higgsfold.io.load_ca`."""
    ca = np.asarray(s["ca"], float); N = len(ca)
    segs = split_segments(N, s["breaks"])
    parts, skipped = [], []
    for a, z in segs:
        if z - a < 4:
            skipped.append([a, z - 1]); parts.append((a, z, None)); continue
        parts.append((a, z, _segment(ca[a:z], wall_penalty, gauge, fit, model, target, refine_global,
                                      global_nfev, baseline, verbose)))
    done = [(a, z, p) for a, z, p in parts if p is not None]
    if not done:
        raise ValueError("no unbroken segment of at least 4 residues")

    def cat(key, fill, dtype=float):
        arr = np.full(N, fill, dtype=dtype)
        for a, z, p in done:
            arr[a:z] = p[key]
        return arr

    kappa, tau = cat("kappa", np.nan), cat("tau", np.nan)
    kg, tg = cat("kg", np.nan), cat("tg", np.nan)
    flips = cat("flips", False, bool); kind = cat("kind", "", object)
    walls = np.concatenate([a + p["walls"] for a, z, p in done]).astype(int)
    w = np.array([p["n"] for a, z, p in done], float)
    check = float(np.sqrt(np.sum(w * np.array([p["check"] for a, z, p in done]) ** 2) / w.sum()))
    bsum = sum(int(p["b"].sum()) for a, z, p in done); valid = sum(p["valid"] for a, z, p in done)
    gamma = float(sum(p["Gamma"] for a, z, p in done))
    # correlation function: pool numerators and denominators over segments
    if len(done) == 1:
        p = done[0][2]; C = correlation(p["b"][1:-3], KMAX)
    else:
        num = np.zeros(KMAX + 1); den = np.zeros(KMAX + 1)
        for a, z, p in done:
            bb = p["b"][1:-3].astype(float); m = len(bb)
            for k in range(min(KMAX, m - 1) + 1):
                num[k] += np.sum(bb[: m - k] * bb[k:]); den[k] += m - k
        C = num / np.maximum(den, 1)
    res = {
        "name": s["name"], "chain": s["chain"], "n_residues": N, "sequence": s["sequence"],
        "chain_breaks": s["breaks"], "n_segments": len(segs), "skipped_segments": skipped,
        "reconstruction_check_rmsd": check, "gauge": gauge, "wall_penalty": wall_penalty,
        "n_domain_walls": int(len(walls)),
        "topology": {"n_flattening": int(np.sum(kind == "flattening")), "n_branch": int(np.sum(kind == "branch")),
                     "C0": float(bsum / max(valid, 1)), "Gamma": gamma, "folding_index": gamma / np.pi,
                     "n_inflection_like": int(sum(p["n_infl"] for a, z, p in done))},
        "C_k": C.tolist(),
    }
    per = {"resnum": np.asarray(s["resnum"]), "aa": list(s["sequence"]), "kappa": kappa, "tau": tau,
           "kappa_gauged": kg, "tau_gauged": tg, "z2_flip": flips.astype(int), "crossing": kind,
           "region": [landscape_label(k, t) for k, t in zip(kappa, tau)],
           "domain_wall": np.isin(np.arange(N), walls).astype(int)}
    if fit:
        solitons, idx = [], 0
        for si, (a, z, p) in enumerate(done):
            for x in p["sols"]:
                d = x.asdict(); d["index"] = idx; d["segment"] = si
                d["start"] += a; d["end"] += a
                for key in ("s", "s_tau"):
                    if key in d:
                        d[key] += a
                solitons.append(d); idx += 1
        res["soliton_model"] = model
        res["n_solitons"] = len(solitons)
        res["n_solitons_at_domain_walls"] = int(sum(d["from_domain_wall"] for d in solitons))
        res["solitons"] = solitons
        res["n_soliton_parameters"] = NPAR[model] * len(solitons)
        res["soliton_window_rmsd_median"] = float(np.median([d["rmsd_local"] for d in solitons]))
        res["soliton_model_rmsd"] = float(np.sqrt(np.sum(w * np.array([p["rmsd"] for a, z, p in done]) ** 2) / w.sum()))
        hit = sum(p["bound_hit"] for a, z, p in done); tot = sum(p["bound_tot"] for a, z, p in done)
        res["params_at_bounds"] = {"count": int(hit), "of": int(tot), "fraction": hit / max(tot, 1)}
        per["kappa_model"] = cat("kappa_model", np.nan); per["tau_model"] = cat("tau_model", np.nan)
        res["_model_ca"] = np.vstack([p["model_ca"] for a, z, p in done])
        if baseline:
            res["baseline"] = {
                "kind": "cosine basis, same parameter count and refinement budget",
                "n_params": int(sum(p["baseline"]["n_params"] for a, z, p in done)),
                "rmsd": float(np.sqrt(np.sum(w * np.array([p["baseline"]["rmsd"] for a, z, p in done]) ** 2) / w.sum()))}
    res["_per_residue"] = per; res["_ca"] = ca
    return res

def analyze(source, chain=None, wall_penalty=1.0, gauge="smooth", fit=True, model="two-field",
            target=1.0, refine_global=True, global_nfev=1500, baseline=False, verbose=False):
    return analyze_loaded(load_ca(source, chain), wall_penalty, gauge, fit, model, target,
                          refine_global, global_nfev, baseline, verbose)

def write_outputs(res, outdir):
    import csv
    os.makedirs(outdir, exist_ok=True)
    per = res["_per_residue"]; keys = list(per.keys())
    with open(os.path.join(outdir, f"{res['name']}_{res['chain']}_per_residue.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for i in range(res["n_residues"]):
            row = []
            for k in keys:
                v = per[k][i]
                row.append(round(float(v), 4) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if isinstance(v, (float, np.floating)) else v))
            w.writerow(row)
    clean = {k: v for k, v in res.items() if not k.startswith("_")}
    with open(os.path.join(outdir, f"{res['name']}_{res['chain']}_summary.json"), "w") as f:
        json.dump(clean, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if "_model_ca" in res:      # with chain breaks each segment is rebuilt in its own frame
        with open(os.path.join(outdir, f"{res['name']}_{res['chain']}_soliton_model_CA.pdb"), "w") as f:
            for i, (x, y, z) in enumerate(res["_model_ca"]):
                f.write("ATOM  %5d  CA  GLY A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n" % (i + 1, i + 1, x, y, z))
            f.write("END\n")
    return outdir
