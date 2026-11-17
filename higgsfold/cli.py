"""Command line: higgsfold analyze 1UBQ --out results/"""
from __future__ import annotations
import argparse, os, json

def main(argv=None):
    ap = argparse.ArgumentParser(prog="higgsfold", description="Soliton / local-topology analysis of protein backbones")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyze a structure file, PDB ID or AlphaFold UniProt accession")
    a.add_argument("source"); a.add_argument("--chain"); a.add_argument("--out", default="higgsfold_out")
    a.add_argument("--wall-penalty", type=float, default=1.0); a.add_argument("--no-fit", action="store_true")
    a.add_argument("--model", choices=["two-field", "paper"], default="two-field")
    a.add_argument("--gauge", choices=["smooth", "torsion"], default="smooth")
    a.add_argument("--target", type=float, default=1.0, help="per-window RMSD target (A) for adaptive soliton placement")
    a.add_argument("--fast", action="store_true", help="short global refinement (~3x faster, less accurate)")
    a.add_argument("--baseline", action="store_true", help="also fit a generic cosine basis with the same number of parameters, for comparison")
    t = sub.add_parser("ensemble", help="perestroika statistics over an MD trajectory / NMR models")
    t.add_argument("topology"); t.add_argument("trajectory", nargs="?"); t.add_argument("--out", default="higgsfold_out")
    t.add_argument("--step", type=int, default=1)
    args = ap.parse_args(argv)
    from . import plots
    if args.cmd == "analyze":
        from .analyze import analyze, write_outputs
        res = analyze(args.source, args.chain, args.wall_penalty, gauge=args.gauge, fit=not args.no_fit,
                      model=args.model, target=args.target, global_nfev=300 if args.fast else 1500, baseline=args.baseline, verbose=True)
        write_outputs(res, args.out)
        stem = os.path.join(args.out, f"{res['name']}_{res['chain']}")
        plots.profiles(res, stem + "_profiles.png"); plots.landscape(res, stem + "_landscape.png")
        plots.correlation_plot(res["C_k"], stem + "_Ck.png")
        print(json.dumps({k: v for k, v in res.items() if not k.startswith("_") and k not in ("solitons", "C_k")}, indent=2, default=str))
    else:
        import numpy as np
        from .io import load_trajectory_ca
        from .ensemble import analyze_ensemble
        frames = load_trajectory_ca(args.topology, args.trajectory, step=args.step)
        e = analyze_ensemble(frames)
        os.makedirs(args.out, exist_ok=True)
        np.savetxt(os.path.join(args.out, "Q_per_frame.txt"), e["Q"])
        np.savetxt(os.path.join(args.out, "perestroika_frequency.txt"), e["perestroika_freq"])
        plots.correlation_plot(e["C_mean"], os.path.join(args.out, "Ck_mean.png"))
        print(json.dumps({"frames": len(frames), "Q_mean": float(e["Q"].mean()), "binder": e["binder"], "susceptibility": e["susceptibility"]}, indent=2))

if __name__ == "__main__":
    main()
