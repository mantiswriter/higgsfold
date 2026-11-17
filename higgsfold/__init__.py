"""higgsfold: soliton and local-topology analysis of protein backbones.

Implements the discrete Frenet frame / Z2 gauge / dark-soliton framework of A. J. Niemi and
co-workers (e.g. Begun, Chernodub, Molochkov & Niemi, Phys. Rev. E 111, 024406 (2025)).
This is an independent re-implementation, not the authors' code.
"""
from .geometry import kappa_tau, reconstruct, reconstruct_fast, kabsch_rmsd, stereographic
from .gauge import gauge_fix, gauge_fix_smooth, apply_flips, domain_walls
from .solitons import fit_solitons, kink, torsion_law
from .topology import crossings, correlation, folding_index, summary
from .analyze import analyze, analyze_loaded, write_outputs
__version__ = "0.3.0"
