"""Loading C-alpha coordinates from PDB / mmCIF files, PDB IDs or AlphaFold DB entries."""
from __future__ import annotations
import os, urllib.request
import numpy as np

def _fetch(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

def fetch(identifier: str, outdir: str = ".") -> str:
    """Download a structure. 4-character IDs come from RCSB; 'AF-<UniProt>' or a UniProt
    accession (6-10 chars) comes from the AlphaFold Protein Structure Database."""
    os.makedirs(outdir, exist_ok=True)
    ident = identifier.strip()
    if len(ident) == 4:
        return _fetch(f"https://files.rcsb.org/download/{ident.upper()}.cif", os.path.join(outdir, f"{ident.upper()}.cif"))
    acc = ident[3:] if ident.upper().startswith("AF-") else ident
    acc = acc.split("-")[0].upper()
    return _fetch(f"https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.cif", os.path.join(outdir, f"AF-{acc}.cif"))

def load_ca(source: str, chain: str | None = None, model: int = 0, max_gap: float = 4.2):
    """Return dict with C-alpha coordinates, residue numbers, sequence and chain ID.

    Chain breaks (consecutive C-alpha further apart than `max_gap` Angstrom) are reported in
    'breaks'; downstream analysis treats each unbroken segment separately.
    """
    import gemmi
    path = source if os.path.exists(source) else fetch(source)
    st = gemmi.read_structure(path)
    st.remove_ligands_and_waters(); st.remove_hydrogens()
    mdl = st[model]
    chains = [c for c in mdl if (chain is None or c.name == chain)]
    if not chains:
        raise ValueError(f"chain {chain} not found; available: {[c.name for c in mdl]}")
    ch = max(chains, key=len) if chain is None else chains[0]
    ca, resnum, names = [], [], []
    for r in ch:
        a = r.find_atom("CA", "*")
        if a is None or gemmi.find_tabulated_residue(r.name) is None or not gemmi.find_tabulated_residue(r.name).is_amino_acid():
            continue
        ca.append(a.pos.tolist()); resnum.append(r.seqid.num); names.append(r.name)
    ca = np.array(ca)
    d = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    breaks = [int(i) for i in np.where(d > max_gap)[0]]
    return {"ca": ca, "resnum": np.array(resnum), "sequence": gemmi.one_letter_code(names),
            "chain": ch.name, "path": path, "breaks": breaks, "name": os.path.splitext(os.path.basename(path))[0]}

def load_trajectory_ca(topology: str, trajectory: str | None = None, selection: str = "name CA", step: int = 1):
    """C-alpha coordinates for every frame of an MD trajectory (requires MDAnalysis)."""
    import MDAnalysis as mda
    u = mda.Universe(topology, trajectory) if trajectory else mda.Universe(topology)
    sel = u.select_atoms(selection)
    return np.array([sel.positions.copy() for _ in u.trajectory[::step]])
