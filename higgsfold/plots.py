"""Figures: (kappa, tau) profiles with soliton walls, stereographic landscape map, C(k)."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .geometry import stereographic

C_DATA, C_MODEL, C_WALL = "#1f5f99", "#d4772a", "#9a9a9a"

def profiles(res, path=None):
    p = res["_per_residue"]; x = np.asarray(p["resnum"])
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(x, p["kappa_gauged"], ".-", color=C_DATA, lw=1, ms=3, label="native (gauge-fixed)")
    ax[1].plot(x, p["tau_gauged"], ".-", color=C_DATA, lw=1, ms=3)
    if "kappa_model" in p:
        ax[0].plot(x, p["kappa_model"], "-", color=C_MODEL, lw=1.6, label="soliton model")
        tn = np.asarray(p["tau_gauged"], float); tmod = np.asarray(p["tau_model"], float)
        tplot = np.where(np.isnan(tn), tmod, tn + (np.mod(tmod - tn + np.pi, 2 * np.pi) - np.pi))  # same 2pi branch
        ax[1].plot(x, tplot, "-", color=C_MODEL, lw=1.6)
    for w in x[np.asarray(p["domain_wall"]) == 1]:
        for a in ax:
            a.axvline(w + 0.5, color=C_WALL, lw=0.6, ls=":")
    ax[0].set_ylabel("bond angle κ (rad)"); ax[1].set_ylabel("torsion τ (rad, mod 2π)"); ax[1].set_xlabel("residue")
    ax[0].axhline(0, color="k", lw=0.5); ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    ttl = f"{res['name']} chain {res['chain']}: {res['n_domain_walls']} Z2 domain walls (dotted)"
    if "soliton_model_rmsd" in res:
        ttl += f"; {res['n_solitons']} solitons, model Cα RMSD {res['soliton_model_rmsd']:.2f} Å"
    ax[0].set_title(ttl, fontsize=10, loc="left")
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    return fig

def landscape(res, path=None):
    p = res["_per_residue"]
    x, y = stereographic(p["kappa"], p["tau"])
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(x, y, c=np.asarray(p["resnum"]), cmap="viridis", s=14)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.set_xlabel("Re tan(κ/2)e^{-iτ}"); ax.set_ylabel("Im")
    for r in (np.tan(0.5), np.tan(0.75)):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=C_WALL, lw=0.6, ls="--"))
    fig.colorbar(sc, ax=ax, label="residue", shrink=0.8)
    ax.set_title("Stereographic (κ, τ) map", fontsize=10, loc="left")
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    return fig

def correlation_plot(C, path=None, ref=(0.13, 0.44)):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(range(len(C)), C, "o-", color=C_DATA, ms=4)
    ax.axhline(ref[0], color=C_MODEL, lw=0.8, ls="--"); ax.text(len(C) - 1, ref[0], " folded myoglobin (paper)", va="bottom", ha="right", fontsize=7)
    ax.axhline(ref[1], color=C_WALL, lw=0.8, ls="--"); ax.text(len(C) - 1, ref[1], " random coil (paper)", va="bottom", ha="right", fontsize=7)
    ax.set_xlabel("k"); ax.set_ylabel("C(k)"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    return fig
