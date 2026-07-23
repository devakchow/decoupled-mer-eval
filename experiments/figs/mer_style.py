"""Shared IEEE-figure style + artifact loaders for the MER/SPL paper.

Every figure and table in the paper is generated from the result JSONs by the
scripts in this directory; no number is ever typed by hand. This module is the
single place that (a) fixes the plotting style to IEEE conventions and (b) loads
the artifacts, so a single provenance change propagates everywhere.

IEEE compliance notes (do not "simplify" away):
  * pdf.fonttype / ps.fonttype = 42 embeds TrueType outlines. matplotlib's
    default is Type 3, which IEEE PDF eXpress REJECTS -> desk return. Verified
    post-hoc by tools/check_pdf_fonts.py.
  * Serif text + STIX mathtext matches IEEEtran (Times-like).
  * Column geometry: single column 3.5 in, double column 7.16 in (IEEEtran
    \\columnwidth / \\textwidth at 10pt, two-column).
  * Lines carry BOTH colour and dash/marker so the figure survives greyscale
    printing (IEEE is still printed mono).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- geometry (inches) ----------------------------------------------------- #
COL_SINGLE = 3.5
COL_DOUBLE = 7.16

# --- per-system visual identity (colour + dash + marker, greyscale-safe) --- #
SYSTEMS = ("polytune", "laddersym_unprompted", "laddersym_prompted")
LABEL = {
    "polytune": "Polytune",
    "laddersym_unprompted": "LadderSym (unprompted)",
    "laddersym_prompted": "LadderSym (prompted)",
}
COLOR = {
    "polytune": "#1b1b1b",
    "laddersym_unprompted": "#3b6fb0",
    "laddersym_prompted": "#b03b52",
}
DASH = {
    "polytune": (0, ()),
    "laddersym_unprompted": (0, (4, 1.5)),
    "laddersym_prompted": (0, (1, 1)),
}
MARKER = {"polytune": "o", "laddersym_unprompted": "s", "laddersym_prompted": "^"}

# result-file basenames per system
_STEM = {
    "polytune": "A_polytune_maestro",
    "laddersym_unprompted": "B_laddersym_maestro_unprompted",
    "laddersym_prompted": "B_laddersym_maestro_prompted",
}

HERE = os.path.dirname(os.path.abspath(__file__))
EXPD = os.path.dirname(HERE)
GIL = os.path.join(EXPD, "results", "gilbreth")


def apply_style() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })


# --- artifact loaders ------------------------------------------------------ #
def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sweep(system: str) -> Dict[str, List[float]]:
    """tau grid + point/CI for HM and Loc-F, sorted by tau, from bootstrap."""
    d = _load(os.path.join(GIL, f"{_STEM[system]}_strict_eps05.json"))
    b = d["bootstrap"]
    taus = sorted(b, key=lambda k: float(k))
    return {
        "tau_ms": [int(k) for k in taus],
        "hm": [b[k]["point_hm"] for k in taus],
        "hm_lo": [b[k]["hm_ci95"][0] for k in taus],
        "hm_hi": [b[k]["hm_ci95"][1] for k in taus],
        "loc": [b[k]["point_loc_f1"] for k in taus],
        "loc_lo": [b[k]["loc_f1_ci95"][0] for k in taus],
        "loc_hi": [b[k]["loc_f1_ci95"][1] for k in taus],
    }


def bridge_band(system: str) -> Tuple[float, float, int, int]:
    """Admissible HM band [0, hi] a shipped report alone permits, at 50 ms.

    hi = Xmax / (T + Xmax),  T = TP_m + TP_e,
    Xmax = min(FP_m, FN_e) + min(FP_e, FN_m).   (bridge_checks.py C11)
    Computed here from the shipped artifact so the figure cannot disagree with
    the number the gate checks.
    """
    sh = _load(os.path.join(GIL, f"{_STEM[system]}_shipped.json"))["shipped_50ms"]
    m, e = sh["missed"], sh["extra"]
    T = m["tp"] + e["tp"]
    xmax = min(m["fp"], e["fn"]) + min(e["fp"], m["fn"])
    hi = xmax / (T + xmax) if (T + xmax) else 0.0
    return 0.0, hi, T, xmax


def constructed_sweep() -> dict:
    """The exact non-identifiability witness (shipped F flat, HM rising)."""
    return _load(os.path.join(EXPD, "results", "nonidentifiability_sweep.json"))


def ablation() -> dict:
    return _load(os.path.join(GIL, "paired_prompted_vs_unprompted.json"))


def coco_census() -> Dict[str, dict]:
    out = {}
    for tag, fn in [
        ("polytune", "C_polytune_coco_track_census.json"),
        ("laddersym_prompted", "D_laddersym_coco_prompted_track_census.json"),
        ("laddersym_unprompted", "D_laddersym_coco_unprompted_track_census.json"),
    ]:
        out[tag] = _load(os.path.join(GIL, fn))
    return out
