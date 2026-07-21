"""Supplementary figure -- the exact constructed non-identifiability witness.

Corroborates Proposition 1 (whose statement + proof stay in the main text): a
family of corpora with an IDENTICAL shipped per-class report while HM sweeps
0 -> 0.34. Real measured HM sits inside this same range, so the property is not
a corner case. Kept in supplementary because the analytic bridge band (Fig. 2,
Table I) carries the point in the main text on real data.

Run: python figS_witness.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

import mer_style as S

S.apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    d = S.constructed_sweep()
    pts = d["points"]
    hm = [p["hm"] for p in pts]
    series = [
        ("$F_\\mathrm{miss}$", [p["f1_missed"] for p in pts], "#1b1b1b", (0, ())),
        ("$F_\\mathrm{extra}$", [p["f1_extra"] for p in pts], "#3b6fb0", (0, (4, 1.5))),
        ("$F_\\mathrm{wrong}$", [p["f1_wrong"] for p in pts], "#b03b52", (0, (1, 1))),
    ]
    fig, ax = plt.subplots(figsize=(S.COL_SINGLE, 2.1))
    for name, ys, c, dash in series:
        ax.plot(hm, ys, color=c, linestyle=dash, linewidth=1.0, label=name)
    ax.set_xlabel("hidden mass  $\\mathrm{HM}$ (constructed)")
    ax.set_ylabel("shipped per-class $F$")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(min(hm), max(hm))
    ax.legend(loc="center left", handlelength=1.8, bbox_to_anchor=(0.02, 0.32))
    ax.set_title("shipped $F$ invariant across the whole $\\mathrm{HM}$ range",
                 loc="left")
    ax.annotate("identical shipped report,\n$\\mathrm{HM}=0\\ \\rightarrow\\ %.2f$"
                % max(hm), xy=(max(hm) * 0.55, 0.80), fontsize=6.4,
                ha="center", va="center")
    fig.tight_layout(pad=0.4)
    out = os.path.join(HERE, "figS_witness.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out[:-4] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
