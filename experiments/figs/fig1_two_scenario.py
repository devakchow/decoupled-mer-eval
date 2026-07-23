"""Figure 1 -- real-data non-identifiability (two scenarios, single column).

Verified exact case (Polytune x MAESTRO, piece 01-03_R1_2014_..--4, t=178.95 s;
see verify_two_scenario.py, claim holds): the system reports a substitution --
extra @ 68 plus missed @ 69 -- where the reference has only an inserted extra @ 68.

(a) As measured: the system's missed@69 is co-located with its extra@68, a
    localized substitution the decoupled measure books as HM = 1.
(b) Counterfactual: the SAME three events with missed@69 relocated far away (an
    unrelated deletion). The extra now matches the reference correctly; HM = 0.

Both configurations produce the IDENTICAL shipped per-class report
(extra: 1 TP; missed: 1 FP) -- so the shipped metric cannot tell them apart,
while HM distinguishes 1 from 0. Bar widths are schematic; onsets/pitches/classes
are the measured values.

Run: python fig1_two_scenario.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

import mer_style as S

S.apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))

C_EXTRA = "#3b6fb0"    # extra (insertion)
C_MISS = "#c46a10"     # missed (deletion)
C_REF = "#b03b52"      # reference outline
DUR = 0.10             # schematic bar width (s)


def note(ax, x, pitch, face, edge="none", hatch=None, lw=0.0, z=3, alpha=1.0):
    ax.add_patch(Rectangle((x, pitch - 0.30), DUR, 0.60, facecolor=face,
                           edgecolor=edge, hatch=hatch, linewidth=lw,
                           zorder=z, alpha=alpha))


def panel(ax, scenario):
    # reference: extra @ 68 (hollow red outline) -- same in both panels
    note(ax, 0.0045, 68, "none", edge=C_REF, hatch="////", lw=1.2, z=5)
    # prediction extra @ 68 (solid blue) -- same in both panels
    note(ax, 0.0, 68, C_EXTRA, z=4)

    if scenario == "A":
        # prediction missed @ 69 co-located -> substitution -> HM+1
        note(ax, 0.0, 69, C_MISS, z=4)
        ax.annotate("", xy=(0.05, 68.35), xytext=(0.05, 68.65),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#555555",
                                    shrinkA=0, shrinkB=0))
        ax.text(0.15, 69.62, "system: substitution\n(missed 69 + extra 68)",
                fontsize=7.0, va="center", ha="center")
        ax.text(0.15, 67.45, "ref: insertion (extra 68)", fontsize=7.0,
                va="center", ha="center", color=C_REF)
        ax.set_title("(a) as measured: $\\mathrm{HM}=1$", loc="left")
        ax.set_xlim(0.0, 0.30)
    else:
        # extra matches reference correctly; missed relocated far away
        ax.text(0.15, 67.45, "extra matches ref\n$\\Rightarrow$ correctly located",
                fontsize=7.0, va="center", ha="center", color=C_EXTRA)
        arr = FancyArrowPatch((0.115, 69), (0.28, 69), arrowstyle="->",
                              mutation_scale=7, lw=0.8, color=C_MISS,
                              linestyle="--")
        ax.add_patch(arr)
        note(ax, 0.0, 69, C_MISS, z=4, alpha=0.8, hatch="xxx",
             edge="#8a4a0b", lw=0.5)
        ax.text(0.15, 69.62, "missed 69 relocated\n(unrelated)", fontsize=7.0,
                va="center", ha="center", color=C_MISS)
        ax.set_title("(b) counterfactual: $\\mathrm{HM}=0$", loc="left")
        ax.set_xlim(0.0, 0.30)

    ax.set_ylim(67.2, 69.9)
    ax.set_yticks([68, 69])
    ax.set_xlabel("time (s, schematic)")
    ax.set_ylabel("MIDI pitch")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(S.COL_SINGLE, 2.5),
                             gridspec_kw=dict(wspace=0.42))
    panel(axes[0], "A")
    panel(axes[1], "B")

    legend = [
        Patch(facecolor="none", edgecolor=C_REF, hatch="////",
              label="reference: extra"),
        Patch(facecolor=C_EXTRA, label="prediction: extra"),
        Patch(facecolor=C_MISS, label="prediction: missed"),
    ]
    # legend above the panels, clear of the axis tick and axis labels
    fig.legend(handles=legend, loc="upper center", ncol=3, fontsize=7.0,
               handlelength=1.3, columnspacing=1.0, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.13, right=0.97,
                        wspace=0.42)
    out = os.path.join(HERE, "fig1_two_scenario.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out[:-4] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
