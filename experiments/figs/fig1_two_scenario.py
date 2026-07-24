"""Figure 1 -- real-data non-identifiability (two scenarios, lane layout).

Verified exact case (Polytune x MAESTRO, piece 01-03_R1_2014_..--4, t=178.95 s;
see verify_two_scenario.py, claim holds): the reference contains ONE inserted
note (extra @ 68); the system outputs TWO events, extra @ 68 plus missed @ 69
-- its encoding of a wrong note.

(a) As measured: the system pair is co-located with the reference note -- the
    mistake is found but misnamed (decoupled: localized, misclassified, HM = 1).
(b) Counterfactual: the extra stays on the reference note (correctly reported
    insertion); the missed @ 69 is an unrelated false-alarm deletion claim
    (HM = 0).

Both configurations produce the IDENTICAL published per-class report
(extra: 1 TP; missed: 1 FP) -- the single shared report box at the bottom is
the punchline. Box placement is schematic; onsets/pitches/classes are the
measured values (see verify_two_scenario.py for the exact numbers).

Design: two horizontal lanes per panel ("reference" / "system output"); note
events are labeled boxes, so pitch is a label rather than an axis and no
legend is needed. HM appears only in the caption (it is defined in Sec. III).

Run: python fig1_two_scenario.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle

import mer_style as S

S.apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))

C_EXTRA = "#3b6fb0"     # system extra (insertion claim) -- solid blue
C_MISS = "#c46a10"      # system missed (deletion claim) -- solid orange
C_REF = "#b03b52"       # reference -- red outline, near-white fill
C_BAND = "#f2f2f2"      # lane background band
C_GRAY = "#555555"      # headers / annotations / guides

# lane geometry (axes coords; xlim 0..1, ylim -0.16..1.04)
REF_BAND = (0.60, 0.90)
SYS_BAND = (0.00, 0.48)
BX0, BX1 = 0.05, 0.37   # x-span of the onset-time boxes
FARX0, FARX1 = 0.63, 0.95  # x-span of the relocated missed box in (b)
GUIDE_X = (BX0 + BX1) / 2.0


def box(ax, x0, x1, y0, y1, face, edge, label, tcol, lw=0.8):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=face,
                           edgecolor=edge, linewidth=lw, zorder=3))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, fontsize=7.0, ha="center",
            va="center", color=tcol, zorder=4)


def panel(ax, scenario):
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.16, 1.04)
    ax.set_axis_off()

    # lane bands + plain-language lane headers
    for (y0, y1) in (REF_BAND, SYS_BAND):
        ax.add_patch(Rectangle((0, y0), 1, y1 - y0, facecolor=C_BAND,
                               edgecolor="none", zorder=0))
    ax.text(0.0, REF_BAND[1] + 0.02, "reference (what happened)",
            fontsize=7.0, ha="left", va="bottom", color=C_GRAY, zorder=2)
    ax.text(0.0, SYS_BAND[1] + 0.02, "system output",
            fontsize=7.0, ha="left", va="bottom", color=C_GRAY, zorder=2)

    # dotted guide: the shared instant of the reference note
    ax.plot([GUIDE_X, GUIDE_X], [SYS_BAND[0], REF_BAND[1]], ls=":", lw=0.6,
            color="#aaaaaa", zorder=1)

    # reference lane (identical in both panels): ONE inserted note
    box(ax, BX0, BX1, 0.66, 0.84, "#fdf5f7", C_REF, "extra 68", C_REF, lw=1.1)

    # time arrow under the system lane
    ax.add_patch(FancyArrowPatch((0.0, -0.09), (1.0, -0.09), arrowstyle="->",
                                 mutation_scale=6, lw=0.6, color="#999999"))
    ax.text(0.5, -0.155, "time (schematic)", fontsize=7.0, ha="center",
            va="bottom", color="#999999")

    if scenario == "A":
        # system pair stacked at the SAME instant as the reference note
        box(ax, BX0, BX1, 0.26, 0.44, C_MISS, "#8a4a0b", "missed 69", "white")
        box(ax, BX0, BX1, 0.04, 0.22, C_EXTRA, "#2a5182", "extra 68", "white")
        # bracket tying the pair, opening toward the boxes
        bx = BX1 + 0.03
        ax.plot([bx, bx], [0.04, 0.44], lw=0.7, color=C_GRAY, zorder=3)
        ax.plot([bx - 0.015, bx], [0.44, 0.44], lw=0.7, color=C_GRAY, zorder=3)
        ax.plot([bx - 0.015, bx], [0.04, 0.04], lw=0.7, color=C_GRAY, zorder=3)
        ax.text(bx + 0.035, 0.24,
                "co-located pair:\nclaims a wrong note\n(played 68 for 69)",
                fontsize=7.0, ha="left", va="center", color="#333333",
                linespacing=1.35)
        # gloss the reference event in plain language (panel (a) only; the
        # reference lane is identical in (b))
        ax.text(BX1 + 0.065, 0.75, "one inserted note", fontsize=7.0,
                ha="left", va="center", color=C_GRAY, style="italic")
        ax.set_title("(a) as measured:\nfound the mistake, misnamed it",
                     loc="left", fontsize=7.5, pad=3)
    else:
        # extra stays on the reference instant; missed relocated far away
        box(ax, BX0, BX1, 0.05, 0.23, C_EXTRA, "#2a5182", "extra 68", "white")
        box(ax, FARX0, FARX1, 0.05, 0.23, C_MISS, "#8a4a0b", "missed 69",
            "white")
        ax.text((FARX0 + FARX1) / 2, 0.27, "unrelated\ndeletion claim",
                fontsize=7.0, ha="center", va="bottom", color="#333333",
                linespacing=1.35)
        ax.set_title("(b) counterfactual:\nsame report, nothing misnamed",
                     loc="left", fontsize=7.5, pad=3)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(S.COL_SINGLE, 2.45))
    fig.subplots_adjust(left=0.02, right=0.985, top=0.845, bottom=0.315,
                        wspace=0.22)
    panel(axes[0], "A")
    panel(axes[1], "B")

    # the punchline: ONE shared report box fed by BOTH panels
    fig.text(0.5, 0.105,
             "identical published report from (a) and (b):\n"
             "extra: 1 TP        missed: 1 FP",
             ha="center", va="center", fontsize=7.2, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7",
                       edgecolor="#444444", linewidth=0.8))
    for ax, xb in ((axes[0], 0.40), (axes[1], 0.60)):
        fig.add_artist(ConnectionPatch(
            xyA=(0.5, 0.0), coordsA="axes fraction", axesA=ax,
            xyB=(xb, 0.165), coordsB="figure fraction",
            arrowstyle="->", mutation_scale=6, lw=0.7, color="#444444"))

    out = os.path.join(HERE, "fig1_two_scenario.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out[:-4] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
