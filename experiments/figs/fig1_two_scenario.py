"""Figure 1 -- real-data non-identifiability (two scenarios, piano-roll layout).

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

Design (piano roll, per Prof. Lu): pitch is a real y-axis with two pitch rows,
A4 (69) above G-sharp-4 (68); each pitch row is split into a "ref" (upper) and
"sys" (lower) sub-row (light rule between sub-rows, heavier rule between pitch
groups; ref sub-rows lightly shaded). Note events are horizontal bars on a
schematic time axis, so the y-axis carries pitch and bars carry only the class
label ("extra"/"missed"); no legend is needed. HM appears only in the caption
(it is defined in Sec. III).

Run: python fig1_two_scenario.py
"""
from __future__ import annotations

import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle

import mer_style as S

S.apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))

C_EXTRA = "#3b6fb0"     # system extra (insertion claim) -- solid blue
C_MISS = "#c46a10"      # system missed (deletion claim) -- solid orange
C_REF = "#b03b52"       # reference -- red outline, near-white fill
C_REFBAND = "#f5f5f5"   # light shading behind the ref sub-rows
C_GRAY = "#555555"      # tags / annotations

# piano-roll geometry (data coords). x = schematic time; y = pitch rows.
# Pitch group 68 occupies y [0, 2] (sys [0, 1], ref [1, 2]);
# pitch group 69 occupies y [2, 4] (sys [2, 3], ref [3, 4]).
X_MAX = 0.32
ONSET = 0.045           # shared onset of the co-located events
REF_END = 0.100         # ref bar ends earlier so the bracket grabs sys bars only
SYS_END = 0.115
FAR0, FAR1 = 0.215, 0.285   # relocated missed bar in (b)
ROW_SYS68, ROW_REF68, ROW_SYS69, ROW_REF69 = 0, 1, 2, 3


def bar(ax, x0, x1, row_y0, face, edge, label, tcol, lw=0.8):
    y0, h = row_y0 + 0.19, 0.62
    ax.add_patch(Rectangle((x0, y0), x1 - x0, h, facecolor=face,
                           edgecolor=edge, linewidth=lw, zorder=3))
    ax.text((x0 + x1) / 2, y0 + h / 2, label, fontsize=7.0, ha="center",
            va="center", color=tcol, zorder=4)


def panel(ax, scenario):
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-1.5, 4.05)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_bounds(0, 4)
    ax.set_xticks([])
    ax.set_yticks([1, 3])
    if scenario == "A":
        ax.set_yticklabels(["G$\\sharp$4\n(68)", "A4\n(69)"], fontsize=7.0)
        ax.tick_params(axis="y", pad=15, length=2.0)
        # ref/sys sub-row tags between the pitch labels and the spine
        for y, t in ((3.5, "ref"), (2.5, "sys"), (1.5, "ref"), (0.5, "sys")):
            ax.text(-0.012, y, t, fontsize=6.5, ha="right", va="center",
                    color=C_GRAY, clip_on=False)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=2.0)

    # roll furniture: shaded ref sub-rows, sub-row rules, pitch-group rule
    for y0 in (ROW_REF68, ROW_REF69):
        ax.add_patch(Rectangle((0, y0), X_MAX, 1, facecolor=C_REFBAND,
                               edgecolor="none", zorder=0))
    ax.hlines([0, 4], 0, X_MAX, lw=0.5, color="#bbbbbb", zorder=1)
    ax.hlines([1, 3], 0, X_MAX, lw=0.4, color="#cccccc", zorder=1)
    ax.hlines(2, 0, X_MAX, lw=0.9, color="#8a8a8a", zorder=1)

    # reference row (identical in both panels): ONE inserted note @ 68
    bar(ax, ONSET, REF_END, ROW_REF68, "#fdf5f7", C_REF, "extra", C_REF,
        lw=1.1)
    # system extra @ 68, co-located with the reference note in BOTH panels
    bar(ax, ONSET, SYS_END, ROW_SYS68, C_EXTRA, "#2a5182", "extra", "white")

    # time arrow below the roll
    ax.add_patch(FancyArrowPatch((0.0, -0.55), (X_MAX, -0.55),
                                 arrowstyle="->", mutation_scale=6, lw=0.6,
                                 color="#999999"))
    ax.text(X_MAX / 2, -0.92, "time (schematic)", fontsize=7.0, ha="center",
            va="top", color="#999999")

    if scenario == "A":
        # dotted guide: the shared onset, through all rows
        ax.plot([ONSET, ONSET], [0, 4], ls=":", lw=0.6, color="#aaaaaa",
                zorder=2)
        # system missed @ 69 at the SAME onset
        bar(ax, ONSET, SYS_END, ROW_SYS69, C_MISS, "#8a4a0b", "missed",
            "white")
        # bracket tying the two SYS bars (ref bar ends earlier, stays outside)
        bx = 0.128
        ax.plot([bx, bx], [0.5, 2.5], lw=0.7, color=C_GRAY, zorder=3)
        ax.plot([bx - 0.008, bx], [2.5, 2.5], lw=0.7, color=C_GRAY, zorder=3)
        ax.plot([bx - 0.008, bx], [0.5, 0.5], lw=0.7, color=C_GRAY, zorder=3)
        # white stroke keeps the label readable where it crosses the row rules
        ax.text(bx + 0.014, 1.5,
                "co-located pair:\nclaims a wrong note\n(played 68 for 69)",
                fontsize=7.0, ha="left", va="center", color="#333333",
                linespacing=1.35,
                path_effects=[pe.withStroke(linewidth=1.6,
                                            foreground="white")])
        ax.set_title("(a) as measured:\nfound the mistake, misnamed it",
                     loc="left", fontsize=7.5, pad=3)
    else:
        # dotted guide only through the 68 group: extra still on the ref note
        ax.plot([ONSET, ONSET], [0, 2], ls=":", lw=0.6, color="#aaaaaa",
                zorder=2)
        # system missed @ 69 relocated far away; annotation shares its sub-row
        bar(ax, FAR0, FAR1, ROW_SYS69, C_MISS, "#8a4a0b", "missed", "white")
        ax.text(FAR0 - 0.014, 2.5, "unrelated\ndeletion claim",
                fontsize=7.0, ha="right", va="center", color="#333333",
                linespacing=1.35)
        ax.set_title("(b) counterfactual:\nsame report, nothing misnamed",
                     loc="left", fontsize=7.5, pad=3)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(S.COL_SINGLE, 2.6))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.855, bottom=0.33,
                        wspace=0.18)
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
            xyB=(xb, 0.155), coordsB="figure fraction",
            arrowstyle="->", mutation_scale=6, lw=0.7, color="#444444"))

    out = os.path.join(HERE, "fig1_two_scenario.pdf")
    # pad must clear the report box's rounded frame stroke at the bottom edge
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out[:-4] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    print("wrote", out)


if __name__ == "__main__":
    main()
