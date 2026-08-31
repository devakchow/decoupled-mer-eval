"""Figure 1 -- real-data non-identifiability (three behaviors, piano-roll layout).

Verified exact case (Polytune x MAESTRO, piece 01-03_R1_2014_..--4, t=178.95 s;
see verify_two_scenario.py, claim holds): the reference contains ONE inserted
note (extra @ 68); the system outputs TWO events, extra @ 68 plus missed @ 69
-- its encoding of a wrong note.

Three panels, one per system behavior at the mistake (introduced in v3 at Prof. Lu's
request for a figure that walks through the scenarios):

(a) Silent: the system makes no claim (report: extra 1 FN).
(b) Counterfactual: the extra stays on the reference note (correctly reported
    insertion); the missed @ 69 is an unrelated false-alarm deletion claim
    (HM = 0).
(c) As measured: the system pair is co-located with the reference note -- the
    mistake is found but misnamed (decoupled: localized, misclassified, HM = 1).

(b) and (c) produce the IDENTICAL published per-class report
(extra: 1 TP; missed: 1 FP) -- the single shared report box at the bottom is
the punchline; (a) is distinguishable but earns no credit either. Box
placement is schematic; onsets/pitches/classes are the measured values (see
verify_two_scenario.py for the exact numbers).

Design (piano roll, per Prof. Lu): pitch is a real y-axis with two pitch rows,
A4 (69) above G-sharp-4 (68); each pitch row is split into a "ref" (upper) and
"sys" (lower) sub-row (light rule between sub-rows, heavier rule between pitch
groups; ref sub-rows lightly shaded). Panels are stacked vertically and share
the schematic time axis. Note events are horizontal bars, so the y-axis
carries pitch and bars carry only the class label ("extra"/"missed"); no
legend is needed. HM appears only in the caption (it is defined in Sec. III).
Per IEEE graphics rules (FG-021) the subfigure labels are bare "(a)"/"(b)"/
"(c)" centered below each panel in 8 pt Times; all descriptive wording
("silent", "as measured") belongs in the LaTeX caption. In-figure type >= 8 pt
placed, body ~9 pt (FG-019/FG-020).

Run: python fig1_two_scenario.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

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
REF_END = 0.118         # ref bar ends earlier so the bracket grabs sys bars only
SYS_END = 0.135
FAR0, FAR1 = 0.205, 0.295   # relocated missed bar in (b)
ROW_SYS68, ROW_REF68, ROW_SYS69, ROW_REF69 = 0, 1, 2, 3


def bar(ax, x0, x1, row_y0, face, edge, label, tcol, lw=0.8):
    y0, h = row_y0 + 0.19, 0.62
    ax.add_patch(Rectangle((x0, y0), x1 - x0, h, facecolor=face,
                           edgecolor=edge, linewidth=lw, zorder=3))
    ax.text((x0 + x1) / 2, y0 + h / 2, label, fontsize=8.0, ha="center",
            va="center", color=tcol, zorder=4)


def panel(ax, scenario, last=False):
    """scenario: 'SILENT' (a), 'FAR' (b), 'MEASURED' (c)."""
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-0.95 if not last else -2.35, 4.05)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_bounds(0, 4)
    ax.set_xticks([])
    ax.set_yticks([1, 3])
    # single-line pitch labels keep the outer gutter collision-free at the
    # compressed panel height
    ax.set_yticklabels(["G$\\sharp$4 (68)", "A4 (69)"], fontsize=8.5)
    ax.tick_params(axis="y", pad=3, length=2.0)
    # ref/sys sub-row tags sit INSIDE the roll, in the empty strip left of
    # the first onset (bars start at ONSET=0.045)
    for y, t in ((3.5, "ref"), (2.5, "sys"), (1.5, "ref"), (0.5, "sys")):
        ax.text(0.006, y, t, fontsize=8.0, ha="left", va="center",
                color=C_GRAY, zorder=2)

    # roll furniture: shaded ref sub-rows, sub-row rules, pitch-group rule
    for y0 in (ROW_REF68, ROW_REF69):
        ax.add_patch(Rectangle((0, y0), X_MAX, 1, facecolor=C_REFBAND,
                               edgecolor="none", zorder=0))
    ax.hlines([0, 4], 0, X_MAX, lw=0.5, color="#bbbbbb", zorder=1)
    ax.hlines([1, 3], 0, X_MAX, lw=0.4, color="#cccccc", zorder=1)
    ax.hlines(2, 0, X_MAX, lw=0.9, color="#8a8a8a", zorder=1)

    # reference row (identical in all panels): ONE inserted note @ 68
    bar(ax, ONSET, REF_END, ROW_REF68, "#fdf5f7", C_REF, "extra", C_REF,
        lw=1.1)

    # bare subfigure label centered below the roll (FG-021); the bottom
    # panel first gets the shared time arrow, then its label further down
    if last:
        ax.add_patch(FancyArrowPatch((0.0, -0.42), (X_MAX, -0.42),
                                     arrowstyle="->", mutation_scale=6, lw=0.6,
                                     color="#999999"))
        ax.text(X_MAX / 2, -0.62, "time (schematic)", fontsize=8.0,
                ha="center", va="top", color="#999999")
        ax.text(X_MAX / 2, -1.95, "(c)", fontsize=8.0, ha="center",
                va="center", color="black")
    else:
        ax.text(X_MAX / 2, -0.60, "(a)" if scenario == "SILENT" else "(b)",
                fontsize=8.0, ha="center", va="center", color="black")

    if scenario == "SILENT":
        # dotted guide at the mistake's onset; sys sub-rows stay empty
        ax.plot([ONSET, ONSET], [0, 2], ls=":", lw=0.6, color="#aaaaaa",
                zorder=2)
        ax.text(0.150, 0.5, "no claim", fontsize=8.5, ha="left",
                va="center", color=C_GRAY, style="italic")
        return

    # system extra @ 68, co-located with the reference note in (b) and (c)
    bar(ax, ONSET, SYS_END, ROW_SYS68, C_EXTRA, "#2a5182", "extra", "white")

    if scenario == "MEASURED":
        # dotted guide: the shared onset, through all rows
        ax.plot([ONSET, ONSET], [0, 4], ls=":", lw=0.6, color="#aaaaaa",
                zorder=2)
        # system missed @ 69 at the SAME onset
        bar(ax, ONSET, SYS_END, ROW_SYS69, C_MISS, "#8a4a0b", "missed",
            "white")
        # bracket tying the two SYS bars (ref bar ends earlier, stays outside)
        bx = 0.150
        ax.plot([bx, bx], [0.5, 2.5], lw=0.7, color=C_GRAY, zorder=3)
        ax.plot([bx - 0.008, bx], [2.5, 2.5], lw=0.7, color=C_GRAY, zorder=3)
        ax.plot([bx - 0.008, bx], [0.5, 0.5], lw=0.7, color=C_GRAY, zorder=3)
        ax.text(bx + 0.014, 1.5,
                "claims one wrong note (68 for 69)",
                fontsize=8.5, ha="left", va="center", color="#333333",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    else:  # FAR
        # dotted guide only through the 68 group: extra still on the ref note
        ax.plot([ONSET, ONSET], [0, 2], ls=":", lw=0.6, color="#aaaaaa",
                zorder=2)
        # system missed @ 69 relocated far away; annotation shares its sub-row
        bar(ax, FAR0, FAR1, ROW_SYS69, C_MISS, "#8a4a0b", "missed", "white")
        ax.text(FAR0 - 0.014, 2.5, "unrelated deletion claim",
                fontsize=8.5, ha="right", va="center", color="#333333")


def main() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(S.COL_SINGLE, 2.52))
    fig.subplots_adjust(left=0.165, right=0.985, top=0.985, bottom=0.185,
                        hspace=0.30)
    panel(axes[0], "SILENT")
    panel(axes[1], "FAR")
    panel(axes[2], "MEASURED", last=True)

    # the punchline: ONE shared report box naming panels (b) and (c); no
    # connector arrows -- with stacked panels they would cross panel (c)
    fig.text(0.5, 0.052,
             "identical published report from (b) and (c):\n"
             "extra: 1 TP        missed: 1 FP",
             ha="center", va="center", fontsize=9.0, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7",
                       edgecolor="#444444", linewidth=0.8))

    out = os.path.join(HERE, "fig1_two_scenario.pdf")
    # pad must clear the report box's rounded frame stroke at the bottom edge
    fig.savefig(out, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out[:-4] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    print("wrote", out)


if __name__ == "__main__":
    main()
