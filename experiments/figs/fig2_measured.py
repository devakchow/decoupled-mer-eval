"""Figure 2 -- measured decoupled behaviour (double column, two panels).

(a) Measured HM(tau) for three published systems with per-piece bootstrap 95%
    bands, shown against the inner bound Prop. 2 derives from Polytune's own
    report counts (analytic bridge). The bound is not an upper limit on HM.
(b) Measured localization F(tau).

Numbers loaded from artifacts via mer_style; nothing hand-set.
Run: python fig2_measured.py
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

import mer_style as S

S.apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
TAUS = [50, 75, 100, 150, 200, 500]


def _tau_axis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(TAUS))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in TAUS]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(46, 560)


def _hm_lower(sysname):
    """The span's lower endpoint at tau = 50 ms: HM with the ambiguous and
    unfounded merged events set aside (HM_G in the letter). Previously this
    plotted a third, unlabelled convention -- the whole dominant cell moved to
    the diagonal -- which the letter never defined."""
    from make_tables import tide_bins
    return tide_bins(sysname)["hm"]


def panel_hm(ax) -> None:
    for sysname in S.SYSTEMS:
        sw = S.sweep(sysname)
        ax.fill_between(sw["tau_ms"], sw["hm_lo"], sw["hm_hi"],
                        color=S.COLOR[sysname], alpha=0.14, linewidth=0)
        ax.plot(sw["tau_ms"], sw["hm"], color=S.COLOR[sysname],
                linestyle=S.DASH[sysname], marker=S.MARKER[sysname],
                label=S.LABEL[sysname])
    # competing reading of the dominant cell, at the 50 ms operating point
    lows = [_hm_lower(x) for x in S.SYSTEMS]
    # open markers spread +-5% about 50 ms so the three do not overprint
    ax.scatter([47.5, 50.0, 52.6][:len(lows)], lows, s=22, facecolors="none",
               edgecolors=[S.COLOR[x] for x in S.SYSTEMS], linewidths=1.1,
               zorder=5, label="$\\mathrm{HM}_G$")
    # Prop. 1's band is deliberately NOT drawn here. It is an inner bound over
    # collapse-free configurations, while these curves are measured under the
    # collapse; sharing an axis with them asserts a containment the proposition
    # does not license, and no caption disclaimer undoes that visually. The
    # bound is stated in Prop. 1 and in Sec. III instead.
    _tau_axis(ax)
    ax.set_xlabel("onset tolerance  $\\tau$ (ms, log scale)")
    ax.set_ylabel("$\\mathrm{HM}(\\tau)$")
    ax.set_ylim(0.0, 0.46)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              fontsize=6.0, handlelength=1.5, labelspacing=0.22,
              borderpad=0.2, handletextpad=0.4, columnspacing=0.9,
              framealpha=0.95)


def panel_loc(ax) -> None:
    for sysname in S.SYSTEMS:
        sw = S.sweep(sysname)
        ax.fill_between(sw["tau_ms"], sw["loc_lo"], sw["loc_hi"],
                        color=S.COLOR[sysname], alpha=0.14, linewidth=0)
        ax.plot(sw["tau_ms"], sw["loc"], color=S.COLOR[sysname],
                linestyle=S.DASH[sysname], marker=S.MARKER[sysname])
    _tau_axis(ax)
    ax.set_xlabel("onset tolerance  $\\tau$ (ms, log scale)")
    ax.set_ylabel("$F(\\tau)$")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(S.COL_DOUBLE, 1.72))
    panel_hm(axes[0])
    panel_loc(axes[1])
    # reserve a band under the axes for the bare subfigure labels (FG-021:
    # "(a)"/"(b)" centered below each panel, 8 pt Times; descriptive
    # wording lives in the LaTeX caption)
    fig.tight_layout(w_pad=1.4, rect=(0, 0.085, 1, 1))
    for ax, lab in ((axes[0], "(a)"), (axes[1], "(b)")):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.015, lab, fontsize=8.0,
                 ha="center", va="bottom", color="black")
    out = os.path.join(HERE, "fig2_measured.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out[:-4] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
