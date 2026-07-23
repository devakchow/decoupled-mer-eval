"""Figure 2 -- measured decoupled behaviour (double column, two panels).

(a) Measured HM(tau) for three published systems with per-piece bootstrap 95%
    bands, shown against the band Polytune's own shipped report alone admits
    (analytic bridge). The admissible band dwarfs the between-system effect.
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


def panel_hm(ax) -> None:
    for sysname in S.SYSTEMS:
        sw = S.sweep(sysname)
        ax.fill_between(sw["tau_ms"], sw["hm_lo"], sw["hm_hi"],
                        color=S.COLOR[sysname], alpha=0.14, linewidth=0)
        ax.plot(sw["tau_ms"], sw["hm"], color=S.COLOR[sysname],
                linestyle=S.DASH[sysname], marker=S.MARKER[sysname],
                label=S.LABEL[sysname])
    _, hi, _, _ = S.bridge_band("polytune")
    ax.axhspan(0.0, hi, color="#999999", alpha=0.13, linewidth=0, zorder=0)
    ax.axhline(hi, color="#7a7a7a", lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax.annotate("Polytune shipped-report admissible: $[0,\\,%.2f]$" % hi,
                xy=(52, hi), xytext=(52, hi - 0.028),
                fontsize=7.5, va="top", ha="left", color="#2b2b2b")
    _tau_axis(ax)
    ax.set_xlabel("onset tolerance  $\\tau$ (ms)")
    ax.set_ylabel("hidden mass  $\\mathrm{HM}(\\tau)$")
    ax.set_ylim(0.0, 0.50)
    ax.legend(loc="lower left", handlelength=2.0, bbox_to_anchor=(0.0, 0.02))
    ax.set_title("(a) measured $\\mathrm{HM}$ vs. admissible band", loc="left")


def panel_loc(ax) -> None:
    for sysname in S.SYSTEMS:
        sw = S.sweep(sysname)
        ax.fill_between(sw["tau_ms"], sw["loc_lo"], sw["loc_hi"],
                        color=S.COLOR[sysname], alpha=0.14, linewidth=0)
        ax.plot(sw["tau_ms"], sw["loc"], color=S.COLOR[sysname],
                linestyle=S.DASH[sysname], marker=S.MARKER[sysname])
    _tau_axis(ax)
    ax.set_xlabel("onset tolerance  $\\tau$ (ms)")
    ax.set_ylabel("localization  $F(\\tau)$")
    ax.set_title("(b) measured localization $F$", loc="left")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(S.COL_DOUBLE, 2.3))
    panel_hm(axes[0])
    panel_loc(axes[1])
    fig.tight_layout(w_pad=1.4)
    out = os.path.join(HERE, "fig2_measured.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out[:-4] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    main()
