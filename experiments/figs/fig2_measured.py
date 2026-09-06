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
    # left limit leaves clearance for the HM_G markers offset about 50 ms, so
    # none of them sits on the y-axis spine
    ax.set_xlim(41, 560)


def _tint(hexcolor: str, alpha: float = 0.14):
    """Opaque stand-in for `alpha` blending over white (IEEE FG-026 forbids
    live transparency in submitted graphics): 1 - alpha * (1 - c) per channel."""
    from matplotlib.colors import to_rgb
    return tuple(1.0 - alpha * (1.0 - c) for c in to_rgb(hexcolor))


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
                        facecolor=_tint(S.COLOR[sysname]),
                        edgecolor=_tint(S.COLOR[sysname]), linewidth=0)
        ax.plot(sw["tau_ms"], sw["hm"], color=S.COLOR[sysname],
                linestyle=S.DASH[sysname], marker=S.MARKER[sysname],
                markeredgecolor="white", markeredgewidth=0.4,
                label=S.LABEL[sysname])
    # the span at the 50 ms operating point: a thin vertical line from the raw
    # HM down to its adjudicated lower endpoint HM_G (open marker), one per
    # configuration; the three are spread about 50 ms so the open markers
    # (whose HM_G values differ by only 0.013) never touch
    lows = [_hm_lower(x) for x in S.SYSTEMS]
    xs = [46.3, 50.0, 54.0][:len(lows)]
    for x, lo, sysname in zip(xs, lows, S.SYSTEMS):
        raw50 = S.sweep(sysname)["hm"][0]
        ax.plot([x, x], [lo, raw50], color=S.COLOR[sysname], linewidth=0.8,
                linestyle="-", zorder=4)
    for x, lo, sysname in zip(xs, lows, S.SYSTEMS):
        ax.plot([x], [lo], linestyle="none", marker=S.MARKER[sysname],
                markersize=4.6, markerfacecolor="white",
                markeredgecolor=S.COLOR[sysname], markeredgewidth=1.0, zorder=5)
    # one legend entry for the open-marker convention (proxy handle)
    ax.plot([], [], linestyle="none", marker="o", markersize=4.6,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.0,
            label="open: $\\mathrm{HM}_G$")
    # Prop. 1's band is deliberately NOT drawn here. It is an inner bound over
    # collapse-free configurations, while these curves are measured under the
    # collapse; sharing an axis with them asserts a containment the proposition
    # does not license, and no caption disclaimer undoes that visually. The
    # bound is stated in Prop. 1 and in Sec. III instead.
    _tau_axis(ax)
    ax.set_xlabel("onset tolerance  $\\tau$ (ms, log scale)")
    ax.set_ylabel("$\\mathrm{HM}(\\tau)$")
    ax.set_ylim(0.0, 0.46)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    # the legend is drawn once, above both panels, by main(): an in-axes
    # legend at the IEEE 8 pt floor (FG-020) cannot avoid the curves


def panel_loc(ax) -> None:
    for sysname in S.SYSTEMS:
        sw = S.sweep(sysname)
        ax.fill_between(sw["tau_ms"], sw["loc_lo"], sw["loc_hi"],
                        facecolor=_tint(S.COLOR[sysname]),
                        edgecolor=_tint(S.COLOR[sysname]), linewidth=0)
        ax.plot(sw["tau_ms"], sw["loc"], color=S.COLOR[sysname],
                linestyle=S.DASH[sysname], marker=S.MARKER[sysname],
                markeredgecolor="white", markeredgewidth=0.4)
    _tau_axis(ax)
    ax.set_xlabel("onset tolerance  $\\tau$ (ms, log scale)")
    ax.set_ylabel("$F(\\tau)$")
    ax.set_ylim(0.62, 0.87)
    ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85])


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(S.COL_DOUBLE, 1.45))
    for ax in axes:
        # the axes background patch defaults to edgecolor "none", which the PDF
        # backend writes as a stroke alpha of 0 (an ExtGState IEEE FG-026 rejects)
        ax.patch.set_edgecolor("white")
    panel_hm(axes[0])
    panel_loc(axes[1])
    # one shared legend row above both panels, outside the axes, 8 pt
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.0,
               frameon=False, handlelength=1.6, columnspacing=1.4,
               handletextpad=0.5, bbox_to_anchor=(0.5, 1.0))
    # reserve a band under the axes for the bare subfigure labels (FG-021:
    # "(a)"/"(b)" centered below each panel, 8 pt Times; descriptive
    # wording lives in the LaTeX caption) and a band above for the legend
    fig.tight_layout(w_pad=1.4, rect=(0, 0.08, 1, 0.9))
    for ax, lab in ((axes[0], "(a)"), (axes[1], "(b)")):
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.015, lab, fontsize=8.0,
                 ha="center", va="bottom", color="black")
    out = os.path.join(HERE, "fig2_measured.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out[:-4] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    # IEEE FG-026: no live transparency may survive in the shipped PDF
    raw = open(out, "rb").read()
    import re as _re
    bad = [m.group(0) for m in _re.finditer(rb"/(?:CA|ca)\s+(0?\.\d+|0)\b", raw)
           if float(m.group(1)) < 1.0]
    if bad:
        raise SystemExit(f"transparency survived in {out}: {bad[:4]}")
    print("wrote", out)


if __name__ == "__main__":
    main()
