"""Generate every LaTeX table in the paper from result artifacts (booktabs).

No number is typed by hand: each cell is pulled from a JSON via mer_style and
formatted here, so a re-run of the sweep regenerates the tables verbatim and the
verify_shipped gate can re-derive every printed value.

Tables:
  I   main results   -> tab_main.tex
  II  prompting ablation (paired) -> tab_ablation.tex
  III Coco fragility census -> tab_coco.tex

Run:  python make_tables.py
"""
from __future__ import annotations

import os

import mer_style as S

HERE = os.path.dirname(os.path.abspath(__file__))


def _f(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def tide_bins(system: str):
    """Partition merged predicted-wrong events by what their missed half names in
    the score: genuine (a score note the performance omits), ambiguous (a score note
    the performance played), unfounded (no such score note). Counts come from the
    cluster validation artifact; every cell is checksummed against the shipped
    confusion before use."""
    boot = S._load(os.path.join(HERE, '..', 'results', 'cluster', 'boot_bins.json'))
    cv = S._load(os.path.join(HERE, '..', 'results', 'cluster', 'collapse_validation.json'))
    sh = S._load(os.path.join(S.GIL, f"{S._STEM[system]}_shipped.json"))
    t = [x for x in sh['decoupled']['per_tau'] if x['tau_ms'] == 50][0]
    cs, M = t['confusion_sparse'], t['n_localized']
    cells = {k: cv[S._STEM[system]][k] for k in ('wrong->wrong', 'extra->wrong', 'missed->wrong')}
    for k, v in cells.items():
        assert cs[k] == v['n'], (system, k)
    genuine = sum(v['in_score_removed'] for v in cells.values())
    ambig = sum(v['in_score_correct'] for v in cells.values())
    unfounded = sum(v['absent_from_score'] for v in cells.values())
    off_total = sum(v for k, v in cs.items() if k.split('->')[0] != k.split('->')[1])
    off_predwrong = cells['extra->wrong']['n'] + cells['missed->wrong']['n']
    off_ret = (cells['extra->wrong']['in_score_removed']
               + cells['missed->wrong']['in_score_removed']
               + off_total - off_predwrong)
    denom = M - ambig - unfounded
    # charging U as misclassification instead of setting it aside: the other
    # endpoint of the span the data do not identify a convention between.
    unf_off = (cells['extra->wrong']['absent_from_score']
               + cells['missed->wrong']['absent_from_score'])
    hm_charge_u = (off_ret + unf_off) / (M - ambig)
    ci = boot[S._STEM[system]]['ci95']
    return dict(hm=off_ret / denom, unfounded=unfounded / M, ambiguous=ambig / M,
                genuine=genuine, numerator=off_ret, denominator=denom,
                hm_charging_u=hm_charge_u,
                hm_ci=ci['hm_g'], unfounded_ci=ci['unfounded'])


def hm_lower_reading(system: str) -> float:
    """HM with the (ref extra -> pred wrong) cell counted on the diagonal.

    The upper reading treats that cell as one localized-but-misnamed event; the
    lower reading treats it as a correctly localized extra plus a spurious
    missed claim. Both are computed from the same shipped confusion.
    """
    sh = S._load(os.path.join(S.GIL, f"{S._STEM[system]}_shipped.json"))
    t = sh["decoupled"]["per_tau"][0]
    assert t["tau_ms"] == 50
    cs = t["confusion_sparse"]
    off = sum(v for k, v in cs.items() if k.split("->")[0] != k.split("->")[1])
    return (off - cs.get("extra->wrong", 0)) / t["n_localized"]


def shipped_mean_error_f1(system: str) -> float:
    sh = S._load(os.path.join(S.GIL, f"{S._STEM[system]}_shipped.json"))
    return sh["shipped_50ms"]["_mean_error_f1"]


# --------------------------------------------------------------------------- #
# Row labels for the letter's tables; the unprompted configuration is named
# explicitly (letter terminology: "unprompted"/"prompted").
TLABEL = {
    "polytune": "Polytune",
    "laddersym_unprompted": "LadderSym (unprompted)",
    "laddersym_prompted": "LadderSym (prompted)",
}


def table_main() -> str:
    data = []
    for sysname in S.SYSTEMS:
        sw = S.sweep(sysname)
        i50 = sw["tau_ms"].index(50)
        hm, hm_lo, hm_hi = sw["hm"][i50], sw["hm_lo"][i50], sw["hm_hi"][i50]
        loc, loc_lo, loc_hi = sw["loc"][i50], sw["loc_lo"][i50], sw["loc_hi"][i50]
        _, band_hi, _, _ = S.bridge_band(sysname)
        tb = tide_bins(sysname)
        sf = shipped_mean_error_f1(sysname)
        data.append((sysname, sf, hm, hm_lo, hm_hi, tb,
                     loc, loc_lo, loc_hi, band_hi))
    best_hm = min(d[2] for d in data)   # lower HM is better
    best_loc = max(d[6] for d in data)  # higher F is better
    rows = []
    for (sysname, sf, hm, hm_lo, hm_hi, tb,
         loc, loc_lo, loc_hi, band_hi) in data:
        # No entry is marked best: merge rates differ across rows, so neither
        # column is a like-for-like comparison (see caption).
        hm_s, loc_s = _f(hm), _f(loc)
        rows.append(
            f"{TLABEL[sysname]} & {_f(sf)} & "
            f"{hm_s}~[{_f(hm_lo)},\\,{_f(hm_hi)}] & "
            f"{_f(tb['hm'])}~[{_f(tb['hm_ci'][0])},\\,{_f(tb['hm_ci'][1])}] & "
            f"{_f(100*tb['unfounded'],1)}~[{_f(100*tb['unfounded_ci'][0],1)},"
            f"\\,{_f(100*tb['unfounded_ci'][1],1)}]\\% & "
            f"{loc_s}~[{_f(loc_lo)},\\,{_f(loc_hi)}] \\\\")
    body = "\n".join(rows)
    return f"""% AUTO-GENERATED by figs/make_tables.py -- do not edit by hand.
\\begin{{table*}}[!t]
\\centering
\\caption{{Main Results on the MAESTRO-E Test Split ($\\tau=50$\\,ms,
$\\epsilon=50$\\,ms; Per-Piece Bootstrap 95\\% CIs, $10^3$ Resamples, $n=177$).
\\emph{{Published}} Is Our Replication's Pooled Mean of the Two Per-Class
Missed/Extra $F_1$ Under the Systems' Protocol, Not a Number [4] or [5]
Prints. All CIs Are Per-Piece Cluster Bootstraps Covering Piece-Sampling Only.
Merge Rates Differ Across Rows, So No Entry Is Marked Best, and [5] Labels the
Two LadderSym Rows \\emph{{Ladder}} and \\emph{{LadderSym}}. \\emph{{Raw}} Charges
Every Off-Diagonal Event; \\emph{{Genuine}} Charges One Only If Unmerged or Its
Deletion Claim Names an Omitted Score Note, Over $|M|-|A|-|U|$;
\\emph{{Unfounded}} Is $|U|/|M|$ Over \\emph{{All}} Localized Errors
(Sec.~\\ref{{sec:experiments}})}}
\\label{{tab:main}}
\\begin{{tabular}}{{@{{}}l c c c c c@{{}}}}
\\toprule
 & & \\multicolumn{{2}}{{c}}{{$\\mathrm{{HM}}\\!\\downarrow$}} & & \\\\
\\cmidrule(lr){{3-4}}
System & Published & raw & genuine & unfounded$\\!\\downarrow$ & $F\\!\\uparrow$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def table_ablation() -> str:
    a = S.ablation()
    n_nz_hm = a["hm"]["wilcoxon_n_nonzero"]
    n_nz_f = a["loc_f1"]["wilcoxon_n_nonzero"]
    def line(metric):
        m = a[metric]
        lo, hi = m["paired_bootstrap_ci95"]
        p = m["wilcoxon_p"]
        if p < 1e-12:
            p_s = "<10^{-12}"
        elif p < 1e-9:
            p_s = "<10^{-9}"
        else:
            p_s = f"{p:.1e}".replace("e", "\\!\\times\\!10^{") + "}"
        name = "$\\mathrm{HM}$" if metric == "hm" else "$F$"
        return (f"{name} & {_f(m['mean_a'])} & {_f(m['mean_b'])} & "
                f"${m['mean_diff']:+.3f}$~$[{lo:+.3f},\\,{hi:+.3f}]$ & "
                f"${p_s}$ & ${m['cliffs_delta']:.2f}$ \\\\")
    body = line("hm") + "\n" + line("loc_f1")
    return f"""% AUTO-GENERATED by figs/make_tables.py -- do not edit by hand.
\\begin{{table}}[!tb]
\\centering
\\caption{{Score Conditioning on LadderSym, Per-Piece Paired
($n={S.ablation()['n_used']}$, $\\tau=50$\\,ms). Prompt./Unpr.\\ Are
Per-Piece Means of Raw $\\mathrm{{HM}}$ and $F$ and So Differ From
Table~\\ref{{tab:main}}'s Pooled Values. $\\Delta$ Is Prompted$-$Unprompted
($10^4$-Resample Interval); $p$ From Two-Sided Wilcoxon on Nonzero Differences
($n={n_nz_hm}$, ${n_nz_f}$); $\\delta$ Is Cliff's Dominance on the Marginals.
The Contrast Is Scorer-Internal}}
\\label{{tab:ablation}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\begin{{tabular}}{{@{{}}l c c c c c@{{}}}}
\\toprule
 & Prompt. & Unpr. & $\\Delta$ [95\\% CI] & $p$ & $\\delta$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_coco() -> str:
    cen = S.coco_census()
    order = [("polytune", "Polytune"),
             ("laddersym_unprompted", "LadderSym (unpr.)"),
             ("laddersym_prompted", "LadderSym (pr.)")]
    rows = []
    for key, lab in order:
        c = cen[key]
        n = c["n_files"]
        n_s = f"{n:,}".replace(",", "{,}")
        h = {int(k): v for k, v in c["histogram"].items()}
        t1, t2, t3 = h.get(1, 0), h.get(2, 0), h.get(3, 0)
        under = c["n_offenders"]
        rows.append(f"{lab} & {n_s} & {100*t3/n:.1f} & {100*t2/n:.1f} & "
                    f"{100*t1/n:.1f} & {100*under/n:.1f} \\\\")
    body = "\n".join(rows)
    return f"""% AUTO-GENERATED by figs/make_tables.py -- do not edit by hand.
\\begin{{table}}[!t]
\\centering
\\caption{{CocoChorales Per-Stem Class-Track Census: The Multi-Instrument Output
Format Encodes Error Class Only by Track Position, so a Stem Is
Decodable Only When All Three Class Tracks Are Emitted (``3-trk'');
``Under-spec.'' Counts Stems With Fewer Than Three Tracks}}
\\label{{tab:coco}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\begin{{tabular}}{{@{{}}l c c c c c@{{}}}}
\\toprule
System & Stems & 3-trk\\,\\% & 2-trk\\,\\% & 1-trk\\,\\% & Under-spec.\\,\\% \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_null() -> str:
    """Supplementary Table S1: circular-shift null, from null_colo_*.json."""
    def _n(x: float) -> str:
        return f"{round(x):,}".replace(",", "{,}")

    rows = []
    for sysname in S.SYSTEMS:
        d = S._load(os.path.join(
            S.GIL, f"null_colo_{S._STEM[sysname]}.json"))
        ot, nt = d["observed_matched_total"], d["null_matched_total"]["mean"]
        oo, no = d["observed_off_diagonal"], d["null_off_diagonal"]["mean"]
        rows.append(f"{S.LABEL[sysname]} & {_n(ot)}/{_n(nt)} & "
                    f"{d['enrichment_total']:.1f} & {_n(oo)}/{_n(no)} & "
                    f"{d['enrichment_off']:.1f} \\\\")
    body = "\n".join(rows)
    n_perm = S._load(os.path.join(
        S.GIL, f"null_colo_{S._STEM['polytune']}.json"))["n_perm"]
    return f"""% AUTO-GENERATED by figs/make_tables.py -- do not edit by hand.
\\begin{{table}}[!t]
\\centering
\\caption{{Circular-Shift Null at $\\tau=50$\\,ms ({n_perm} Rotations per
Piece)}}
\\label{{tab:snull}}
\\footnotesize
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
Configuration & \\multicolumn{{2}}{{c}}{{matched total}} & \\multicolumn{{2}}{{c}}{{off-diagonal}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}}
 & obs./null mean & $\\times$ & obs./null mean & $\\times$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_confusion() -> str:
    """Supplementary Table S3: Polytune's full decoupled confusion at 50 ms.

    Orientation matches decoupled_scorer.DecoupledResult.confusion:
    rows = reference class, columns = predicted class.
    """
    d = S._load(os.path.join(S.GIL, f"{S._STEM['polytune']}_strict_eps05.json"))
    t = d["decoupled"]["per_tau"][0]
    assert t["tau_ms"] == 50
    conf = t["confusion"]
    classes = ["missed", "extra", "wrong"]

    def _n(x: int) -> str:
        return f"{x:,}".replace(",", "{,}")

    rows = []
    for r in classes:
        cells = " & ".join(_n(conf[r][c]) for c in classes)
        rows.append(f"{r} & {cells} \\\\")
    body = "\n".join(rows)
    return f"""% AUTO-GENERATED by figs/make_tables.py -- do not edit by hand.
\\begin{{table}}[!t]
\\centering
\\caption{{Polytune Decoupled Confusion at $\\tau=50$\\,ms (Rows: Reference
Class; Columns: Predicted Class)}}
\\label{{tab:sconf}}
\\footnotesize
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
Ref.\\ class & pred.\\ missed & pred.\\ extra & pred.\\ wrong \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def check_band_spread_claim() -> None:
    """Guard for the letter's claim that each system's admissible band is
    '$1.9$--$3.4\\times$' the between-system HM spread at 50 ms.

    Recomputed from the artifacts; fails the table build if the printed
    endpoints would be false (min ratio must round to 1.9 at one decimal,
    max ratio to 3.4).
    """
    hms, bands = [], []
    for sysname in S.SYSTEMS:
        sw = S.sweep(sysname)
        hms.append(sw["hm"][sw["tau_ms"].index(50)])
        bands.append(S.bridge_band(sysname)[1])
    spread = max(hms) - min(hms)
    ratios = [b / spread for b in bands]
    lo, hi = min(ratios), max(ratios)
    assert round(lo, 1) == 1.9 and round(hi, 1) == 3.4, (
        "letter claim 'band was $1.9$--$3.4\\times$ the spread' no longer "
        f"holds: band/spread ratios {[round(r, 2) for r in ratios]} "
        f"(spread {spread:.4f})")
    print(f"band/spread guard OK: ratios "
          f"{[round(r, 2) for r in ratios]} (spread {spread:.4f})")


def main() -> None:
    check_band_spread_claim()
    outdir = os.environ.get("MER_TABLE_OUTDIR", HERE)
    for name, fn in [(table_main(), "tab_main.tex"),
                     (table_ablation(), "tab_ablation.tex"),
                     (table_coco(), "tab_coco.tex"),
                     (table_null(), "tab_null.tex"),
                     (table_confusion(), "tab_confusion.tex")]:
        dest = os.path.join(outdir, fn)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(name)
        print("wrote", dest)


if __name__ == "__main__":
    main()
