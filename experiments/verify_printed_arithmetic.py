"""Recompute every displayed formula of the SPL letter from the PRINTED numbers.

verify_shipped.py derives each printed number from a result artifact and pins
the prose to it (artifact -> prose). This script closes the other direction
(prose -> prose): it parses the numbers as they appear in the letter and the
supplement sources and checks that every relation the reader can form from
them holds -- matrix margins, the K = G + A + U partition, each hidden-mass
convention with its numerator and denominator, Proposition 1's interval from
the published per-class counts, the post-collapse totals, the filter gains,
the null-model ratios, the MAESTRO-EI recoveries, and the abstract's rounded
ranges. A number that reaches the page through an artifact but no longer
agrees with its neighbours fails here, deterministically, on every run.

Reads only proposal/spl_letter_v5.tex, proposal/spl_supplementary_v5.tex and
the two generated tables experiments/figs/tab_main.tex (Table I) and
experiments/figs/tab_null.tex (Table II) -- a table cell is a printed number
like any other, and Sec. IV moved most of its per-configuration counts into
those two tables. No artifact is consulted. Exit code 0 = every
relation holds; 1 = a relation failed or an anchor pattern was not found
(the check refuses to pass on a missing sentence).

Run: python experiments/verify_printed_arithmetic.py
"""
from __future__ import annotations

import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LETTER = os.path.join(ROOT, "proposal", "spl_letter_v5.tex")
SUPP = os.path.join(ROOT, "proposal", "spl_supplementary_v5.tex")
TAB_NULL = os.path.join(HERE, "figs", "tab_null.tex")
TAB_MAIN = os.path.join(HERE, "figs", "tab_main.tex")

PASSES: list[str] = []
FAILS: list[str] = []


def check(label: str, got, want, tol: float = 0.0) -> None:
    """Record a relation; numeric comparisons use an absolute tolerance
    equal to half a unit in the printed precision (default exact)."""
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        good = abs(got - want) <= tol + 1e-12
    else:
        good = got == want
    (PASSES if good else FAILS).append(f"{label}: got {got!r}, printed {want!r}")


def num(s: str) -> float:
    """'12{,}307' -> 12307.0 ; '0.293' -> 0.293 ; '24.1' -> 24.1"""
    return float(s.replace("{,}", "").replace(",", ""))


def grab(text: str, pattern: str, label: str) -> tuple[str, ...]:
    """Return the groups of the first match, or fail loudly and abort: an
    anchor sentence that has moved is itself a defect of the check."""
    m = re.search(pattern, text, re.S)
    if not m:
        FAILS.append(f"anchor not found: {label} /{pattern[:60]}.../")
        raise SystemExit(_report())
    return m.groups()


def triple(s: str) -> tuple[float, float, float]:
    a, b, c = s.split("/")
    return num(a), num(b), num(c)


def hull(vals, nd: int):
    """Outward-rounded range at nd decimals: every summarized value lies inside
    the printed range under any reading (lower end down, upper end up)."""
    f = 10 ** nd
    return math.floor(min(vals) * f) / f, math.ceil(max(vals) * f) / f


def _report() -> int:
    for p in PASSES:
        print("[PASS]", p)
    for f in FAILS:
        print("[FAIL]", f)
    print(f"\n{len(PASSES)} relations hold, {len(FAILS)} fail")
    print("ALL PRINTED-ARITHMETIC CHECKS PASSED" if not FAILS else "PRINTED-ARITHMETIC CHECKS FAILED")
    return 1 if FAILS else 0


def main() -> int:
    def read(path: str) -> str:
        with open(path, encoding="utf-8") as fh:
            return re.sub(r"\s+", " ", fh.read())

    tex, sup, tab, tabm = read(LETTER), read(SUPP), read(TAB_NULL), read(TAB_MAIN)
    N_ = r"([\d{,}]+)"
    F_ = r"([\d.]+)"

    # ---- Table II (tab_null.tex): one row per quantity, one column per
    # configuration. Sec. IV's per-configuration counts (the K partition, the
    # two merged cells, mu_p, both hidden-mass conventions, the null means and
    # the equal-pitch share) are printed here now, so the relations below read
    # their operands out of the table instead of out of the prose.
    def trow(label: str, cell: str, what: str) -> tuple[str, ...]:
        return grab(tab, re.escape(label) + (" & " + cell) * 3 + r" \\\\", what)

    OBS_ = N_ + r"~\(" + N_ + r"\)"          # "30{,}460~(3{,}804)"
    t_tot = trow("Matched total, obs.\\ (null)", OBS_, "Table II matched total")
    t_off = trow("Off-diagonal, obs.\\ (null)", OBS_, "Table II off-diagonal")
    t_ew = trow("extra$\\to$wrong (genuine)", OBS_, "Table II extra->wrong")
    t_mw = trow("missed$\\to$wrong (genuine)", OBS_, "Table II missed->wrong")
    G = tuple(num(x) for x in trow("$|G|$ genuine", N_, "Table II |G|"))
    A = tuple(num(x) for x in trow("$|A|$ ambiguous", N_, "Table II |A|"))
    U = tuple(num(x) for x in trow("$|U|$ unfounded", N_, "Table II |U|"))
    mp = tuple(num(x) for x in trow("$\\mu_p$ predicted-side merges", N_, "Table II mu_p"))
    hmg_tab = tuple(num(x) for x in trow("$\\mathrm{HM}_G$", F_, "Table II HM_G"))
    cu_vals = tuple(num(x) for x in trow("$\\mathrm{HM}_U$ (denom.\\ $|M|-|A|$)", F_, "Table II HM_U"))
    eq_share = tuple(num(x) for x in trow("Equal-pitch share (\\%)", F_, "Table II equal-pitch"))
    Ms = [num(t_tot[2 * i]) for i in range(3)]
    Ms_null = [num(t_tot[2 * i + 1]) for i in range(3)]
    offs = [num(t_off[2 * i]) for i in range(3)]
    offs_null = [num(t_off[2 * i + 1]) for i in range(3)]
    ew = [num(t_ew[2 * i]) for i in range(3)]
    ewg = [num(t_ew[2 * i + 1]) for i in range(3)]
    mw = [num(t_mw[2 * i]) for i in range(3)]
    mwg = [num(t_mw[2 * i + 1]) for i in range(3)]
    K = [G[i] + A[i] + U[i] for i in range(3)]

    # ---- Table I (tab_main.tex): Prop. 1's interval width and the realized
    # collapse-free hidden mass are its last two columns.
    TLABEL = ("Polytune", "LadderSym unpr.", "LadderSym pr.")
    widths, hm0, mloc, filt_f1 = [], [], [], []
    for lab in TLABEL:
        w, h, ml, ff = grab(tabm, re.escape(lab) + r" & .*? & " + F_ + " & " + F_ + " & " + F_
                            + " & " + F_ + r" \\\\", "Table I row " + lab)
        widths.append(num(w)); hm0.append(num(h))
        mloc.append(num(ml)); filt_f1.append(num(ff))

    # ---- Polytune's confusion matrix and its margins ----------------------
    _row = lambda first: ((r"N: & " if first else r"& ") + N_ + " & " + N_ + " & " + N_
                          + " & " + N_ + r"\\\\")
    g = grab(tex, _row(True) + " " + _row(False) + " " + _row(False)
             + r" \\cline\{2-5\} \\gamma': & " + N_ + " & " + N_ + " & " + N_ + r" & \\\\",
             "confusion array with row sums and reference totals")
    N = [[num(g[4 * i + j]) for j in range(3)] for i in range(3)]
    sums_disp = [num(g[4 * i + 3]) for i in range(3)]      # the printed Sigma column
    gam_disp = [num(g[12 + j]) for j in range(3)]          # the printed gamma' row
    M = sum(map(sum, N))
    off = M - sum(N[i][i] for i in range(3))
    K_col = sum(N[i][2] for i in range(3))
    outside_wrong = off - N[0][2] - N[1][2]

    # |K| itself is no longer printed: Table II gives its three parts, so the
    # partition is read as G+A+U (the per-configuration identity against the
    # adjudication cells is checked from the artifacts in verify_shipped).
    check("|K| = |G|+|A|+|U| (Polytune) = predicted-wrong column of N", K_col, K[0])
    for i in range(3):
        check(f"|K| does not exceed the matched total (configuration {i + 1})", K[i] <= Ms[i], True)
        check(f"genuine merged events include the two off-diagonal cells' genuine ones (configuration {i + 1})",
              G[i] >= ewg[i] + mwg[i], True)

    hm_num2, hm_den, hm_val, hm_num, hm_gen = grab(
        tex, r"\\mathrm\{HM\}_G=" + N_ + "/" + N_ + r"=([\d.]+)\$ for Polytune \(\$" + N_
        + r"\$ unmerged off-diagonal events plus \$(\d+)\$ genuine merged ones", "HM_G formula")
    check("1,173 = off-diagonal cells outside the predicted-wrong column", outside_wrong, num(hm_num))
    check("HM_G numerator 1,173 + 136 = 1,309", num(hm_num) + num(hm_gen), num(hm_num2))
    check("HM_G denominator = |M| - |A| - |U|", M - A[0] - U[0], num(hm_den))
    check("HM_G = 1,309 / 20,694", num(hm_num2) / num(hm_den), num(hm_val), 5e-4)
    check("136 genuine off-diagonal = Table II's 37 extra->wrong + 99 missed->wrong",
          ewg[0] + mwg[0], num(hm_gen))

    raw_val = grab(tex, r"Counting every off-diagonal match gives the raw \$([\d.]+)\$", "raw convention")[0]
    check("off-diagonal count of N = Table II's off-diagonal (Polytune)", off, offs[0])
    check("|M| = Table II's matched total (Polytune)", M, Ms[0])
    check("raw HM = 8,928 / 30,460", off / M, num(raw_val), 5e-4)
    lo, hi = grab(tex, r"the convention span \$([\d.]+)\$--\$([\d.]+)\$ for Polytune", "span")
    check("span endpoints = (HM_G, raw HM)", (num(lo), num(hi)), (num(hm_val), num(raw_val)))
    h_lo, h_hi = grab(tex, r"Polytune's hidden mass spans \$([\d.]+)\$--\$([\d.]+)\$ by convention", "convention-span heading")
    check("heading's span = the printed span endpoints", (num(lo), num(hi)), (num(h_lo), num(h_hi)))
    dom, dom_of = grab(tex, r"holds \$" + N_ + r"\$ of \$" + N_ + r"\$ off-diagonal events", "dominant cell")
    check("dominant cell = N[extra][wrong]", N[1][2], num(dom))
    check("dominant cell denominator = off-diagonal count", off, num(dom_of))
    d_lo, d_hi = grab(tex, r"which carries \$([\d.]+)\$--\$([\d.]+)\\%\$ of the raw hidden mass", "dominant share range")
    check("Polytune dominant-cell share 82.11% lies below the range's upper end", 100 * N[1][2] / off <= num(d_hi), True)
    # every printed multi-configuration range is rounded outward (lower end
    # down, upper end up) at the printed precision, so each value lies inside
    dom_sh = [100 * ew[i] / offs[i] for i in range(3)]
    check("dominant-cell share range = outward-rounded min/max of Table II's extra->wrong / off-diagonal",
          hull(dom_sh, 1), (num(d_lo), num(d_hi)))
    for i in range(3):
        check(f"dominant-cell share inside the printed range (configuration {i + 1})", num(d_lo) <= dom_sh[i] <= num(d_hi), True)
    # the letter prints Polytune's |A| and |U| beside HM_G's denominator
    a_l, u_l = grab(tex, r"\(\$\|A\|=" + N_ + r"\$, \$\|U\|=" + N_ + r"\$ for Polytune;", "letter |A|, |U|")
    check("letter's |A| (Polytune) = Table II's", A[0], num(a_l))
    check("letter's |U| (Polytune) = Table II's", U[0], num(u_l))

    # ---- row localization rates against the post-collapse reference totals -
    r1, r2, r3 = grab(
        tex, r"so \$([\d.]+)\\%\$ of its missed .*? \$([\d.]+)\\%\$ of extra .*? \$([\d.]+)\\%\$ of wrong", "row rates")
    ml1, ml2, ml3 = grab(tex, r"last column, \$([\d.]+)/([\d.]+)/([\d.]+)\\%\$", "letter's missed-localization triple")
    le, lw = grab(tex, r"shares stay above \$([\d.]+)\$ and \$([\d.]+)\\%\$ for LadderSym", "LadderSym class lower bounds")
    check("letter's missed-localization triple = Table I's column", (num(ml1), num(ml2), num(ml3)), tuple(mloc))
    check("Polytune's missed rate is the triple's first entry", num(r1), num(ml1))
    for i, lab in enumerate(("unprompted", "prompted")):
        check(f"LadderSym {lab}: omissions less localized than the stated extra/wrong floors", mloc[i + 1] < num(lw) < num(le), True)
    check("Polytune: omissions are the least-localized class (missed < wrong < extra)", num(r1) < num(r3) < num(r2), True)
    t1, t2, t3 = (str(int(x)) for x in gam_disp)
    for i, (t, s, r) in enumerate(((t1, sums_disp[0], r1), (t2, sums_disp[1], r2), (t3, sums_disp[2], r3))):
        check(f"displayed row sum {i} = sum of N's row", sum(N[i]), s)
        check(f"row {i} localized share = row sum / post-collapse total", 100 * sum(N[i]) / num(t), num(r), 0.05)
    mu_r, gamma_post = grab(tex, r"\$" + N_ + r"\$ merged pairs among \$" + N_ + r"\$ post-collapse reference errors", "reference merges")
    check("post-collapse reference totals sum to 45,367", num(t1) + num(t2) + num(t3), num(gamma_post))
    check("wrong-class reference total = merged reference pairs", num(t3), num(mu_r))
    sub_pct, auth_pct, auth_n, auth_d, ratio = grab(
        tex, r"mix at \$([\d.]+)\\%\$ substitutions against \$([\d.]+)\\%\$ \(\$(\d+)/(\d+)\$\).*?a \$([\d.]+)\\times\$", "substitution mix")
    check("27.0% = 12,258 / 45,367", 100 * num(mu_r) / num(gamma_post), num(sub_pct), 0.05)
    check("46.6% = 75 / 161", 100 * num(auth_n) / num(auth_d), num(auth_pct), 0.05)
    check("1.7x = 46.6 / 27.0", num(auth_pct) / num(sub_pct), num(ratio), 0.05)
    c75, c51, c35 = grab(tex, r"\(\$(\d+)\$ vs \$(\d+)\$ extra and \$(\d+)\$ missed\)", "authentic counts")
    check("161 = 75 + 51 + 35", num(c75) + num(c51) + num(c35), num(auth_d))

    # ---- Proposition 1's interval from the published per-class counts ------
    pub = {}
    for name in ("Polytune", "unprompted", "prompted"):
        m_tp, m_fp, m_fn, e_tp, e_fp, e_fn = grab(
            sup, r"(?<![a-z])" + name + r" \$\(" + N_ + r";\\,(?:\\,)?" + N_ + r";\\,(?:\\,)?" + N_ + r"\)\$, \$\(" + N_ + r";\\,(?:\\,)?" + N_ + r";\\,(?:\\,)?" + N_ + r"\)\$", f"published counts {name}")
        pub[name] = tuple(num(x) for x in (m_tp, m_fp, m_fn, e_tp, e_fp, e_fn))
    T_p, X_p, XTX_p = grab(sup, r"\$T=" + N_ + r"\$, \$X=" + N_ + r"\$, \$X/\(T\{\+\}X\)=([\d.]+)\$", "T, X Polytune")
    m_tp, m_fp, m_fn, e_tp, e_fp, e_fn = pub["Polytune"]
    T = m_tp + e_tp
    X = min(m_fp, e_fn) + min(e_fp, m_fn)
    check("T = TP_m + TP_e", T, num(T_p))
    check("X = min(FP_m, FN_e) + min(FP_e, FN_m)", X, num(X_p))
    check("X/(T+X) Polytune", X / (T + X), num(XTX_p), 5e-4)
    # the three interval widths now live in Table I's "$X/(T{+}X)$" column
    check("Table I X/(T+X) (Polytune) = the supplement's printed value", widths[0], num(XTX_p))
    for i, name in enumerate(("Polytune", "unprompted", "prompted")):
        m_tp, m_fp, m_fn, e_tp, e_fp, e_fn = pub[name]
        Ti = m_tp + e_tp
        Xi = min(m_fp, e_fn) + min(e_fp, m_fn)
        check(f"X/(T+X) {name} from published counts", Xi / (Ti + Xi), widths[i], 5e-4)
    # (Table I's HM_0 column is checked against the artifacts in verify_shipped;
    # the supplement no longer duplicates the three values)
    # the section heading's factor: how much wider the interval a report leaves
    # open is than the realized collapse-free hidden mass, printed as integers
    # by standard rounding (from Table I's cells: 5.84/6.91/7.86 -> 6/7/8; the
    # artifact-exact 5.86/6.95/7.99 round the same way, checked in verify_shipped)
    f_lo, f_hi = grab(tex, r"The report leaves a \$([\d.]+)\$--\$([\d.]+)\\times\$ interval open", "interval-factor heading")
    g_lo_f, g_hi_f = grab(tex, r"the report leaves open \$([\d.]+)\$--\$([\d.]+)\\times\$ what is realized", "interval-factor sentence")
    check("heading and sentence print the same interval factor", (f_lo, f_hi), (g_lo_f, g_hi_f))
    # the range is the outward hull of the artifact-exact factors (checked in
    # verify_shipped); here: the factors a reader forms from Table I's rounded
    # cells must lie inside it
    factors = [widths[i] / hm0[i] for i in range(3)]
    for i in range(3):
        check(f"interval factor from Table I's cells inside the printed range (configuration {i + 1})",
              num(f_lo) <= factors[i] <= num(f_hi), True)
    pc = grab(sup, r"up to \$([\d.]+)\$ for Polytune's counts", "post-collapse bound")[0]
    check("post-collapse bound X/(max TP + X) Polytune", X / (max(pub["Polytune"][0], pub["Polytune"][3]) + X), num(pc), 5e-4)
    # post-collapse totals: gamma' = (gamma_m - mu_r, gamma_e - mu_r, mu_r)
    check("post-collapse missed total = TP_m + FN_m - merged pairs", pub["Polytune"][0] + pub["Polytune"][2] - num(mu_r), num(t1))
    check("post-collapse extra total = TP_e + FN_e - merged pairs", pub["Polytune"][3] + pub["Polytune"][5] - num(mu_r), num(t2))
    # predicted-side merges as a share of predicted missed events rho_m = TP_m + FP_m
    ms = triple(grab(tex, r"merges \$([\d./]+)\\%\$ of predicted \\emph\{missed\} events", "merged share")[0])
    # mu_p (the predicted-side merge counts) is Table II's row
    for i, name in enumerate(("Polytune", "unprompted", "prompted")):
        rho_m = pub[name][0] + pub[name][1]
        check(f"merged share of predicted missed {name}", 100 * mp[i] / rho_m, ms[i], 0.05)

    # ---- unfounded shares, |M| and the merged cells from Table II -----------
    check("Table II |M| (Polytune) = matrix total", Ms[0], M)
    unf = grab(tex, r"reads \$([\d.]+)\\%\$, \$([\d.]+)\\%\$, \$([\d.]+)\\%\$", "unfounded shares")
    for i in range(3):
        check(f"unfounded share |U|/|M| (configuration {i + 1})", 100 * U[i] / Ms[i], num(unf[i]), 0.05)
    tot_lo, tot_hi, off_lo, off_hi = grab(
        tex, r"matched totals at \$([\d.]+)\$--\$([\d.]+)\$ and off-diagonal counts at \$([\d.]+)\$--\$([\d.]+)\$ times their null means", "null ratios")
    tot_r = [Ms[i] / Ms_null[i] for i in range(3)]
    off_r = [offs[i] / offs_null[i] for i in range(3)]
    check("null matched-total ratio range = outward hull", hull(tot_r, 1), (num(tot_lo), num(tot_hi)))
    check("null off-diagonal ratio range = outward hull", hull(off_r, 1), (num(off_lo), num(off_hi)))
    for i in range(3):
        check(f"Table II matched/null ratio inside the printed range (configuration {i + 1})",
              num(tot_lo) <= tot_r[i] <= num(tot_hi), True)
        check(f"Table II off-diagonal/null ratio inside the printed range (configuration {i + 1})",
              num(off_lo) <= off_r[i] <= num(off_hi), True)
    # Table II's merged-cell rows close the letter's ranges and LadderSym's
    # HM_G values, which the letter states but does not derive
    check("Table II extra->wrong (Polytune) = N[extra][wrong]", ew[0], N[1][2])
    check("Table II missed->wrong (Polytune) = N[missed][wrong]", mw[0], N[0][2])
    check("Table II off-diagonal (Polytune) = N's off-diagonal", offs[0], off)
    for i in range(3):
        check(f"the two merged cells fit inside the off-diagonal (configuration {i + 1})",
              ew[i] + mw[i] <= offs[i], True)
    shares = [100 * ew[i] / offs[i] for i in range(3)]
    check("dominant-cell share range (second derivation) = outward hull", hull(shares, 1), (num(d_lo), num(d_hi)))
    g_lo, g_hi = grab(tex, r"genuine for only \$([\d.]+)\$--\$([\d.]+)\$ of that cell's merged events", "dominant-cell genuine range")
    rates = [ewg[i] / ew[i] for i in range(3)]
    check("dominant-cell genuine range = rounded min/max of genuine / extra->wrong",
          (round(min(rates), 3), round(max(rates), 3)), (num(g_lo), num(g_hi)))
    hmg_l = [num(x) for x in grab(tex, r"LadderSym unprompted and prompted give \$([\d.]+)\$ and \$([\d.]+)\$", "HM_G LadderSym")]
    for i in (1, 2):
        val = (offs[i] - ew[i] - mw[i] + ewg[i] + mwg[i]) / (Ms[i] - A[i] - U[i])
        check(f"HM_G (configuration {i + 1}) = (unmerged off-diagonal + genuine) / (|M| - |A| - |U|) from printed counts", val, hmg_l[i - 1], 5e-4)
    for i, v in enumerate((num(hm_val), hmg_l[0], hmg_l[1])):
        check(f"Table II HM_G row = the letter's printed HM_G (configuration {i + 1})", hmg_tab[i], v)

    # ---- filter gains and the mean error F1 ---------------------------------
    b, a = (triple(x) for x in grab(tex, r"raises missed-class \$F_1\$ from \$([\d./]+)\$ to Table~\\ref\{tab:main\}'s \$([\d./]+)\$", "missed F1 before/after"))
    mb, ma = (triple(x) for x in grab(sup, r"mean error \$F_1\$ from \$([\d./]+)\$ to \$([\d./]+)\$ on MAESTRO-E", "mean F1 before/after (supplement)"))
    r_lo, r_hi = grab(tex, r"the mean error \$F_1\$ by \$([\d.]+)\$--\$([\d.]+)\$", "mean F1 gain range")
    mgains = [ma[i] - mb[i] for i in range(3)]
    check("letter's mean-F1 gain range = outward hull of the supplement's before/after gains",
          hull(mgains, 2), (num(r_lo), num(r_hi)))
    g_lo, g_hi = grab(tex, r"rais(?:es|ing) missed-class \$F_1\$ under the published protocol by \$([\d.]+)\$ to \$([\d.]+)\$ on MAESTRO-E", "conclusion gain")
    tp_lost = grab(tex, r"\$(\d+)/(\d+)/(\d+)\$ of the \$([\d{,}/]+)\$ missed-class true positives \(\$(\d+)/(\d+)/(\d+)\$ on MAESTRO-EI\)", "filter TP losses")
    tp_of = triple(tp_lost[3])
    for i, name in enumerate(("Polytune", "unprompted", "prompted")):
        check(f"filter's TP base = published TP_m ({name})", tp_of[i], pub[name][0])
        check(f"TP losses are non-negative ({name})", num(tp_lost[i]) >= 0 and num(tp_lost[4 + i]) >= 0, True)
    for i in range(3):
        check(f"post-filter missed F1 in prose = Table I's column (configuration {i + 1})", a[i], filt_f1[i])
    gains = [a[i] - b[i] for i in range(3)]
    check("conclusion gain range = outward hull of the three filter gains", hull(gains, 2), (num(g_lo), num(g_hi)))
    pooled = re.findall(r"([\d.]+)/([\d.]+) \((?:Polytune|LadderSym unprompted|LadderSym prompted)", sup)[:3]
    for i in range(3):
        check(f"mean error F1 = mean of pooled missed/extra F1 (configuration {i + 1})", (num(pooled[i][0]) + num(pooled[i][1])) / 2, mb[i], 5e-4)
        check(f"supplement pooled missed F1 = letter's before value (configuration {i + 1})", num(pooled[i][0]), b[i])
        check(f"filtered mean F1 = mean of filtered missed F1 and the untouched extra F1 (configuration {i + 1})", (a[i] + num(pooled[i][1])) / 2, ma[i], 5e-4)
    eb, ea, emb, ema = (triple(x) for x in grab(
        sup, r"on MAESTRO-EI, missed-class \$F_1\$ from \$([\d./]+)\$ to \$([\d./]+)\$ and mean error \$F_1\$ from \$([\d./]+)\$ to \$([\d./]+)\$", "EI before/after (supplement)"))
    eg_m, eg_e = grab(tex, r"raising missed-class \$F_1\$ by \$([\d.]+)\$ and mean error \$F_1\$ by \$([\d.]+)\$ in every configuration", "EI uniform gains")
    for i in range(3):
        check(f"MAESTRO-EI missed-F1 gain rounds to the printed uniform gain (configuration {i + 1})", round(ea[i] - eb[i], 2), num(eg_m))
        check(f"MAESTRO-EI mean-F1 gain rounds to the printed uniform gain (configuration {i + 1})", round(ema[i] - emb[i], 2), num(eg_e))
    c_ei = grab(tex, r"on MAESTRO-E and by \$([\d.]+)\$ on MAESTRO-EI", "conclusion EI gain")[0]
    check("conclusion EI gain = Sec. IV's uniform missed-F1 gain", num(c_ei), num(eg_m))
    # the EI published mean error F1 triple left Sec. IV for the supplement's
    # filter sentence (captured above as emb); the letter now states only that
    # it ranks the configurations the way raw HM and F do, so that ordering is
    # what the printed numbers must support
    for i in range(2):
        check(f"EI published mean error F1 rises with configuration ({i + 1} -> {i + 2})", emb[i] < emb[i + 1], True)

    # ---- MAESTRO-EI recoveries ---------------------------------------------
    # the exact per-configuration rates moved to the supplement; the letter
    # prints their rounded hulls
    ww_s, ww_gen_s, ww_rate_s, ew_s, ew_gen_s, ew_rate_s = grab(
        sup, r"merged-cell counts \$([\d{,}/]+)\$ \(wrong\$\\to\$wrong; \$([\d{,}/]+)\$, i\.e\.\\ \$([\d./]+)\$, name the deleted note\) and \$([\d{,}/]+)\$ \(extra\$\\to\$wrong; \$([\d{,}/]+)\$, i\.e\.\\ \$([\d./]+)\$\)", "EI merged cells")
    cells, ww_gen, ew_cells, ew_gen = triple(ww_s), triple(ww_gen_s), triple(ew_s), triple(ew_gen_s)
    rate, dom_rate = triple(ww_rate_s), triple(ew_rate_s)
    for i in range(3):
        check(f"EI wrong->wrong naming rate = genuine / cell (configuration {i + 1})", ww_gen[i] / cells[i], rate[i], 5e-4)
        check(f"EI extra->wrong naming rate = genuine / cell (configuration {i + 1})", ew_gen[i] / ew_cells[i], dom_rate[i], 5e-4)
    r_lo_ei, r_hi_ei = grab(tex, r"name the manifest's deleted note in \$([\d.]+)\$--\$([\d.]+)\$ of their merged", "EI naming range")
    check("letter's wrong->wrong naming range = outward hull of the supplement's rates",
          hull(rate, 2), (num(r_lo_ei), num(r_hi_ei)))
    d_lo_ei, d_hi_ei = grab(tex, r"only \$([\d.]+)\$--\$([\d.]+)\$ of merged events name a deleted note", "EI dominant-cell naming range")
    check("letter's extra->wrong naming range = outward hull of the supplement's rates (2 dp)",
          hull(dom_rate, 2), (num(d_lo_ei), num(d_hi_ei)))
    s_lo, s_hi, s_den = grab(tex, r"\$([\d.]+)\$--\$([\d.]+)\$ of the \$" + N_ + r"\$ planted sites", "EI site-level range")
    planted = num(grab(tex, r"recovers \$([\d.]+)\$ of the \$" + N_ + r"\$ planted substitutions", "planted substitutions")[1])
    check("site-level denominator = the planted substitutions", planted, num(s_den))
    site = [rate[i] * cells[i] / planted for i in range(3)]
    check("EI site-level range = outward hull of rate x cell / planted", hull(site, 2), (num(s_lo), num(s_hi)))
    for i in range(3):
        check(f"EI site-level recovery inside the printed range (configuration {i + 1})",
              num(s_lo) <= site[i] <= num(s_hi), True)
    merges, manifest, rec_s, planted_s, prec = grab(
        sup, r"collapse merges \$" + N_ + r"\$ pairs, \$" + N_ + r"\$ of them manifest substitutions \(recall \$([\d.]+)\$ of the \$" + N_ + r"\$ planted; precision \$([\d.]+)\$", "EI recall and precision")
    check("EI collapse precision = manifest merges / all merges", num(manifest) / num(merges), num(prec), 5e-4)
    check("EI collapse recall = manifest merges / planted substitutions", num(manifest) / num(planted_s), num(rec_s), 5e-5)
    check("supplement planted count = letter's planted count", num(planted_s), planted)
    rec_l, _pl, prec_l = grab(tex, r"recovers \$([\d.]+)\$ of the \$" + N_ + r"\$ planted substitutions at precision \$([\d.]+)\$", "EI recall/precision (letter)")
    check("letter recall = supplement recall", num(rec_l), num(rec_s))
    check("letter precision = supplement precision", num(prec_l), num(prec))
    rem = grab(sup, r"precision \$([\d.]+)\$; the \$([\d.]+)\\%\$ remainder", "EI remainder")
    check("EI remainder = 100 - 100 x precision", round(100 - 100 * num(rem[0]), 1), num(rem[1]))
    tot, sub5, nonsub = grab(sup, r"The \$" + N_ + r"\$ insertion, omission, and decoy events .*? include \$" + N_ + r"\$ within \$5\$~ms .*? the other \$" + N_ + r"\$ are the non-substitution", "EI event totals")
    check("non-substitution events = all - within-5-ms", num(tot) - num(sub5), num(nonsub))
    f1, f2, f3, f4 = grab(sup, r"\(\$" + N_ + r"\$ matched off-diagonal, \$(\d+)\$ diagonal, \$" + N_ + r"\$ collapse-absorbed, \$" + N_ + r"\$ unmatched\)", "flip fates")
    flips = num(grab(sup, r"planting \$" + N_ + r"\$ misclassifications", "planted flips")[0])
    check("planted flips = sum of the four fates", num(f1) + num(f2) + num(f3) + num(f4), flips)
    off_meas, f_off, coinc = grab(sup, r"measured off-diagonal of \$" + N_ + r"\$ holds those \$" + N_ + r"\$ plus \$" + N_ + r"\$ coincidental", "flip off-diagonal decomposition")
    check("measured off-diagonal = matched flips + coincidental pairings", num(f_off) + num(coinc), num(off_meas))
    check("matched off-diagonal flips = the first fate", num(f_off), num(f1))
    # the sentence no longer prints the sum itself; its two summands are the
    # letter's planted-substitution count and the construction paragraph's
    # non-substitution total, so the arithmetic still closes across the pages
    ei_pre, m_tot, m_drop, e_pre = grab(
        sup, r"holds \$" + N_ + r"\$ error events before the collapse \(the manifest's \$" + N_
        + r"\$ label note-ons less \$(\d+)\$ lacking a note-off, which the MIDI reader drops; MAESTRO-E: \$" + N_ + r"\$\)",
        "pre-collapse reference counts")
    ratio_pre = grab(tex, r"\$([\d.]+)\\times\$ as many reference error events before the collapse", "pre-collapse ratio")[0]
    check("manifest label note-ons = 2 x planted substitutions + non-substitution events",
          2 * planted + num(tot), num(m_tot))
    check("pre-collapse reference events = manifest note-ons - dropped", num(m_tot) - num(m_drop), num(ei_pre))
    check("MAESTRO-E pre-collapse reference events = post-collapse total + merged pairs", num(gamma_post) + num(mu_r), num(e_pre))
    check("EI/E pre-collapse ratio", round(num(ei_pre) / num(e_pre), 1), num(ratio_pre))
    d_ins, d_om, d_neg = grab(sup, r"\(\$" + N_ + r"\$ insertions, \$" + N_ + r"\$ omissions, \$" + N_ + r"\$ decoys counted twice\)", "decoy breakdown")
    check("insertions + omissions + 2 x decoys = 107,643", num(d_ins) + num(d_om) + 2 * num(d_neg), num(tot))
    n_inj = num(grab(tex, r"substitutions; \$" + N_ + r"\$ injections\)", "EI injections")[0])
    check("substitutions + omissions + insertions = injections", planted + num(d_om) + num(d_ins), n_inj)
    jit_l = grab(tex, r"the planted jitter being \$(\d+)\$~ms", "letter jitter")[0]
    jit_s = grab(sup, r"predicted onsets jittered \$(\d+)\$~ms", "supplement jitter")[0]
    check("letter jitter = supplement jitter", num(jit_l), num(jit_s))
    hm_rec = grab(sup, r"recovers \$\\mathrm\{HM\}=([\d.]+)\$", "recovered HM*")[0]
    check("recovered HM* = 1/6 to four decimals", round(1 / 6, 4), num(hm_rec))

    # ---- abstract's rounded ranges -----------------------------------------
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex).group(1)
    u_lo, u_hi = grab(ab, r"that share is (\d+) to (\d+) percent of localized", "abstract unfounded range")
    unf_ei = triple(grab(tex, r"an unfounded share of \$([\d./]+)\\%\$", "EI unfounded shares")[0])
    all_unf = [num(x) for x in unf] + list(unf_ei)
    check("abstract unfounded range = outward hull over both corpora", hull(all_unf, 0), (num(u_lo), num(u_hi)))
    c_lo, c_hi, c_spread = grab(tex, r"it is \$([\d.]+)\$--\$([\d.]+)\$ on both corpora \(differing by at most \$([\d.]+)\$ across configurations on MAESTRO-E\)", "conclusion HM_G levels")
    bound_l = triple(grab(tex, r"raw \$\\mathrm\{HM\}\\ge([\d./]+)\$", "pitch-blind bound")[0])
    # the equal-pitch shares are Table II's last row
    hm_tab = [offs[i] / Ms[i] for i in range(3)]   # raw HM per configuration
    for i in range(3):
        check(f"pitch-blind lower bound <= raw HM x equal-pitch share (configuration {i + 1})", bound_l[i] <= hm_tab[i] * eq_share[i] / 100, True)
    hg_lo, hg_hi = grab(ab, r"misclassification is (\d+) to (\d+) percent", "abstract HM_G range")
    hmg_E = [num(hm_val)] + [num(x) for x in grab(tex, r"LadderSym unprompted and prompted give \$([\d.]+)\$ and \$([\d.]+)\$", "HM_G LadderSym")]
    hmg_EI = triple(grab(tex, r"\\mathrm\{HM\}_G=([\d./]+)\$ there", "EI HM_G")[0])
    allg = hmg_E + list(hmg_EI)
    check("abstract HM_G range = outward hull over both corpora (percent)", hull([100 * g for g in allg], 0), (num(hg_lo), num(hg_hi)))
    raw_all = [offs[i] / Ms[i] for i in range(3)]
    for i in range(3):
        check(f"charge-U HM lies strictly between HM_G and raw HM (configuration {i + 1})", hmg_E[i] < cu_vals[i] < raw_all[i], True)
    n_lo, n_hi = grab(ab, r"substitutions, (\d+) to (\d+) percent name the omitted note", "abstract naming range")
    check("abstract naming range = outward hull of EI naming rates", hull([100 * r for r in rate], 0), (num(n_lo), num(n_hi)))
    ag_lo, ag_hi = grab(ab, r"by ([\d.]+) to ([\d.]+)\.", "abstract gain range")
    all_gains = gains + [ea[i] - eb[i] for i in range(3)]
    check("abstract gain range = outward hull of the six filter gains (both corpora)", hull(all_gains, 2), (num(ag_lo), num(ag_hi)))
    check("abstract gain upper end = conclusion's MAESTRO-E upper end", num(ag_hi), num(g_hi))
    check("conclusion HM_G range = outward hull of the six values (both corpora)", hull(allg, 2), (num(c_lo), num(c_hi)))
    check("conclusion HM_G spread bound covers the printed values' spread", 0 <= num(c_spread) - (max(hmg_E) - min(hmg_E)) <= 0.001 + 1e-9, True)
    for i in range(3):
        check(f"conclusion HM_G range holds the MAESTRO-EI value (configuration {i + 1})", num(c_lo) <= round(hmg_EI[i], 2) <= num(c_hi), True)
    check("conclusion HM_G range = the abstract's range (percent)", (round(100 * num(c_lo)), round(100 * num(c_hi))), (num(hg_lo), num(hg_hi)))
    spread = num(grab(tex, r"differs by at most \$([\d.]+)\$ across the three configurations", "HM_G spread")[0])
    # "at most" is a bound: the printed value must be >= the spread of the
    # printed HM_G values and within one unit of it
    check("HM_G spread bound covers the printed values' spread", 0 <= spread - (max(hmg_E) - min(hmg_E)) <= 0.001 + 1e-9, True)
    check("conclusion's HM_G spread = Sec. IV's", num(c_spread), spread)
    words = len(re.sub(r"\s+", " ", ab).strip().split(" "))
    check("abstract word count within SPL's 100-175", 100 <= words <= 175, True)

    return _report()


if __name__ == "__main__":
    raise SystemExit(main())
