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
experiments/figs/tab_null.tex. No artifact is consulted. Exit code 0 = every
relation holds; 1 = a relation failed or an anchor pattern was not found
(the check refuses to pass on a missing sentence).

Run: python experiments/verify_printed_arithmetic.py
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LETTER = os.path.join(ROOT, "proposal", "spl_letter_v5.tex")
SUPP = os.path.join(ROOT, "proposal", "spl_supplementary_v5.tex")
TAB_NULL = os.path.join(HERE, "figs", "tab_null.tex")

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

    tex, sup, tab = read(LETTER), read(SUPP), read(TAB_NULL)
    N_ = r"([\d{,}]+)"

    # ---- Polytune's confusion matrix and its margins ----------------------
    g = grab(tex, r"\\begin\{bmatrix\}" + N_ + "&" + N_ + "&" + N_ + r"\\\\" + N_ + "&" + N_ + "&" + N_
             + r"\\\\" + N_ + "&" + N_ + "&" + N_ + r"\\end\{bmatrix\}", "bmatrix")
    N = [[num(g[3 * i + j]) for j in range(3)] for i in range(3)]
    M = sum(map(sum, N))
    off = M - sum(N[i][i] for i in range(3))
    K_col = sum(N[i][2] for i in range(3))
    outside_wrong = off - N[0][2] - N[1][2]

    K, G, A, U = (triple(grab(tex, r"\$\|%s\|=([\d{,}/]+)\$" % s, "|%s| triple" % s)[0]) for s in "KGAU")
    check("|K| (Polytune) = predicted-wrong column of N", K_col, K[0])
    for i in range(3):
        check(f"G+A+U = |K| (configuration {i + 1})", G[i] + A[i] + U[i], K[i])

    hm_num, hm_gen, hm_num2, hm_den, hm_val = grab(
        tex, r"\\mathrm\{HM\}_G=\(" + N_ + r"\+" + N_ + r"\)/\(\|M\|-\|A\|-\|U\|\)=" + N_ + "/" + N_ + r"=([\d.]+)", "HM_G formula")
    check("1,173 = off-diagonal cells outside the predicted-wrong column", outside_wrong, num(hm_num))
    check("HM_G numerator 1,173 + 136 = 1,309", num(hm_num) + num(hm_gen), num(hm_num2))
    check("HM_G denominator = |M| - |A| - |U|", M - A[0] - U[0], num(hm_den))
    check("HM_G = 1,309 / 20,694", num(hm_num2) / num(hm_den), num(hm_val), 5e-4)
    a1, a2 = grab(tex, r"split \$(\d+)\$ extra\$\\to\$wrong and \$(\d+)\$ missed\$\\to\$wrong", "136 split")
    check("136 genuine off-diagonal = 37 + 99", num(a1) + num(a2), num(hm_gen))

    cu_vals = triple(grab(sup, r"Charging unfounded events as misclassification too \(denominator \$\|M\|-\|A\|\$\) gives \$([\d./]+)\$", "charge-U convention")[0])
    raw_off, raw_M, raw_val = grab(tex, r"\(\$" + N_ + r"\$ of \$\|M\|=" + N_ + r"\$\) gives the raw \$([\d.]+)\$", "raw convention")
    check("off-diagonal count of N = 8,928", off, num(raw_off))
    check("|M| = 30,460", M, num(raw_M))
    check("raw HM = 8,928 / 30,460", off / M, num(raw_val), 5e-4)
    lo, hi = grab(tex, r"the span \$\[([\d.]+),\\,([\d.]+)\]\$ for Polytune", "span")
    check("span endpoints = (HM_G, raw HM)", (num(lo), num(hi)), (num(hm_val), num(raw_val)))
    dom, dom_of = grab(tex, r"holds \$" + N_ + r"\$ of \$" + N_ + r"\$ off-diagonal events", "dominant cell")
    check("dominant cell = N[extra][wrong]", N[1][2], num(dom))
    check("dominant cell denominator = off-diagonal count", off, num(dom_of))
    d_lo, d_hi = grab(tex, r"which carries \$(\d+)\$--\$(\d+)\\%\$ of the raw hidden mass", "dominant share range")
    check("Polytune dominant-cell share 82% is the range's upper end", round(100 * N[1][2] / off), num(d_hi))

    # ---- row localization rates against the post-collapse reference totals -
    t1, t2, t3, r1, r2, r3 = grab(
        tex, r"reference totals \(\$" + N_ + r"\$, \$" + N_ + r"\$, \$" + N_ + r"\$\), its row sums show that \$([\d.]+)\\%\$ .*? \$([\d.]+)\\%\$ of extra .*? \$([\d.]+)\\%\$ of wrong", "row rates")
    for i, (t, r) in enumerate(((t1, r1), (t2, r2), (t3, r3))):
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
    widths = triple(grab(tex, r"X/\(T\{\+\}X\)=([\d./]+)\$", "interval widths")[0])
    for i, name in enumerate(("Polytune", "unprompted", "prompted")):
        m_tp, m_fp, m_fn, e_tp, e_fp, e_fn = pub[name]
        Ti = m_tp + e_tp
        Xi = min(m_fp, e_fn) + min(e_fp, m_fn)
        check(f"X/(T+X) {name} from published counts", Xi / (Ti + Xi), widths[i], 5e-4)
    pc = grab(sup, r"up to \$([\d.]+)\$ for Polytune's counts", "post-collapse bound")[0]
    check("post-collapse bound X/(max TP + X) Polytune", X / (max(pub["Polytune"][0], pub["Polytune"][3]) + X), num(pc), 5e-4)
    # post-collapse totals: gamma' = (gamma_m - mu_r, gamma_e - mu_r, mu_r)
    check("post-collapse missed total = TP_m + FN_m - merged pairs", pub["Polytune"][0] + pub["Polytune"][2] - num(mu_r), num(t1))
    check("post-collapse extra total = TP_e + FN_e - merged pairs", pub["Polytune"][3] + pub["Polytune"][5] - num(mu_r), num(t2))
    # predicted-side merges as a share of predicted missed events rho_m = TP_m + FP_m
    ms = triple(grab(tex, r"merges \$([\d./]+)\\%\$ of predicted \\emph\{missed\} events", "merged share")[0])
    mp = triple(grab(tex, r"events \(\$([\d{,}/]+)\$ pairs\) and", "predicted merges")[0])
    for i, name in enumerate(("Polytune", "unprompted", "prompted")):
        rho_m = pub[name][0] + pub[name][1]
        check(f"merged share of predicted missed {name}", 100 * mp[i] / rho_m, ms[i], 0.05)

    # ---- unfounded shares, |M| per configuration from Table II --------------
    rows = re.findall(r"& " + N_ + r" & " + N_ + r" & ([\d.]+) & " + N_ + r" & " + N_ + r" & ([\d.]+) \\\\", tab)
    if len(rows) != 3:
        FAILS.append("Table II: expected 3 data rows")
        return _report()
    Ms = [num(r[0]) for r in rows]
    check("Table II |M| (Polytune) = matrix total", Ms[0], M)
    unf = grab(tex, r"reads \$([\d.]+)\\%\$, \$([\d.]+)\\%\$, \$([\d.]+)\\%\$", "unfounded shares")
    for i in range(3):
        check(f"unfounded share |U|/|M| (configuration {i + 1})", 100 * U[i] / Ms[i], num(unf[i]), 0.05)
    tot_lo, tot_hi, off_lo, off_hi = grab(
        tex, r"matched totals \$([\d.]+)\$--\$([\d.]+)\\times\$ and off-diagonal counts \$([\d.]+)\$--\$([\d.]+)\\times\$", "null ratios")
    tot_r = [num(r[0]) / num(r[1]) for r in rows]
    off_r = [num(r[3]) / num(r[4]) for r in rows]
    check("null matched-total ratio range", (round(min(tot_r), 1), round(max(tot_r), 1)), (num(tot_lo), num(tot_hi)))
    check("null off-diagonal ratio range", (round(min(off_r), 1), round(max(off_r), 1)), (num(off_lo), num(off_hi)))
    for i in range(3):
        check(f"Table II printed ratio (matched, row {i + 1})", round(tot_r[i], 1), num(rows[i][2]))
        check(f"Table II printed ratio (off-diagonal, row {i + 1})", round(off_r[i], 1), num(rows[i][5]))
    # supplement's per-configuration merged-cell counts close the letter's
    # ranges and LadderSym's HM_G values, which the letter states but does
    # not derive
    ew_s, ewg1, ewg2, ewg3, off_s = grab(
        sup, r"extra\$\\to\$wrong \$([\d{,}/]+)\$ \(\$(\d+)/(\d+)/(\d+)\$ genuine\) of \$([\d{,}/]+)\$ off-diagonal events", "extra->wrong cells")
    mw_s, mwg1, mwg2, mwg3 = grab(sup, r"missed\$\\to\$wrong \$([\d{,}/]+)\$ \(\$(\d+)/(\d+)/(\d+)\$ genuine\)", "missed->wrong cells")
    ew, offs, mw = triple(ew_s), triple(off_s), triple(mw_s)
    ewg = tuple(num(x) for x in (ewg1, ewg2, ewg3))
    mwg = tuple(num(x) for x in (mwg1, mwg2, mwg3))
    check("supplement extra->wrong (Polytune) = N[extra][wrong]", ew[0], N[1][2])
    check("supplement missed->wrong (Polytune) = N[missed][wrong]", mw[0], N[0][2])
    check("supplement off-diagonal (Polytune) = N's off-diagonal", offs[0], off)
    check("136 genuine off-diagonal = genuine extra->wrong + missed->wrong (Polytune)", ewg[0] + mwg[0], num(hm_gen))
    for i in range(3):
        check(f"Table II off-diagonal = supplement off-diagonal (configuration {i + 1})", num(rows[i][3]), offs[i])
    shares = [ew[i] / offs[i] for i in range(3)]
    check("dominant-cell share range = rounded min/max of extra->wrong / off-diagonal",
          (round(100 * min(shares)), round(100 * max(shares))), (num(d_lo), num(d_hi)))
    g_lo, g_hi = grab(tex, r"genuine for only \$([\d.]+)\$--\$([\d.]+)\$ of that cell's merged events", "dominant-cell genuine range")
    rates = [ewg[i] / ew[i] for i in range(3)]
    check("dominant-cell genuine range = rounded min/max of genuine / extra->wrong",
          (round(min(rates), 3), round(max(rates), 3)), (num(g_lo), num(g_hi)))
    hmg_l = [num(x) for x in grab(tex, r"LadderSym unprompted and prompted give \$([\d.]+)\$ and \$([\d.]+)\$", "HM_G LadderSym")]
    for i in (1, 2):
        val = (offs[i] - ew[i] - mw[i] + ewg[i] + mwg[i]) / (Ms[i] - A[i] - U[i])
        check(f"HM_G (configuration {i + 1}) = (unmerged off-diagonal + genuine) / (|M| - |A| - |U|) from printed counts", val, hmg_l[i - 1], 5e-4)

    # ---- filter gains and the mean error F1 ---------------------------------
    b, a = (triple(x) for x in grab(tex, r"\(our replication\) from \$([\d./]+)\$ to \$([\d./]+)\$", "missed F1 before/after"))
    mb, ma = (triple(x) for x in grab(sup, r"mean error \$F_1\$ from \$([\d./]+)\$ to \$([\d./]+)\$ on MAESTRO-E", "mean F1 before/after (supplement)"))
    r_lo, r_hi = grab(tex, r"Repl\.\\ \$\\bar F_1\$\) by \$([\d.]+)\$--\$([\d.]+)\$", "mean F1 gain range")
    mgains = [ma[i] - mb[i] for i in range(3)]
    check("letter's mean-F1 gain range = rounded min/max of the supplement's before/after",
          (round(min(mgains), 2), round(max(mgains), 2)), (num(r_lo), num(r_hi)))
    g_lo, g_hi = grab(tex, r"raising missed-class \$F_1\$ under the published protocol by \$([\d.]+)\$ to \$([\d.]+)\$ on MAESTRO-E", "conclusion gain")
    gains = [a[i] - b[i] for i in range(3)]
    check("conclusion gain range = min/max of the three filter gains", (round(min(gains), 2), round(max(gains), 2)), (num(g_lo), num(g_hi)))
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
    ei_pub = triple(grab(tex, r"mean error \$F_1\$ does \(\$([\d./]+)\$ here\)", "EI published mean F1")[0])
    for i in range(3):
        check(f"EI published mean F1 in Sec. IV = supplement's before value (configuration {i + 1})", ei_pub[i], emb[i])

    # ---- MAESTRO-EI recoveries ---------------------------------------------
    rate = triple(grab(tex, r"name the manifest's deleted note in \$([\d./]+)\$", "EI naming rate")[0])
    cells = triple(grab(sup, r"merged-cell counts \$([\d{,}/]+)\$ \(wrong\$\\to\$wrong\)", "EI wrong->wrong cells")[0])
    site = triple(grab(tex, r"rather than merged events, \$([\d./]+)\$", "EI site-level")[0])
    planted = num(grab(tex, r"recovers \$([\d.]+)\$ of the \$" + N_ + r"\$ planted substitutions", "planted substitutions")[1])
    for i in range(3):
        check(f"EI site-level recovery = rate x cell / planted (configuration {i + 1})", rate[i] * cells[i] / planted, site[i], 5e-3)
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
    ei_pre, e_pre = grab(sup, r"holds \$" + N_ + r"\$ error events before the collapse \(MAESTRO-E: \$" + N_ + r"\$\)", "pre-collapse reference counts")
    ratio_pre = grab(tex, r"\$([\d.]+)\\times\$ as many pre-collapse reference label events", "pre-collapse ratio")[0]
    check("MAESTRO-E pre-collapse reference events = post-collapse total + merged pairs", num(gamma_post) + num(mu_r), num(e_pre))
    check("EI/E pre-collapse ratio", round(num(ei_pre) / num(e_pre), 1), num(ratio_pre))
    jit_l = grab(tex, r"the planted jitter being \$(\d+)\$~ms", "letter jitter")[0]
    jit_s = grab(sup, r"predicted onsets jittered \$(\d+)\$~ms", "supplement jitter")[0]
    check("letter jitter = supplement jitter", num(jit_l), num(jit_s))
    hm_rec = grab(sup, r"recovers \$\\mathrm\{HM\}=([\d.]+)\$", "recovered HM*")[0]
    check("recovered HM* = 1/6 to four decimals", round(1 / 6, 4), num(hm_rec))

    # ---- abstract's rounded ranges -----------------------------------------
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex).group(1)
    u_lo, u_hi = grab(ab, r"that share is (\d+) to (\d+) percent of localized", "abstract unfounded range")
    check("abstract unfounded range = rounded min/max of Table I shares", (round(min(map(num, unf))), round(max(map(num, unf)))), (num(u_lo), num(u_hi)))
    hg_lo, hg_hi = grab(ab, r"misclassification (\d+) to (\d+) percent", "abstract HM_G range")
    hmg_E = [num(hm_val)] + [num(x) for x in grab(tex, r"LadderSym unprompted and prompted give \$([\d.]+)\$ and \$([\d.]+)\$", "HM_G LadderSym")]
    hmg_EI = triple(grab(tex, r"\\mathrm\{HM\}_G=([\d./]+)\$ there", "EI HM_G")[0])
    allg = hmg_E + list(hmg_EI)
    check("abstract HM_G range = rounded min/max over both corpora (percent)", (round(100 * min(allg)), round(100 * max(allg))), (num(hg_lo), num(hg_hi)))
    raw_all = [num(rows[i][3]) / Ms[i] for i in range(3)]
    for i in range(3):
        check(f"charge-U HM lies strictly between HM_G and raw HM (configuration {i + 1})", hmg_E[i] < cu_vals[i] < raw_all[i], True)
    n_lo, n_hi = grab(ab, r"substitutions, (\d+) to (\d+) percent name the omitted note", "abstract naming range")
    check("abstract naming range = rounded min/max of EI naming rates", (round(100 * min(rate)), round(100 * max(rate))), (num(n_lo), num(n_hi)))
    ag_lo, ag_hi = grab(ab, r"by ([\d.]+) to ([\d.]+)\.", "abstract gain range")
    check("abstract gain range = conclusion gain range", (num(ag_lo), num(ag_hi)), (num(g_lo), num(g_hi)))
    spread = num(grab(tex, r"differs by at most \$([\d.]+)\$ across the three configurations", "HM_G spread")[0])
    check("HM_G spread on MAESTRO-E = max - min", round(max(hmg_E) - min(hmg_E), 3), spread)
    words = len(re.sub(r"\s+", " ", ab).strip().split(" "))
    check("abstract word count within SPL's 100-175", 100 <= words <= 175, True)

    return _report()


if __name__ == "__main__":
    raise SystemExit(main())
