#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bridge_checks.py -- numerical verification of every proposition in
proposal/ANALYTIC-BRIDGE.md against the real 177-piece x 2-system artifacts.

Run:  python experiments/bridge_checks.py
Exit 0 iff every check that is asserted in the .md holds.

Data used (all under experiments/results/gilbreth/):
  {A_polytune_maestro,B_laddersym_maestro_unprompted}_shipped.json
  ..._guard.json           (per-piece 3x3 CROSS-TRACK mir_eval matrix + totals)
  ..._strict_eps05.json    (decoupled sweep, primary config)
  ..._strict_eps0.json     (decoupled sweep, no-collapse sensitivity)
  ..._pitchaware_eps05.json
  nonidentifiability_empirical.json  (per-piece table, 354 observations)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G = os.path.join(HERE, "results", "gilbreth")
sys.path.insert(0, HERE)

from decoupled_scorer import (Event, decoupled_scores, shipped_scores,  # noqa: E402
                              match_events, prf)

SYS = {"Polytune": "A_polytune_maestro",
       "LadderSym": "B_laddersym_maestro_unprompted"}
# guard row/col order is (Extra, Removed, Correct); shipped track names:
GUARD_ORDER = ["extra", "missed", "correct"]

FAIL = []
NOTE = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAIL.append((name, detail))
    print("  [%s] %s%s" % (status, name, ("  -- " + detail) if detail else ""))
    return cond


def load(sysname, suffix):
    with open(os.path.join(G, "%s_%s.json" % (SYS[sysname], suffix)),
              "r", encoding="utf-8") as fh:
        return json.load(fh)


def f1_of(tp, fp, fn):
    return prf(tp, fp, fn)[2]


# --------------------------------------------------------------------------- #
print("\n=== C1. Shipped triple is the guard matrix DIAGONAL + class TOTALS ===")
# Claim: TP_k = G_kk, FP_k = rho_k - G_kk, FN_k = gamma_k - G_kk, where G is the
# coupled onset+pitch cross-track match matrix and rho/gamma are TOTAL per-class
# event counts (including events the matcher leaves unmatched).
for s in SYS:
    ship = load(s, "shipped")["shipped_50ms"]
    guard = load(s, "guard")
    M = np.array(guard["grand_matrix"], dtype=int)          # rows pred, cols gt
    ppc = guard["per_piece"]
    rho = {k: 0 for k in GUARD_ORDER}
    gam = {k: 0 for k in GUARD_ORDER}
    pk = ["Extra", "Removed", "Correct"]
    gk = ["extra_notes_midi", "removed_notes_midi", "correct_notes_midi"]
    for piece in ppc.values():
        for i, k in enumerate(GUARD_ORDER):
            rho[k] += piece["pred_counts"][pk[i]]
            gam[k] += piece["gt_counts"][gk[i]]
    ok_tp = all(int(M[i, i]) == ship[k]["tp"] for i, k in enumerate(GUARD_ORDER))
    ok_fp = all(rho[k] - int(M[i, i]) == ship[k]["fp"] for i, k in enumerate(GUARD_ORDER))
    ok_fn = all(gam[k] - int(M[i, i]) == ship[k]["fn"] for i, k in enumerate(GUARD_ORDER))
    check("%s: TP_k == G_kk" % s, ok_tp,
          "diag=%s vs tp=%s" % ([int(M[i, i]) for i in range(3)],
                                [ship[k]["tp"] for k in GUARD_ORDER]))
    check("%s: FP_k == rho_k - G_kk" % s, ok_fp)
    check("%s: FN_k == gamma_k - G_kk" % s, ok_fn)
    # and: row sums of G are strictly LESS than rho (unmatched events exist)
    strictly_less = all(int(M[i].sum()) < rho[k] for i, k in enumerate(GUARD_ORDER))
    check("%s: rowsum(G)_k < rho_k for all k (rho are TOTALS, not marginals)" % s,
          strictly_less,
          "rowsums=%s rho=%s" % (list(M.sum(1)), [rho[k] for k in GUARD_ORDER]))

# --------------------------------------------------------------------------- #
print("\n=== C2. Guard-derived per-piece shipped F1 == empirical per-piece table ===")
emp = json.load(open(os.path.join(G, "nonidentifiability_empirical.json"),
                     "r", encoding="utf-8"))
per_piece_ship = {}   # (sys, piece) -> dict of track -> (tp, fp, fn)
for s in SYS:
    guard = load(s, "guard")["per_piece"]
    worst = 0.0
    for piece, d in guard.items():
        M = np.array(d["matrix"], dtype=int)
        trip = {}
        for i, k in enumerate(GUARD_ORDER):
            tp = int(M[i, i])
            fp = d["pred_counts"][["Extra", "Removed", "Correct"][i]] - tp
            fn = d["gt_counts"][["extra_notes_midi", "removed_notes_midi",
                                 "correct_notes_midi"][i]] - tp
            trip[k] = (tp, fp, fn)
        per_piece_ship[(s, piece)] = trip
    for row in emp["per_piece"][s]:
        trip = per_piece_ship[(s, row["piece"])]
        for k in GUARD_ORDER:
            worst = max(worst, abs(f1_of(*trip[k]) - row["ship_%s_f1" % k]))
    check("%s: max |guard-derived F1 - table F1| over 177x3" % s, worst < 1e-12,
          "max dev = %.3e" % worst)

# --------------------------------------------------------------------------- #
print("\n=== C3. Decoupled internal identities (corpus, all configs x taus) ===")
worst_locf = 0.0
n_cfg = 0
for s in SYS:
    for cfg in ("strict_eps05", "strict_eps0", "pitchaware_eps05"):
        for r in load(s, cfg)["decoupled"]["per_tau"]:
            n_cfg += 1
            L = r["n_localized"]
            conf = r["confusion"]
            tot = sum(conf[a][b] for a in conf for b in conf[a])
            tr = sum(conf[a][a] for a in conf)
            od = tot - tr
            assert r["localization"]["tp"] == L == tot, (s, cfg, r["tau_ms"])
            assert r["miss_margin"] == r["localization"]["fn"]
            assert r["false_alarm_on_correct"] + r["spurious"] == r["localization"]["fp"]
            assert abs(r["hm"] - od / L) < 1e-12
            worst_locf = max(worst_locf, abs(
                r["localization"]["f1"] - 2.0 * L / (r["n_pred_err"] + r["n_ref_err"])))
check("Loc-TP == sum(N) == n_localized; miss_margin == Loc-FN; "
      "fa+spurious == Loc-FP; HM == 1 - tr(N)/L", True,
      "%d (system,config,tau) cells" % n_cfg)
check("Loc-F == 2L/(|P|+|R|) on all cells", worst_locf < 1e-12,
      "max dev = %.3e" % worst_locf)

worst_pp = 0.0
for s in SYS:
    for row in emp["per_piece"][s]:
        worst_pp = max(worst_pp, abs(
            row["loc_f1"] - 2.0 * row["n_localized"]
            / (row["n_pred_err"] + row["n_ref_err"])))
check("Loc-F == 2L/(|P|+|R|) on all 354 per-piece observations",
      worst_pp < 1e-12, "max dev = %.3e" % worst_pp)

# --------------------------------------------------------------------------- #
print("\n=== C4. N_kk vs shipped TP_k (expected: NOT equal -- different objects) ===")
for s in SYS:
    ship = load(s, "shipped")["shipped_50ms"]
    r50 = [r for r in load(s, "strict_eps05")["decoupled"]["per_tau"]
           if r["tau_ms"] == 50][0]
    d = {k: r50["confusion"][k][k] for k in ("missed", "extra")}
    print("    %s: N_missed,missed=%d vs shipped TP_missed=%d ; "
          "N_extra,extra=%d vs shipped TP_extra=%d"
          % (s, d["missed"], ship["missed"]["tp"], d["extra"], ship["extra"]["tp"]))
    check("%s: N_kk != shipped TP_k (they are different matrices)" % s,
          d["missed"] != ship["missed"]["tp"] or d["extra"] != ship["extra"]["tp"])

# --------------------------------------------------------------------------- #
print("\n=== C5. BOUND  shipped micro error-F1 <= Loc-F(tau) ===")
# micro over the two ERROR tracks {missed, extra}:
#   F_micro = 2*sum_k TP_k / sum_k (rho_k + gamma_k)
for s in SYS:
    # corpus level, both collapse radii
    ship = load(s, "shipped")["shipped_50ms"]
    tp = ship["missed"]["tp"] + ship["extra"]["tp"]
    den = sum(ship[k][x] for k in ("missed", "extra") for x in ("tp", "fp")) \
        + sum(ship[k][x] for k in ("missed", "extra") for x in ("tp", "fn"))
    micro = 2.0 * tp / den
    for cfg in ("strict_eps05", "strict_eps0"):
        r50 = [r for r in load(s, cfg)["decoupled"]["per_tau"]
               if r["tau_ms"] == 50][0]
        check("%s/%s corpus: micro error-F1 %.4f <= Loc-F %.4f"
              % (s, cfg, micro, r50["localization"]["f1"]),
              micro <= r50["localization"]["f1"] + 1e-12)
    print("    %s micro-F1=%.4f  (shipped MACRO mean error F1=%.4f)"
          % (s, micro, ship["_mean_error_f1"]))

# per piece (354 observations), eps = 50 ms (primary config)
viol = []
gaps = []
for s in SYS:
    for row in emp["per_piece"][s]:
        trip = per_piece_ship[(s, row["piece"])]
        tp = trip["missed"][0] + trip["extra"][0]
        den = sum(trip[k][0] * 2 + trip[k][1] + trip[k][2] for k in ("missed", "extra"))
        micro = 2.0 * tp / den if den else 0.0
        gaps.append(row["loc_f1"] - micro)
        if micro > row["loc_f1"] + 1e-12:
            viol.append((s, row["piece"], micro, row["loc_f1"]))
gaps = np.array(gaps)
check("per-piece (354 obs, eps=50ms): micro <= Loc-F holds on >= 353 of 354 "
      "[NOT a theorem under collapse -- documented in P5 Remark]",
      len(viol) <= 1, "%d violation(s); min slack %.4f, median %.4f, max %.4f"
      % (len(viol), gaps.min(), float(np.median(gaps)), gaps.max()))
if viol:
    for v in viol[:10]:
        print("      VIOLATION %s %s micro=%.4f loc_f=%.4f" % v)

# per piece, MACRO mean error F1 (the number the systems actually report)
viol_macro = []
gaps_macro = []
for s in SYS:
    for row in emp["per_piece"][s]:
        macro = 0.5 * (row["ship_missed_f1"] + row["ship_extra_f1"])
        gaps_macro.append(row["loc_f1"] - macro)
        if macro > row["loc_f1"] + 1e-12:
            viol_macro.append((s, row["piece"], macro, row["loc_f1"]))
gm = np.array(gaps_macro)
check("per-piece: MACRO mean error-F1 <= Loc-F holds EMPIRICALLY on 354/354 "
      "[REFUTED as a theorem -- see C11 counterexample]",
      not viol_macro,
      "%d violations of 354; min slack %.4f, median %.4f"
      % (len(viol_macro), gm.min(), float(np.median(gm))))
if viol_macro:
    for v in viol_macro[:6]:
        print("      MACRO VIOLATION %s %s macro=%.4f loc_f=%.4f" % v)

# --------------------------------------------------------------------------- #
print("\n=== C6. Region (Loc-F, HM): is micro-F1 <= (1-HM)-type bound true? ===")
# Candidate tighter bound: micro error-F1 <= (1 - HM) * Loc-F ?  Test it.
viol_hm = []
for s in SYS:
    for row in emp["per_piece"][s]:
        trip = per_piece_ship[(s, row["piece"])]
        tp = trip["missed"][0] + trip["extra"][0]
        den = sum(trip[k][0] * 2 + trip[k][1] + trip[k][2] for k in ("missed", "extra"))
        micro = 2.0 * tp / den if den else 0.0
        if micro > (1.0 - row["hm"]) * row["loc_f1"] + 1e-12:
            viol_hm.append((s, row["piece"], micro, (1 - row["hm"]) * row["loc_f1"]))
check("REFUTED on real data: micro <= (1-HM)*Loc-F fails (equality holds only "
      "under the no-collapse hypothesis; collapse moves diagonal mass to 'wrong')",
      len(viol_hm) > 300, "%d violations of 354 -- as expected" % len(viol_hm))

# --------------------------------------------------------------------------- #
print("\n=== C7. Oracle experiment B: shipped TRIPLE identical, HM and Loc-F differ ===")


def _at(k):
    return 2.0 * k


ref1, pred1, ref2, pred2 = [], [], [], []
k = 1
for _ in range(4):
    ref1.append(Event(_at(k), 60, "missed")); pred1.append(Event(_at(k), 60, "missed"))
    ref2.append(Event(_at(k), 60, "missed")); pred2.append(Event(_at(k), 60, "missed")); k += 1
for _ in range(4):
    ref1.append(Event(_at(k), 60, "extra")); pred1.append(Event(_at(k), 60, "extra"))
    ref2.append(Event(_at(k), 60, "extra")); pred2.append(Event(_at(k), 60, "extra")); k += 1
for _ in range(2):
    ref1.append(Event(_at(k), 60, "missed")); pred1.append(Event(_at(k), 60, "extra"))
    ref2.append(Event(_at(k), 60, "missed")); k += 1
for _ in range(2):
    ref1.append(Event(_at(k), 60, "extra")); pred1.append(Event(_at(k), 60, "missed"))
    ref2.append(Event(_at(k), 60, "extra")); k += 1
for _ in range(2):
    pred2.append(Event(_at(k), 60, "missed")); k += 1
for _ in range(2):
    pred2.append(Event(_at(k), 60, "extra")); k += 1

s1, s2 = shipped_scores(pred1, ref1), shipped_scores(pred2, ref2)
d1 = decoupled_scores(pred1, ref1, tau=0.05)
d2 = decoupled_scores(pred2, ref2, tau=0.05)
same_triple = all(s1[t][x] == s2[t][x] for t in ("missed", "extra", "correct")
                  for x in ("tp", "fp", "fn"))
check("expB: FULL shipped triple (TP,FP,FN) identical in all three tracks",
      same_triple,
      "sys1 missed=(%d,%d,%d) extra=(%d,%d,%d)"
      % (s1["missed"]["tp"], s1["missed"]["fp"], s1["missed"]["fn"],
         s1["extra"]["tp"], s1["extra"]["fp"], s1["extra"]["fn"]))
check("expB: HM differs (1/3 vs 0)",
      abs(d1.hm - 1 / 3) < 1e-12 and d2.hm == 0.0,
      "HM1=%.6f HM2=%.6f" % (d1.hm, d2.hm))
check("expB: Loc-F ALSO differs -> Loc-F is not shipped-recoverable either",
      abs(d1.localization["f1"] - 1.0) < 1e-12
      and abs(d2.localization["f1"] - 2 / 3) < 1e-12,
      "LocF1=%.6f LocF2=%.6f" % (d1.localization["f1"], d2.localization["f1"]))
mic = 2.0 * (s1["missed"]["tp"] + s1["extra"]["tp"]) / (
    sum(s1[t][x] for t in ("missed", "extra") for x in ("tp", "tp", "fp", "fn")))
check("expB: bound is TIGHT on system 2 (micro %.6f == Loc-F %.6f)"
      % (mic, d2.localization["f1"]),
      abs(mic - d2.localization["f1"]) < 1e-12)

# --------------------------------------------------------------------------- #
print("\n=== C8. shipped_scores == mir_eval (bit-exactness claim), randomized ===")
import mir_eval  # noqa: E402


def hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def mir_tp(pev, rev, tau, pitch_tol=50.0):
    if not pev or not rev:
        return 0
    ri = np.array([[e.onset_s, e.onset_s + 0.1] for e in rev])
    rp = np.array([hz(e.pitch_midi) for e in rev])
    pi = np.array([[e.onset_s, e.onset_s + 0.1] for e in pev])
    pp = np.array([hz(e.pitch_midi) for e in pev])
    return len(mir_eval.transcription.match_notes(
        ri, rp, pi, pp, onset_tolerance=tau, pitch_tolerance=pitch_tol,
        offset_ratio=None))


rng = np.random.default_rng(20260720)
bad_ship = bad_loc = 0
n_trials = 400
for _ in range(n_trials):
    def rnd(n):
        return [Event(float(np.round(rng.uniform(0, 2.0), 4)),
                      int(rng.integers(58, 66)),
                      str(rng.choice(("missed", "extra", "correct"))), "p0")
                for _ in range(n)]
    R = rnd(int(rng.integers(0, 14)))
    P = rnd(int(rng.integers(0, 14)))
    sh = shipped_scores(P, R, tau=0.05)
    for t in ("missed", "extra", "correct"):
        if sh[t]["tp"] != mir_tp([e for e in P if e.etype == t],
                                 [e for e in R if e.etype == t], 0.05):
            bad_ship += 1
    # Loc-TP vs mir_eval with pitch ignored (pitch_tolerance huge), no collapse
    d = decoupled_scores(P, R, tau=0.05, epsilon=0.0)
    Pe = [e for e in P if e.etype != "correct"]
    Re = [e for e in R if e.etype != "correct"]
    # replicate the scorer's own collapse at eps=0 before comparing
    from decoupled_scorer import collapse_wrong
    Pc = [e for e in collapse_wrong(Pe, epsilon=0.0) if e.etype != "correct"]
    Rc = [e for e in collapse_wrong(Re, epsilon=0.0) if e.etype != "correct"]
    if d.n_localized != mir_tp(Pc, Rc, 0.05, pitch_tol=1e9):
        bad_loc += 1
check("shipped_scores TP == mir_eval.match_notes TP (%d trials x 3 tracks)"
      % n_trials, bad_ship == 0, "%d mismatches" % bad_ship)
check("Loc-TP == mir_eval TP with pitch_tolerance=inf on the collapsed error "
      "union (%d trials)" % n_trials, bad_loc == 0, "%d mismatches" % bad_loc)

# --------------------------------------------------------------------------- #
print("\n=== C9. tau-boundary: where the N_DECIMALS=4 rounding can disagree ===")
# decoupled mode: slack 1e-9, no rounding -> admits d <= tau + 1e-9
# mir_eval / shipped mode: rounds d to 4 decimals -> admits d <= tau + 0.5e-4
tau = 0.05
d_probe = tau + 4.9e-5           # rounds DOWN to 0.05 -> mir_eval admits
P = [Event(0.0, 60, "extra")]
R = [Event(d_probe, 60, "extra")]
n_ship = shipped_scores(P, R, tau=tau)["extra"]["tp"]
n_dec = len(match_events(P, R, tau, require_pitch=False))
n_mir = mir_tp(P, R, tau)
check("boundary d=tau+4.9e-5: mir_eval and shipped BOTH admit it",
      n_ship == 1 and n_mir == 1, "ship=%d mir=%d" % (n_ship, n_mir))
check("boundary d=tau+4.9e-5: decoupled matcher REJECTS it (semantics differ "
      "in (tau+1e-9, tau+0.5e-4])", n_dec == 0, "dec=%d" % n_dec)

# --------------------------------------------------------------------------- #
print("\n=== C10. Reduction proposition: HM=0 + no-collapse + pitch-consistent ===")
# Construct corpora satisfying (i)-(iv) and check Loc-F == micro shipped error F1
# and N_kk == TP_k.
red_bad = 0
for trial in range(300):
    n = int(rng.integers(1, 10))
    ev_r, ev_p = [], []
    t = 0.0
    for _ in range(n):
        t += 1.0
        cls = str(rng.choice(("missed", "extra")))
        pitch = int(rng.integers(58, 66))
        roll = rng.random()
        if roll < 0.5:      # localized, same class, same pitch -> diagonal
            ev_r.append(Event(t, pitch, cls))
            ev_p.append(Event(t + float(rng.uniform(-0.04, 0.04)), pitch, cls))
        elif roll < 0.75:   # reference-only -> miss margin / FN
            ev_r.append(Event(t, pitch, cls))
        else:               # prediction-only -> spurious / FP
            ev_p.append(Event(t, pitch, cls))
    d = decoupled_scores(ev_p, ev_r, tau=0.05, epsilon=0.0)
    sh = shipped_scores(ev_p, ev_r, tau=0.05)
    if d.hm not in (0.0, None):
        continue
    tp = sh["missed"]["tp"] + sh["extra"]["tp"]
    den = sum(sh[k][x] * (2 if x == "tp" else 1)
              for k in ("missed", "extra") for x in ("tp", "fp", "fn"))
    micro = 2.0 * tp / den if den else 0.0
    nkk_ok = all(d.confusion[k][k] == sh[k]["tp"] for k in ("missed", "extra"))
    if abs(micro - d.localization["f1"]) > 1e-12 or not nkk_ok:
        red_bad += 1
check("reduction: under (i)-(iv), N_kk == TP_k and Loc-F == micro shipped "
      "error-F1 (300 random corpora)", red_bad == 0, "%d failures" % red_bad)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
print("\n=== C11. Constructive RANGE of HM admitted by a fixed shipped report ===")
# Given the shipped error-track triples, build a family of corpora with the
# IDENTICAL shipped report and HM = X/(T+X) for X = 0 .. Xmax, where
#   T = TP_m + TP_e,  Xmax = min(FP_m, FN_e) + min(FP_e, FN_m).


def build_family(TPm, FPm, FNm, TPe, FPe, FNe, X):
    ref, pred = [], []
    k = [0]

    def t():
        k[0] += 1
        return 2.0 * k[0]
    for _ in range(TPm):
        u = t(); ref.append(Event(u, 60, "missed")); pred.append(Event(u, 60, "missed"))
    for _ in range(TPe):
        u = t(); ref.append(Event(u, 60, "extra")); pred.append(Event(u, 60, "extra"))
    xa = min(X, min(FPm, FNe)); xb = X - xa
    assert xb <= min(FPe, FNm)
    for _ in range(xa):
        u = t(); pred.append(Event(u, 60, "missed")); ref.append(Event(u, 60, "extra"))
    for _ in range(xb):
        u = t(); pred.append(Event(u, 60, "extra")); ref.append(Event(u, 60, "missed"))
    for _ in range(FPm - xa):
        pred.append(Event(t(), 60, "missed"))
    for _ in range(FPe - xb):
        pred.append(Event(t(), 60, "extra"))
    for _ in range(FNm - xb):
        ref.append(Event(t(), 60, "missed"))
    for _ in range(FNe - xa):
        ref.append(Event(t(), 60, "extra"))
    return pred, ref


for s in SYS:
    sh = load(s, "shipped")["shipped_50ms"]
    m, e = sh["missed"], sh["extra"]
    T = m["tp"] + e["tp"]
    Xmax = min(m["fp"], e["fn"]) + min(e["fp"], m["fn"])
    tgt = {"missed": (m["tp"], m["fp"], m["fn"]), "extra": (e["tp"], e["fp"], e["fn"])}
    ok = True
    for X in (0, Xmax // 4, Xmax // 2, Xmax):
        P, R = build_family(m["tp"], m["fp"], m["fn"], e["tp"], e["fp"], e["fn"], X)
        s2 = shipped_scores(P, R)
        d = decoupled_scores(P, R, tau=0.05)
        got = {k2: (s2[k2]["tp"], s2[k2]["fp"], s2[k2]["fn"])
               for k2 in ("missed", "extra")}
        ok &= (got == tgt) and abs(d.hm - X / (T + X)) < 1e-12
    check("%s: shipped report invariant while HM sweeps [0, %.4f] (T=%d, Xmax=%d)"
          % (s, Xmax / (T + Xmax), T, Xmax), ok)
    print("    %s: ambiguity band width %.4f vs the paper's between-system HM "
          "gap of 0.0937" % (s, Xmax / (T + Xmax)))

# --------------------------------------------------------------------------- #
print("\n=== C12. Identity micro = (1-HM)*Loc-F under the no-collapse hypothesis ===")
rng2 = np.random.default_rng(7)
bad = tested = badtr = 0
for _ in range(2000):
    n = int(rng2.integers(1, 14)); ref = []; pred = []; t = 0.0
    for _ in range(n):
        t += 2.0
        kind = rng2.random()
        c = str(rng2.choice(("missed", "extra")))
        o = str(rng2.choice(("missed", "extra")))
        p = int(rng2.integers(58, 66))
        j = float(rng2.uniform(-0.04, 0.04))
        if kind < 0.35:
            ref.append(Event(t, p, c)); pred.append(Event(t + j, p, c))
        elif kind < 0.60 and o != c:
            ref.append(Event(t, p, c))
            pred.append(Event(t + j, int(rng2.integers(58, 66)), o))
        elif kind < 0.80:
            ref.append(Event(t, p, c))
        else:
            pred.append(Event(t, p, c))
    d = decoupled_scores(pred, ref, tau=0.05, epsilon=0.05)
    sh = shipped_scores(pred, ref)
    if d.hm is None:
        continue
    tested += 1
    T = sh["missed"]["tp"] + sh["extra"]["tp"]
    tr = sum(d.confusion[k][k] for k in ("missed", "extra", "wrong"))
    if T != tr:
        badtr += 1
    mic = 2.0 * T / sum(sh[k]["tp"] * 2 + sh[k]["fp"] + sh[k]["fn"]
                        for k in ("missed", "extra"))
    if abs(mic - (1 - d.hm) * d.localization["f1"]) > 1e-12:
        bad += 1
check("micro == (1-HM)*Loc-F on %d slot-isolated corpora (no collapse, "
      "pitch-consistent diagonal)" % tested, bad == 0 and badtr == 0,
      "%d identity violations, %d tr(N)!=T" % (bad, badtr))

# --------------------------------------------------------------------------- #
print("\n=== C13. MACRO mean-error-F1 <= Loc-F is REFUTED as a theorem ===")
worst = (-1.0, 0.0, 0.0)
for _ in range(20000):
    n = int(rng2.integers(1, 8)); ref = []; pred = []; t = 0.0
    for _ in range(n):
        t += 2.0
        c = str(rng2.choice(("missed", "extra"))); p = int(rng2.integers(58, 66))
        r = rng2.random()
        if r < 0.5:
            ref.append(Event(t, p, c)); pred.append(Event(t, p, c))
        elif r < 0.75:
            ref.append(Event(t, p, c))
        else:
            pred.append(Event(t, p, c))
    if not ref or not pred:
        continue
    d = decoupled_scores(pred, ref, tau=0.05)
    sh = shipped_scores(pred, ref)
    macro = 0.5 * (sh["missed"]["f1"] + sh["extra"]["f1"])
    if macro - d.localization["f1"] > worst[0]:
        worst = (macro - d.localization["f1"], macro, d.localization["f1"])
check("counterexample to MACRO <= Loc-F exists (so the 354/354 empirical "
      "observation in C5 is NOT a theorem)", worst[0] > 1e-12,
      "max macro-LocF = %.4f (macro=%.4f, Loc-F=%.4f)" % worst)

# --------------------------------------------------------------------------- #
print("\n=== C14. (Loc-F, HM) unit square is fully reachable ===")
ok = True
for f_t, h_t in [(1.0, 1.0), (1.0, 0.0), (0.5, 1.0), (0.5, 0.0),
                 (0.8, 0.5), (0.25, 0.75)]:
    L = 100; X = int(round(h_t * L)); D = int(round(2 * L / f_t))
    sp = (D - 2 * L) // 2; sr = D - 2 * L - sp
    ref = []; pred = []; t = 0.0
    for i in range(L):
        t += 2.0
        ref.append(Event(t, 60, "missed"))
        pred.append(Event(t, 60, "extra" if i < X else "missed"))
    for _ in range(sp):
        t += 2.0; pred.append(Event(t, 60, "extra"))
    for _ in range(sr):
        t += 2.0; ref.append(Event(t, 60, "extra"))
    d = decoupled_scores(pred, ref, tau=0.05)
    ok &= abs(d.localization["f1"] - f_t) < 1e-9 and abs(d.hm - h_t) < 1e-9
check("every probed (Loc-F, HM) target in [0,1]^2 is realized exactly -> no "
      "unreachable region; HM is a free coordinate", ok)

# --------------------------------------------------------------------------- #
print("\n=== C15. eps=0 is NOT collapse-free on real data ===")
for s in SYS:
    sh = load(s, "shipped")["shipped_50ms"]
    rho = sum(sh[k]["tp"] + sh[k]["fp"] for k in ("missed", "extra"))
    gam = sum(sh[k]["tp"] + sh[k]["fn"] for k in ("missed", "extra"))
    r0 = [r for r in load(s, "strict_eps0")["decoupled"]["per_tau"]
          if r["tau_ms"] == 50][0]
    a = rho - r0["n_pred_err"]; b = gam - r0["n_ref_err"]
    check("%s: eps=0 STILL collapses (a=%d pred-side, b=%d ref-side) -> the "
          "no-collapse hypothesis holds in NO shipped config" % (s, a, b),
          a > 0 and b > 0)

# --------------------------------------------------------------------------- #
print("\n=== C16. mir_eval boundary semantics: exact disagreement band ===")
band = []
for delta in (1e-10, 1e-9, 5e-9, 4.9e-5, 5.1e-5):
    P = [Event(0.0, 60, "extra")]; R = [Event(0.05 + delta, 60, "extra")]
    dec = decoupled_scores(P, R, tau=0.05, epsilon=0.0).n_localized
    shp = shipped_scores(P, R, tau=0.05)["extra"]["tp"]
    band.append((delta, dec, shp))
check("shipped mode admits d in (tau, tau+0.5e-4]; decoupled admits only "
      "d <= tau+1e-9 -> semantics differ exactly on (tau+1e-9, tau+0.5e-4]",
      [b[1:] for b in band] == [(1, 1), (1, 1), (0, 1), (0, 1), (0, 0)],
      str(band))



print("\n" + "=" * 74)
if FAIL:
    print("FAILED CHECKS (%d):" % len(FAIL))
    for n, d in FAIL:
        print("  - %s  %s" % (n, d))
    sys.exit(1)
print("ALL BRIDGE CHECKS PASSED.")
sys.exit(0)
