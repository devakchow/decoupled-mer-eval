"""Verify the Figure 1 claim on the REAL representative case, exactly.

Case (Polytune x MAESTRO, piece 01-03_R1_2014_...--4, t=178.95 s):
  reference: extra @ pitch 68              (an inserted note)
  system   : extra @ 68  +  missed @ 69    (a substitution: played 68 for 69)

Scenario A (as measured): the system's missed@69 is co-located with its extra@68.
Scenario B (counterfactual): the SAME three events, but the system's missed@69 is
relocated far away (unrelated deletion elsewhere).

Claim to verify:
  * shipped per-class TP/FP/FN are IDENTICAL in A and B (non-identifiable);
  * decoupled HM differs (A: localized substitution, off-diagonal; B: the extra
    matches correctly and the missed is an unrelated margin event).

If both hold, Figure 1's two-scenario schematic is exact, not rhetorical.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decoupled_scorer as ds

T = 178.95
PIECE = "demo"


def ev(onset, pitch, etype):
    return ds.Event(onset, pitch, etype, PIECE)


def build(scenario):
    # reference: one inserted note (extra) at pitch 68
    ref = [ev(T + 0.0045, 68, "extra")]
    # prediction: the extra@68, plus a missed@69 that is the substitution partner
    if scenario == "A":
        pred = [ev(T, 68, "extra"), ev(T, 69, "missed")]
    else:  # B: relocate the missed far away (unrelated deletion)
        pred = [ev(T, 68, "extra"), ev(T + 47.0, 69, "missed")]
    return pred, ref


def shipped_triples(pred, ref, tau=0.05):
    s = ds.shipped_scores(pred, ref, tau=tau)
    return {k: (s[k]["tp"], s[k]["fp"], s[k]["fn"]) for k in ds.SHIP_TRACKS}


def decoupled_hm(pred, ref, tau=0.05, eps=0.05):
    d = ds.decoupled_scores(pred, ref, tau, epsilon=eps, collapse="strict")
    return d.hm, d.confusion, d.n_localized


def main() -> int:
    A_pred, A_ref = build("A")
    B_pred, B_ref = build("B")

    sa, sb = shipped_triples(A_pred, A_ref), shipped_triples(B_pred, B_ref)
    ha, ca, na = decoupled_hm(A_pred, A_ref)
    hb, cb, nb = decoupled_hm(B_pred, B_ref)

    print("shipped A:", sa)
    print("shipped B:", sb)
    print("shipped identical:", sa == sb)
    print(f"decoupled A: HM={ha}  n_localized={na}")
    print(f"decoupled B: HM={hb}  n_localized={nb}")
    off_a = sum(ca[r][c] for r in ds.ERROR_TYPES for c in ds.ERROR_TYPES if r != c)
    off_b = sum(cb[r][c] for r in ds.ERROR_TYPES for c in ds.ERROR_TYPES if r != c)
    print(f"off-diagonal A={off_a}  B={off_b}")

    ok = (sa == sb) and (off_a == 1) and (off_b == 0)
    print("\nCLAIM HOLDS:" , ok,
          "-- identical shipped report, HM off-diagonal 1 (A) vs 0 (B)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
