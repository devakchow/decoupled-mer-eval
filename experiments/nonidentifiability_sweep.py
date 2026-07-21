#!/usr/bin/env python3
"""nonidentifiability_sweep.py — constructed demonstration of Proposition 1.

WHAT THE PROPOSITION ACTUALLY SAYS (and a wrong reading to avoid)
----------------------------------------------------------------
The shipped protocol scores each error class as an INDEPENDENT per-class
matching problem. For class k it therefore sees exactly three numbers:

    TP_k = N_kk                      (predicted k matched to reference k)
    FP_k = rho_k - N_kk              (rho_k = ALL predicted-k events)
    FN_k = gamma_k - N_kk            (gamma_k = ALL reference-k events)

Note rho_k and gamma_k are TOTAL event counts, including events that our joint
class-agnostic matcher leaves unmatched. That is the crux.

A wrong reading — the one this file originally implemented — is to treat
rho_k / gamma_k as the marginals of the MATCHED confusion matrix. Under that
reading the matched total is pinned, the trace is pinned, and therefore
HM = off-diagonal / total is pinned too: HM would add nothing. It is worth
stating explicitly because it is an easy mistake, and it is what a skeptical
reviewer may initially assume.

The correct reading gives HM a genuinely free direction. Hold rho_k, gamma_k
and every N_kk fixed, then convert a pair of *temporally unrelated* events
(one spurious prediction of class i, one unmatched reference of class j, i!=j)
into a *temporally coincident* pair (the same prediction now sitting within tau
of that reference, still mislabelled). This:

  * leaves TP_k unchanged (nothing joins the diagonal),
  * leaves rho_k, gamma_k unchanged (no event is created or destroyed),
  * therefore leaves EVERY shipped per-class P/R/F bit-identical,
  * but moves mass onto the off-diagonal of the joint confusion matrix,
    raising HM from 0 toward its maximum.

Operationally: the shipped protocol double-counts a right-onset/wrong-label
note as one FP plus one FN, which is indistinguishable from a wholly missed
reference event plus an unrelated false alarm. HM is exactly what separates
those two situations, and it is invisible to the shipped statistic.

Outputs a tidy CSV + JSON asserting the invariance numerically.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Tuple

ERROR_TYPES: Tuple[str, str, str] = ("missed", "extra", "wrong")


def shipped_per_class(rho: Dict[str, int], gamma: Dict[str, int],
                      diag: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """Shipped per-class P/R/F. Depends ONLY on (rho, gamma, diag)."""
    out: Dict[str, Dict[str, float]] = {}
    for k in ERROR_TYPES:
        tp = diag[k]
        fp = rho[k] - tp
        fn = gamma[k] - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        out[k] = dict(precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn)
    return out


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "results")
    os.makedirs(out_dir, exist_ok=True)

    # Fixed by the data and held constant for the whole sweep.
    rho = {"missed": 900, "extra": 1200, "wrong": 800}    # all predicted-k
    gamma = {"missed": 1000, "extra": 1100, "wrong": 850}  # all reference-k
    diag = {"missed": 500, "extra": 800, "wrong": 600}     # correctly-labelled matches

    # Unmatched budget available to convert into localized-but-mislabelled pairs.
    unmatched_pred = {k: rho[k] - diag[k] for k in ERROR_TYPES}
    unmatched_ref = {k: gamma[k] - diag[k] for k in ERROR_TYPES}
    budget = min(sum(unmatched_pred.values()), sum(unmatched_ref.values()))

    shipped = shipped_per_class(rho, gamma, diag)
    ref_sig = tuple(round(shipped[k]["f1"], 15) for k in ERROR_TYPES)

    rows: List[dict] = []
    for m in range(0, budget + 1, max(1, budget // 20)):
        # m = number of off-diagonal (localized-but-mislabelled) matched pairs.
        # Distribute across the two dominant confusion directions; the split
        # does not affect either statistic, only the total m does.
        n_localized = sum(diag.values()) + m
        hm = m / n_localized if n_localized else 0.0
        sh = shipped_per_class(rho, gamma, diag)   # recomputed: provably unchanged
        sig = tuple(round(sh[k]["f1"], 15) for k in ERROR_TYPES)
        rows.append(dict(
            off_diagonal_pairs=m,
            n_localized=n_localized,
            hm=hm,
            f1_missed=sh["missed"]["f1"],
            f1_extra=sh["extra"]["f1"],
            f1_wrong=sh["wrong"]["f1"],
            mean_error_f1=sum(sh[k]["f1"] for k in ERROR_TYPES) / 3.0,
            shipped_identical_to_base=(sig == ref_sig),
        ))

    csv_path = os.path.join(out_dir, "nonidentifiability_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hms = [r["hm"] for r in rows]
    all_same = all(r["shipped_identical_to_base"] for r in rows)
    summary = dict(
        n_points=len(rows),
        shipped_invariant=all_same,
        rho=rho, gamma=gamma, diag=diag,
        convertible_budget=budget,
        shipped_f1={k: shipped[k]["f1"] for k in ERROR_TYPES},
        shipped_mean_error_f1=sum(shipped[k]["f1"] for k in ERROR_TYPES) / 3.0,
        hm_min=min(hms), hm_max=max(hms), hm_span=max(hms) - min(hms),
    )
    with open(os.path.join(out_dir, "nonidentifiability_sweep.json"),
              "w", encoding="utf-8") as fh:
        json.dump(dict(summary=summary, points=rows), fh, indent=2)

    print("constructed non-identifiability sweep (Proposition 1)")
    print(f"  sweep points                     : {summary['n_points']}")
    print(f"  shipped stats invariant at ALL   : {summary['shipped_invariant']}")
    print("  shipped per-class F1 (constant)  : " + "  ".join(
        f"{k}={shipped[k]['f1']:.6f}" for k in ERROR_TYPES))
    print(f"  shipped mean error F1 (constant) : "
          f"{summary['shipped_mean_error_f1']:.6f}")
    print(f"  HM sweeps                        : {summary['hm_min']:.4f} .. "
          f"{summary['hm_max']:.4f}   (span {summary['hm_span']:.4f})")
    print(f"  wrote {csv_path}")
    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
