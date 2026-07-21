#!/usr/bin/env python3
"""nonidentifiability_empirical.py — does Proposition 1's ambiguity actually
OCCUR in the outputs of the published systems, or is it only constructible?

`nonidentifiability_sweep.py` exhibits a constructed family: shipped per-class
P/R/F held bit-identical while HM sweeps 0 -> 0.34. That proves the ambiguity
is *possible*. This script asks the strictly stronger empirical question, using
the real predictions of Polytune and LadderSym on all 177 MAESTRO-E pieces:

    Are there real (piece, system) observations whose SHIPPED per-class
    statistics agree to within epsilon, but whose HM differs materially?

Method
------
For every piece and every system we compute, at tau = 50 ms:
  * the shipped per-class vector (P/R/F for missed, extra, wrong) exactly as
    the shipped protocol computes it, per piece;
  * the decoupled quantities Loc-F and HM.

We then search all cross-system pairs over the same or different pieces for
matches: shipped vectors within L-infinity distance `--eps-shipped`, HM apart
by at least `--min-hm-gap`. Same-piece cross-system pairs are reported
separately because they are the most interpretable: the shipped protocol says
the two systems performed equivalently on that piece; the decoupled measure
says they did not.

Honesty constraints baked in:
  * Pieces whose HM is undefined (no localized error events) are excluded and
    counted, never silently dropped.
  * We report the full distribution, not just the extreme pair, so the reader
    can see whether matches are abundant or cherry-picked.
  * The shipped comparison uses the F1 triple; a stricter variant additionally
    requiring P and R agreement is reported as `strict_pr` for robustness.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoupled_scorer as ds  # noqa: E402

ERROR_TYPES = ds.ERROR_TYPES          # decoupled: missed / extra / wrong
# The shipped protocol scores three TRACKS, which are NOT the decoupled error
# types: it has a "correct" track and no "wrong" track (a right-onset/
# wrong-label note is charged as FP+FN across two tracks rather than named).
# The shipped comparison vector must therefore use these keys.
SHIPPED_TRACKS = ("missed", "extra", "correct")


def per_piece_table(pred_dir: str, gt_meta: str, tau: float,
                    epsilon: float, collapse: str) -> List[dict]:
    """One row per piece: shipped per-class vector + decoupled Loc-F and HM."""
    pred = ds.load_pred_events(pred_dir)
    ref = ds.load_ref_events(gt_meta)
    pieces = sorted({e.piece for e in pred} | {e.piece for e in ref})
    by_pred: Dict[str, List[ds.Event]] = {p: [] for p in pieces}
    by_ref: Dict[str, List[ds.Event]] = {p: [] for p in pieces}
    for e in pred:
        by_pred[e.piece].append(e)
    for e in ref:
        by_ref[e.piece].append(e)

    rows: List[dict] = []
    for pc in pieces:
        p_ev, r_ev = by_pred[pc], by_ref[pc]
        dec = ds.decoupled_scores(p_ev, r_ev, tau=tau, epsilon=epsilon,
                                  collapse=collapse)
        sh = ds.shipped_scores(p_ev, r_ev, tau=tau)
        row = dict(
            piece=pc,
            hm=dec.hm,                       # None when nothing localized
            loc_f1=dec.localization["f1"],
            n_localized=dec.n_localized,
            n_pred_err=dec.n_pred_err,
            n_ref_err=dec.n_ref_err,
        )
        for k in SHIPPED_TRACKS:
            sk = sh.get(k, {})
            row[f"ship_{k}_p"] = sk.get("precision", 0.0)
            row[f"ship_{k}_r"] = sk.get("recall", 0.0)
            row[f"ship_{k}_f1"] = sk.get("f1", 0.0)
        rows.append(row)
    return rows


def shipped_vec(row: dict, strict_pr: bool = False) -> Tuple[float, ...]:
    if strict_pr:
        return tuple(row[f"ship_{k}_{q}"]
                     for k in SHIPPED_TRACKS for q in ("p", "r", "f1"))
    return tuple(row[f"ship_{k}_f1"] for k in SHIPPED_TRACKS)


def linf(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b)) if a else 0.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", nargs=3,
                    metavar=("NAME", "PRED_DIR", "GT_META"), required=True,
                    help="repeatable: system label, prediction dir, gt meta json")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=ds.DEFAULT_EPSILON_S)
    ap.add_argument("--collapse", choices=("strict", "pitch_aware"),
                    default="strict")
    ap.add_argument("--eps-shipped", type=float, default=0.01,
                    help="max L-inf distance between shipped vectors to count "
                         "as 'the shipped protocol cannot separate them'")
    ap.add_argument("--min-hm-gap", type=float, default=0.10,
                    help="min |HM difference| to count as materially different")
    ap.add_argument("--out", default="results/nonidentifiability_empirical.json")
    args = ap.parse_args(argv)

    tables: Dict[str, List[dict]] = {}
    for name, pred_dir, gt_meta in args.system:
        tables[name] = per_piece_table(pred_dir, gt_meta, args.tau,
                                       args.eps, args.collapse)
        print(f"[{name}] scored {len(tables[name])} pieces", flush=True)

    # Flatten to observations, excluding undefined-HM pieces (counted, not hidden)
    obs: List[dict] = []
    n_undefined = 0
    for name, rows in tables.items():
        for r in rows:
            if r["hm"] is None:
                n_undefined += 1
                continue
            o = dict(r)
            o["system"] = name
            obs.append(o)

    names = list(tables)
    same_piece_pairs: List[dict] = []
    cross_pairs: List[dict] = []

    # (a) same piece, different systems — the most interpretable case
    if len(names) >= 2:
        for a, b in itertools.combinations(names, 2):
            ra = {r["piece"]: r for r in tables[a] if r["hm"] is not None}
            rb = {r["piece"]: r for r in tables[b] if r["hm"] is not None}
            for pc in sorted(set(ra) & set(rb)):
                d_ship = linf(shipped_vec(ra[pc]), shipped_vec(rb[pc]))
                d_ship_strict = linf(shipped_vec(ra[pc], True),
                                     shipped_vec(rb[pc], True))
                gap = abs(ra[pc]["hm"] - rb[pc]["hm"])
                if d_ship <= args.eps_shipped and gap >= args.min_hm_gap:
                    same_piece_pairs.append(dict(
                        piece=pc, system_a=a, system_b=b,
                        shipped_linf=d_ship, shipped_linf_strict_pr=d_ship_strict,
                        hm_a=ra[pc]["hm"], hm_b=rb[pc]["hm"], hm_gap=gap,
                        loc_f1_a=ra[pc]["loc_f1"], loc_f1_b=rb[pc]["loc_f1"],
                        n_localized_a=ra[pc]["n_localized"],
                        n_localized_b=rb[pc]["n_localized"],
                        shipped_f1_a=list(shipped_vec(ra[pc])),
                        shipped_f1_b=list(shipped_vec(rb[pc])),
                    ))

    # (b) any two observations (different piece and/or system)
    for i, j in itertools.combinations(range(len(obs)), 2):
        oa, ob = obs[i], obs[j]
        if oa["piece"] == ob["piece"] and oa["system"] == ob["system"]:
            continue
        d_ship = linf(shipped_vec(oa), shipped_vec(ob))
        gap = abs(oa["hm"] - ob["hm"])
        if d_ship <= args.eps_shipped and gap >= args.min_hm_gap:
            cross_pairs.append(dict(
                piece_a=oa["piece"], system_a=oa["system"],
                piece_b=ob["piece"], system_b=ob["system"],
                shipped_linf=d_ship,
                shipped_linf_strict_pr=linf(shipped_vec(oa, True),
                                            shipped_vec(ob, True)),
                hm_a=oa["hm"], hm_b=ob["hm"], hm_gap=gap,
                loc_f1_a=oa["loc_f1"], loc_f1_b=ob["loc_f1"],
                n_localized_a=oa["n_localized"], n_localized_b=ob["n_localized"],
                shipped_f1_a=list(shipped_vec(oa)),
                shipped_f1_b=list(shipped_vec(ob)),
            ))

    same_piece_pairs.sort(key=lambda d: -d["hm_gap"])
    cross_pairs.sort(key=lambda d: -d["hm_gap"])

    n_obs = len(obs)
    n_possible_cross = n_obs * (n_obs - 1) // 2
    summary = dict(
        tau_s=args.tau, epsilon_s=args.eps, collapse=args.collapse,
        eps_shipped=args.eps_shipped, min_hm_gap=args.min_hm_gap,
        systems=names,
        n_observations=n_obs,
        n_pieces_hm_undefined=n_undefined,
        n_same_piece_matches=len(same_piece_pairs),
        n_cross_matches=len(cross_pairs),
        n_cross_pairs_examined=n_possible_cross,
        cross_match_rate=(len(cross_pairs) / n_possible_cross
                          if n_possible_cross else 0.0),
        max_hm_gap_same_piece=(same_piece_pairs[0]["hm_gap"]
                               if same_piece_pairs else None),
        max_hm_gap_cross=(cross_pairs[0]["hm_gap"] if cross_pairs else None),
    )

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dict(summary=summary,
                       same_piece_matches=same_piece_pairs[:200],
                       cross_matches=cross_pairs[:200],
                       per_piece=tables), fh, indent=2)

    print("\n=== empirical non-identifiability search ===")
    print(f"  observations (piece x system) : {n_obs}")
    print(f"  pieces with HM undefined      : {n_undefined} (excluded)")
    print(f"  criterion: shipped F1 vectors within {args.eps_shipped} (L-inf) "
          f"AND |HM gap| >= {args.min_hm_gap}")
    print(f"  same-piece cross-system matches: {len(same_piece_pairs)}")
    print(f"  any-pair matches               : {len(cross_pairs)} of "
          f"{n_possible_cross} pairs examined "
          f"({100*summary['cross_match_rate']:.3f}%)")
    if same_piece_pairs:
        m = same_piece_pairs[0]
        print("\n  strongest SAME-PIECE case:")
        print(f"    piece {m['piece']}")
        print(f"    shipped F1 {m['system_a']}: "
              + ", ".join(f"{v:.4f}" for v in m["shipped_f1_a"]))
        print(f"    shipped F1 {m['system_b']}: "
              + ", ".join(f"{v:.4f}" for v in m["shipped_f1_b"])
              + f"   (L-inf {m['shipped_linf']:.5f})")
        print(f"    HM {m['system_a']}={m['hm_a']:.4f}  "
              f"{m['system_b']}={m['hm_b']:.4f}   gap {m['hm_gap']:.4f}")
    if cross_pairs:
        m = cross_pairs[0]
        print("\n  strongest ANY-PAIR case:")
        print(f"    {m['system_a']}/{m['piece_a'][:44]}")
        print(f"    {m['system_b']}/{m['piece_b'][:44]}")
        print(f"    shipped L-inf {m['shipped_linf']:.5f}   "
              f"HM {m['hm_a']:.4f} vs {m['hm_b']:.4f}   gap {m['hm_gap']:.4f}")
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
