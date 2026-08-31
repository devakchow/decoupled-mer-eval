"""Locate a REAL, clean localized-but-misclassified note for the Figure 1
worked example. No synthetic data: every candidate is an actual matched pair in
the Polytune x MAESTRO predictions whose reference and predicted error TYPE
disagree while the onset is matched within tau.

Honest framing this figure will assert (verified against decoupled_scorer.py):
  the pair is co-located (matched) => the system localized the error; the types
  differ => it named the wrong class. The shipped per-track metric scores the
  classes independently and so cannot credit the correct localization; on the
  case selected for Fig. 1 it books extra: 1 TP + missed: 1 FP (see
  verify_two_scenario.py).

We rank candidates by temporal isolation (few other events nearby) and prefer
pairs where BOTH sides are real single notes (neither is a collapsed `wrong`),
so the piano-roll shows two concrete notes, not a synthetic merge.

Run on the cluster (has preds + gt + scorer):
  python find_worked_case.py --pred_dir <...> --gt_meta <...> --out cand.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decoupled_scorer as ds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_meta", required=True)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--collapse", default="strict")
    ap.add_argument("--isolation", type=float, default=0.6,
                    help="seconds around the pair that should be sparse")
    ap.add_argument("--out", default="worked_case_candidates.json")
    args = ap.parse_args()

    preds = ds.load_pred_events(args.pred_dir)
    refs = ds.load_ref_events(args.gt_meta)

    # group by piece
    from collections import defaultdict
    P = defaultdict(list)
    R = defaultdict(list)
    for e in preds:
        P[e.piece].append(e)
    for e in refs:
        R[e.piece].append(e)

    cands = []
    for piece in sorted(set(P) & set(R)):
        predc = ds._canonical(ds.collapse_wrong(P[piece], epsilon=args.eps,
                                                mode=args.collapse))
        refc = ds._canonical(ds.collapse_wrong(R[piece], epsilon=args.eps,
                                               mode=args.collapse))
        pred_err = [e for e in predc if e.etype in ds.ERROR_TYPES]
        ref_err = [e for e in refc if e.etype in ds.ERROR_TYPES]
        if not pred_err or not ref_err:
            continue
        match = ds.match_events(pred_err, ref_err, args.tau, require_pitch=False)
        # all error onsets in this piece, for isolation scoring
        all_on = sorted([e.onset_s for e in pred_err] + [e.onset_s for e in ref_err])
        ref_correct = [e for e in refc if e.etype == "correct"]
        corr_on = sorted(e.onset_s for e in ref_correct)
        for pi, ri in match:
            pe, re = pred_err[pi], ref_err[ri]
            if pe.etype == re.etype:
                continue  # diagonal, not HM
            if pe.etype == "wrong" or re.etype == "wrong":
                continue  # prefer two real notes over a collapsed merge
            t = 0.5 * (pe.onset_s + re.onset_s)
            near = sum(1 for o in all_on if abs(o - t) <= args.isolation) - 2
            near_corr = sum(1 for o in corr_on if abs(o - t) <= args.isolation)
            cands.append({
                "piece": piece, "onset_mid": t,
                "pred_onset": pe.onset_s, "pred_pitch": pe.pitch_midi,
                "pred_type": pe.etype,
                "ref_onset": re.onset_s, "ref_pitch": re.pitch_midi,
                "ref_type": re.etype,
                "onset_gap_ms": abs(pe.onset_s - re.onset_s) * 1000.0,
                "pitch_gap": abs(pe.pitch_midi - re.pitch_midi),
                "neighbors_within_iso": near,
                "correct_notes_within_iso": near_corr,
            })

    # cleanest first: few error neighbours, small onset gap, some correct
    # context so the excerpt is not empty, modest pitch gap for legibility.
    cands.sort(key=lambda c: (c["neighbors_within_iso"],
                              c["onset_gap_ms"],
                              -min(c["correct_notes_within_iso"], 8),
                              c["pitch_gap"]))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"n_candidates": len(cands), "candidates": cands[:40]}, fh,
                  indent=2)
    print(f"{len(cands)} localized-but-misclassified pairs; top:")
    for c in cands[:6]:
        print("  %s  gap=%.1fms pitchd=%d  ref=%s pred=%s  neigh=%d corr=%d"
              % (c["piece"][:34], c["onset_gap_ms"], c["pitch_gap"],
                 c["ref_type"], c["pred_type"], c["neighbors_within_iso"],
                 c["correct_notes_within_iso"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
