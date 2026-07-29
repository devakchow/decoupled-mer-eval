#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""revision_analysis.py -- SPL revision-round analyses (reviewer items M2, M3).

Read-only companion to ``decoupled_scorer.py`` (imported, never modified).
Produces two NEW analyses from the released artifacts; every recomputed
aggregate is regression-gated against the shipped result JSON (exact integer
equality on the tau=50 ms confusion) before any stratified number is emitted.

ANALYSIS 1 (M2, ``pitch`` subcommand) -- pitch-stratified off-diagonal
----------------------------------------------------------------------
Re-scores one configuration from the raw prediction MIDIs + ground-truth
metadata with the released scorer (v1.1.0 pipeline, tau=50 ms, eps=50 ms,
strict collapse), recording per matched pair:
(ref_etype, pred_etype, ref_pitch, pred_pitch, onset_distance).
Off-diagonal (localized-but-misclassified) pairs are stratified into
equal-pitch vs unequal-pitch, per confusion cell and overall, with
onset-distance median/IQR per stratum and the HM that survives if
unequal-pitch pairs are discarded.

STRUCTURAL FACT (verified by a runnable assertion, see
``_structural_selfcheck``): the wrong-note collapse replaces a co-located
(missed, extra) pair by a single ``wrong`` event that carries the PLAYED
(extra) half's onset AND pitch (decoupled_scorer.collapse_wrong, the
``Event(x.onset_s, x.pitch_midi, "wrong", ...)`` construction); the
missed-half (score) pitch is discarded. Therefore, for the dominant cell
(ref extra -> pred wrong), the recorded pred_pitch IS the collapsed event's
extra-half pitch, and ref_pitch vs pred_pitch compares two played-note
pitches -- the meaningful comparison. It is NOT equal by construction: the
strict collapse pairs any pitches within eps, so equal-pitch fractions
reported here are empirical. For cells with ref_etype == missed the ref
pitch is a score pitch while the wrong event's pitch is a played pitch;
pitch equality there mixes score-vs-played semantics (flagged in output).

ANALYSIS 2 (M3, ``excess`` subcommand) -- excess-over-null hidden mass
----------------------------------------------------------------------
Pure arithmetic on the existing 200-rotation circular-shift null artifacts
(figs/null_colocation.py outputs) and the shipped strict-eps05 results:

    HM_excess = (obs_off - null_mean_off) / (obs_matched - null_mean_matched)

plus the simple subtraction HM_obs - HM_null (null off-diagonal fraction),
and the enrichment ratios for cross-checking the letter's reported ranges.

Usage (pitch mode runs wherever the raw artifacts live, e.g. Gilbreth):

    python revision_analysis.py pitch \
        --pred_dir <preds/CONFIG> --gt_meta <gt_meta_maestro.json> \
        --expected <CONFIG_strict_eps05.json> --out <out.json> \
        [--tau 0.05] [--eps 0.05]

    python revision_analysis.py excess \
        --triple <label> <null_colo.json> <strict_eps05.json> [--triple ...] \
        --out <out.json>

No side effects on import. Depends only on decoupled_scorer's dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import decoupled_scorer as ds  # noqa: E402

ANALYSIS_TAU_S = 0.050
ANALYSIS_EPS_S = 0.050


# --------------------------------------------------------------------------- #
# Structural self-check: what pitch does a collapsed ``wrong`` event carry?   #
# --------------------------------------------------------------------------- #

def _structural_selfcheck() -> None:
    """Runnable proof of the collapse construction used in the M2 writeup.

    A co-located (missed@60, extra@64) pair collapses to ONE wrong event
    carrying the extra half's onset and pitch (64); the missed-half (score)
    pitch 60 is discarded. If a future scorer version changed this, the
    stratification semantics below would be wrong -- so we assert it.
    """
    evs = [ds.Event(1.000, 60, "missed", "p"), ds.Event(1.020, 64, "extra", "p")]
    out = ds.collapse_wrong(evs, epsilon=ANALYSIS_EPS_S, mode="strict")
    wrong = [e for e in out if e.etype == "wrong"]
    assert len(wrong) == 1, "collapse did not produce exactly one wrong event"
    assert wrong[0].pitch_midi == 64 and abs(wrong[0].onset_s - 1.020) < 1e-12, (
        "collapsed wrong event no longer carries the extra (played) half's "
        "onset+pitch -- stratification semantics invalid")
    assert len(out) == 1, "leftover halves after a clean collapse"


# --------------------------------------------------------------------------- #
# Analysis 1: per-pair recording + pitch stratification.                      #
# --------------------------------------------------------------------------- #

def matched_pairs(pred: Sequence[ds.Event], ref: Sequence[ds.Event],
                  tau: float, eps: float
                  ) -> Tuple[List[Tuple[str, str, int, int, float]], int, int]:
    """Replicates decoupled_scores' pipeline (strict collapse, type-ignored
    exact matching) and returns per-pair tuples
    (ref_etype, pred_etype, ref_pitch, pred_pitch, onset_dist_s)
    plus (n_pred_err, n_ref_err). Deterministic, identical semantics."""
    predc = ds._canonical(ds.collapse_wrong(pred, epsilon=eps, mode="strict"))
    refc = ds._canonical(ds.collapse_wrong(ref, epsilon=eps, mode="strict"))
    pred_err = [e for e in predc if e.etype in ds.ERROR_TYPES]
    ref_err = [e for e in refc if e.etype in ds.ERROR_TYPES]
    match = ds.match_events(pred_err, ref_err, tau, require_pitch=False)
    pairs = [(ref_err[ri].etype, pred_err[pi].etype,
              int(ref_err[ri].pitch_midi), int(pred_err[pi].pitch_midi),
              abs(pred_err[pi].onset_s - ref_err[ri].onset_s))
             for pi, ri in match]
    return pairs, len(pred_err), len(ref_err)


def _dist_stats_ms(dists_s: Sequence[float]) -> Optional[Dict[str, float]]:
    if not dists_s:
        return None
    import numpy as np
    a = np.asarray(dists_s, dtype=float) * 1000.0
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return dict(n=int(a.size), median_ms=float(med), q1_ms=float(q1),
                q3_ms=float(q3), iqr_ms=float(q3 - q1),
                mean_ms=float(a.mean()), max_ms=float(a.max()))


def _pitch_diff_stats(diffs: Sequence[int]) -> Optional[Dict[str, float]]:
    """Summary of |pitch difference| in semitones for the unequal stratum."""
    if not diffs:
        return None
    import numpy as np
    a = np.abs(np.asarray(diffs, dtype=float))
    return dict(n=int(a.size), median_semitones=float(np.median(a)),
                p90_semitones=float(np.percentile(a, 90)),
                frac_within_2_semitones=float((a <= 2).mean()),
                frac_octave_multiple=float(((a % 12) == 0).mean()))


def stratify(pairs: List[Tuple[str, str, int, int, float]]) -> Dict[str, object]:
    n_loc = len(pairs)
    off = [p for p in pairs if p[0] != p[1]]
    n_off = len(off)

    def stratum(sub):
        eq = [p for p in sub if p[2] == p[3]]
        ne = [p for p in sub if p[2] != p[3]]
        n = len(sub)
        return dict(
            n=n,
            equal_pitch=dict(count=len(eq),
                             fraction=(len(eq) / n) if n else None,
                             onset_dist=_dist_stats_ms([p[4] for p in eq])),
            unequal_pitch=dict(count=len(ne),
                               fraction=(len(ne) / n) if n else None,
                               onset_dist=_dist_stats_ms([p[4] for p in ne]),
                               abs_pitch_diff=_pitch_diff_stats(
                                   [p[2] - p[3] for p in ne])),
        )

    cells: Dict[str, object] = {}
    for rt in ds.ERROR_TYPES:
        for pt in ds.ERROR_TYPES:
            if rt == pt:
                continue
            sub = [p for p in off if p[0] == rt and p[1] == pt]
            if sub:
                cells["%s->%s" % (rt, pt)] = stratum(sub)

    eq_off = sum(1 for p in off if p[2] == p[3])
    ne_off = n_off - eq_off
    hm_obs = (n_off / n_loc) if n_loc else None
    return dict(
        n_localized=n_loc,
        off_diagonal_total=n_off,
        hm_observed=hm_obs,
        overall_off_diagonal=stratum(off),
        per_cell=cells,
        dominant_cell="extra->wrong",
        hm_if_unequal_pitch_discarded=dict(
            equal_pitch_off_diagonal=eq_off,
            unequal_pitch_off_diagonal=ne_off,
            hm_numerator_only=(eq_off / n_loc) if n_loc else None,
            hm_pairs_removed=(eq_off / (n_loc - ne_off))
            if (n_loc - ne_off) else None,
            surviving_fraction_of_hm=(eq_off / n_off) if n_off else None,
        ),
        pitch_semantics_note=(
            "pred_pitch of a 'wrong' event is the collapsed pair's PLAYED "
            "(extra-half) pitch by construction; the missed-half (score) "
            "pitch is discarded. extra->wrong therefore compares two "
            "played-note pitches (meaningful). Cells with ref_etype='missed' "
            "compare a score pitch to a played pitch (interpret with care)."),
    )


def run_pitch(args: argparse.Namespace) -> int:
    _structural_selfcheck()
    tau, eps = ds._to_seconds(args.tau), ds._to_seconds(args.eps)

    pred = ds.load_pred_events(args.pred_dir)
    ref = ds.load_ref_events(args.gt_meta)

    # Regression gate 1: full released-scorer result at this tau.
    res = ds.decoupled_scores(pred, ref, tau=tau, epsilon=eps, collapse="strict")

    # Regression gate 2: shipped aggregate JSON must match EXACTLY.
    with open(args.expected, "r", encoding="utf-8") as fh:
        expected = json.load(fh)
    exp_tau = [t for t in expected["decoupled"]["per_tau"]
               if t["tau_ms"] == int(round(tau * 1000))]
    if len(exp_tau) != 1:
        raise RuntimeError("expected JSON has no unique tau=%d ms entry"
                           % int(round(tau * 1000)))
    exp = exp_tau[0]
    if exp["confusion"] != res.confusion or \
            exp["localization"]["tp"] != res.localization["tp"]:
        raise RuntimeError(
            "recomputed confusion does not match shipped artifact %s -- "
            "refusing to emit stratified numbers" % args.expected)

    pairs, n_pred_err, n_ref_err = matched_pairs(pred, ref, tau, eps)
    if len(pairs) != res.n_localized:
        raise RuntimeError("pair recorder disagrees with scorer on |M(tau)|")
    conf_check: Dict[str, int] = {}
    for rt, pt, _rp, _pp, _d in pairs:
        key = "%s->%s" % (rt, pt)
        conf_check[key] = conf_check.get(key, 0) + 1
    if conf_check != {k: v for k, v in exp["confusion_sparse"].items()}:
        raise RuntimeError("pair recorder confusion != shipped confusion_sparse")

    out = dict(
        analysis="M2 pitch-stratified off-diagonal",
        scorer_version=ds.__version__,
        config=dict(pred_dir=os.path.abspath(args.pred_dir),
                    gt_meta=os.path.abspath(args.gt_meta),
                    expected=os.path.abspath(args.expected),
                    tau_s=tau, epsilon_s=eps, collapse="strict"),
        regression_gate="EXACT match vs shipped confusion at tau=%d ms"
                        % int(round(tau * 1000)),
        n_pred_err=n_pred_err,
        n_ref_err=n_ref_err,
        stratified=stratify(pairs),
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["stratified"], indent=2))
    print("wrote %s" % os.path.abspath(args.out))
    return 0


# --------------------------------------------------------------------------- #
# Analysis 2: excess-over-null hidden mass from existing null artifacts.      #
# --------------------------------------------------------------------------- #

def excess_for(null_path: str, result_path: str) -> Dict[str, object]:
    with open(null_path, "r", encoding="utf-8") as fh:
        null = json.load(fh)
    with open(result_path, "r", encoding="utf-8") as fh:
        result = json.load(fh)
    tau_ms = int(round(null["tau"] * 1000))
    exp = [t for t in result["decoupled"]["per_tau"] if t["tau_ms"] == tau_ms]
    if len(exp) != 1:
        raise RuntimeError("no unique tau=%d ms entry in %s" % (tau_ms, result_path))
    exp = exp[0]
    obs_off = sum(exp["confusion"][rt][pt] for rt in ds.ERROR_TYPES
                  for pt in ds.ERROR_TYPES if rt != pt)
    obs_tot = exp["n_localized"]
    # Cross-check the null artifact's own observed numbers against the
    # shipped aggregate (they were computed independently).
    if null["observed_off_diagonal"] != obs_off or \
            null["observed_matched_total"] != obs_tot:
        raise RuntimeError(
            "null artifact observed counts disagree with shipped result "
            "(%s vs %s)" % (null_path, result_path))
    null_off = null["null_off_diagonal"]["mean"]
    null_tot = null["null_matched_total"]["mean"]
    hm_obs = obs_off / obs_tot
    hm_null = null_off / null_tot
    hm_excess = (obs_off - null_off) / (obs_tot - null_tot)
    return dict(
        tau_ms=tau_ms,
        n_perm=null["n_perm"],
        observed=dict(off_diagonal=obs_off, matched_total=obs_tot,
                      hm=hm_obs),
        null_mean=dict(off_diagonal=null_off, matched_total=null_tot,
                       hm_null=hm_null),
        enrichment=dict(off=obs_off / null_off, total=obs_tot / null_tot,
                        p_off_ge_observed=null["p_off_ge_observed"],
                        p_total_ge_observed=null["p_total_ge_observed"]),
        hm_excess_over_null=hm_excess,
        hm_simple_subtraction=hm_obs - hm_null,
        note=("hm_excess_over_null = (obs_off - null_off) / "
              "(obs_matched - null_matched): the misclassified fraction of "
              "the ABOVE-CHANCE matched mass. hm_simple_subtraction can be "
              "negative because chance co-locations are mostly "
              "misclassified (hm_null >> hm_obs); it is reported for "
              "completeness, not as the headline quantity."),
    )


def run_excess(args: argparse.Namespace) -> int:
    out = dict(
        analysis="M3 excess-over-null hidden mass",
        scorer_version=ds.__version__,
        configurations={label: excess_for(null_path, result_path)
                        for label, null_path, result_path in args.triple},
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote %s" % os.path.abspath(args.out))
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="revision_analysis.py",
        description="SPL revision-round analyses M2 (pitch-stratified "
                    "off-diagonal) and M3 (excess-over-null HM).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pitch", help="M2: pitch-stratified off-diagonal")
    p1.add_argument("--pred_dir", required=True)
    p1.add_argument("--gt_meta", required=True)
    p1.add_argument("--expected", required=True,
                    help="shipped strict-eps05 result JSON (regression gate)")
    p1.add_argument("--tau", type=float, default=ANALYSIS_TAU_S,
                    help=">1 = ms, <=1 = s (default 0.05)")
    p1.add_argument("--eps", type=float, default=ANALYSIS_EPS_S,
                    help=">1 = ms, <=1 = s (default 0.05)")
    p1.add_argument("--out", required=True)

    p2 = sub.add_parser("excess", help="M3: excess-over-null hidden mass")
    p2.add_argument("--triple", nargs=3, action="append", required=True,
                    metavar=("LABEL", "NULL_JSON", "RESULT_JSON"))
    p2.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "pitch":
        return run_pitch(args)
    return run_excess(args)


if __name__ == "__main__":
    sys.exit(main())
