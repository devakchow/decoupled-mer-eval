#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""comparator_mireval_onset.py -- THIRD-PARTY localization comparator (IR-11).

Purpose (desk-reject row IR-11, "insufficient numerical data and comparisons"):
compute an independent, third-party comparator on EXACTLY the split scored in
Table II (MAESTRO-E test, n=177 pieces): onset-only, pitch-ignoring note
matching via ``mir_eval.transcription.match_note_onsets`` -- mir_eval's own
maximum bipartite matcher, not ours -- evaluated on the same event population
as the paper's Loc-F.

WHY THIS COMPARATOR. Of the three candidates named in the review dossier:
  * mir_eval onset-only matching: CPU-only, pip-installable, operates on the
    exact (onset, pitch) event lists our pipeline already loads. FEASIBLE.
  * MV2H: requires Java + full symbolic-alignment inputs we do not have.
  * PSDS: requires sound-event detection rosters; not applicable to
    note-level MER events.

BLOCKED-CLUSTER STATUS (2026-07-31). The raw predicted multi-track MIDI and
ground-truth MIDI live ONLY on WSL (/home/devak/mer/run/...) and Gilbreth
(/scratch/gilbreth/dcharapa/mer/run/...). The Windows workstation holds
aggregate JSONs (counts and confusion matrices) from which onset lists cannot
be reconstructed. This script is therefore READY TO RUN where the data lives;
it has been smoke-tested locally on synthetic events only (``--self-check``).

WHAT IT COMPUTES (per system, at each tau; default tau = 50 ms):
  * variant "postcollapse": the paper's primary event population -- both
    sides collapsed (strict, eps = 50 ms) exactly as in Table II -- matched by
    the THIRD-PARTY matcher instead of ours. This is the apples-to-apples
    comparator column for Loc-F@50.
  * variant "raw": the un-collapsed missed+extra error union, same matcher.
  * pooled (micro) Loc-P/R/F1; per-piece bootstrap 95% CI, 1000 resamples,
    seed 20260718 -- identical conventions to Table II.
  * a per-piece cross-check against our exact max-cardinality matcher
    (``decoupled_scorer.match_events``). Both matchers are maximum-cardinality
    on (almost) the same feasibility graph, so TP should agree piece-by-piece
    up to a documented boundary-semantics difference: mir_eval rounds onset
    distances to 4 decimals before the tau comparison (N_DECIMALS = 4),
    whereas the decoupled matcher uses unrounded distances with 1e-9 slack.
    Disagreements are COUNTED AND REPORTED, never hidden.

SPLIT-IDENTITY GUARANTEE. Pieces are keyed by id (gt_meta keys / MIDI file
stems), never positionally. The output records the sorted piece-id list and
its SHA-256 fingerprint so the comparator run can be verified against the
track-census artifacts for the same system.

Run on WSL/Gilbreth (mirror of wsl_smoke/t8_score.sh conventions):

    python comparator_mireval_onset.py \
        --pred_dir /home/devak/mer/run/preds/A_polytune_maestro \
        --gt_meta  /home/devak/mer/run/gt_meta_maestro.json \
        --tau 50 --eps 0.05 --collapse strict \
        --bootstrap 1000 --seed 20260718 \
        --out results/gilbreth/A_polytune_maestro_comparator_mireval.json

    python comparator_mireval_onset.py --self-check

Output naming convention (consumed later by figs/make_tables.py once real
numbers exist): ``<stem>_comparator_mireval.json`` next to the system's other
gilbreth artifacts. Tables are NEVER hand-edited; the table column is added by
extending the generator only after this JSON exists for all three systems.

Dependencies: numpy, mir_eval (matcher under test), pretty_midi (loader),
scipy (via decoupled_scorer import).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse the paper pipeline's loaders / collapse / matcher so the comparator is
# guaranteed to see the identical split and event population.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoupled_scorer import (  # noqa: E402
    DEFAULT_EPSILON_S,
    ERROR_TYPES,
    Event,
    collapse_wrong,
    load_pred_events,
    load_ref_events,
    match_events,
    prf,
)

__version__ = "1.0.0"

# Fixed nominal duration for interval construction; match_note_onsets ignores
# offsets entirely, this only builds the (n, 2) interval arrays mir_eval wants.
_NOTE_DUR_S = 0.1


def _intervals(events: Sequence[Event]) -> np.ndarray:
    if not events:
        return np.zeros((0, 2), dtype=float)
    on = np.array([e.onset_s for e in events], dtype=float)
    return np.stack([on, on + _NOTE_DUR_S], axis=1)


def mireval_onset_tp(pred: Sequence[Event], ref: Sequence[Event],
                     tau: float) -> int:
    """Third-party matched-pair count: mir_eval.transcription.match_note_onsets
    (maximum bipartite matching on |ref_onset - est_onset| <= tau, rounded to
    4 decimals inside mir_eval; pitch is ignored by construction)."""
    import mir_eval  # the matcher under comparison; import here so
    # --self-check failure messages are explicit if it is missing

    if not pred or not ref:
        return 0
    matching = mir_eval.transcription.match_note_onsets(
        _intervals(ref), _intervals(pred), onset_tolerance=tau)
    return len(matching)


def _piece_fingerprint(pieces: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(pieces)).encode("utf-8")).hexdigest()


def _population(pred: Sequence[Event], ref: Sequence[Event], variant: str,
                epsilon: float, collapse: str
                ) -> Tuple[List[Event], List[Event]]:
    """Event population for one variant.

    postcollapse: symmetric pre-match collapse on both sides (paper primary);
    raw: un-collapsed missed+extra union.
    """
    if variant == "postcollapse":
        p = [e for e in collapse_wrong(pred, epsilon=epsilon, mode=collapse)
             if e.etype in ERROR_TYPES]
        r = [e for e in collapse_wrong(ref, epsilon=epsilon, mode=collapse)
             if e.etype in ERROR_TYPES]
    elif variant == "raw":
        p = [e for e in pred if e.etype in ("missed", "extra")]
        r = [e for e in ref if e.etype in ("missed", "extra")]
    else:
        raise ValueError("unknown variant %r" % variant)
    return p, r


def score_variant(pred: Sequence[Event], ref: Sequence[Event], tau: float,
                  variant: str, epsilon: float, collapse: str,
                  n_boot: int, seed: int) -> Dict[str, object]:
    """Pooled micro P/R/F via the third-party matcher + per-piece bootstrap CI
    + per-piece cross-check against the paper's exact matcher."""
    p_all, r_all = _population(pred, ref, variant, epsilon, collapse)
    pieces = sorted({e.piece for e in p_all} | {e.piece for e in r_all})

    per_piece = []           # (tp_mireval, n_pred, n_ref)
    tp_ours_total = 0
    n_disagree = 0
    disagree_pieces: List[Dict[str, object]] = []
    for pc in pieces:
        pp = [e for e in p_all if e.piece == pc]
        rp = [e for e in r_all if e.piece == pc]
        tp_me = mireval_onset_tp(pp, rp, tau)
        tp_ours = len(match_events(pp, rp, tau, require_pitch=False))
        tp_ours_total += tp_ours
        if tp_me != tp_ours:
            n_disagree += 1
            disagree_pieces.append(dict(piece=pc, tp_mireval=tp_me,
                                        tp_ours=tp_ours))
        per_piece.append((tp_me, len(pp), len(rp)))

    arr = np.array(per_piece, dtype=float) if per_piece else np.zeros((0, 3))
    tp = int(arr[:, 0].sum())
    n_pred = int(arr[:, 1].sum())
    n_ref = int(arr[:, 2].sum())
    p, r, f = prf(tp, n_pred - tp, n_ref - tp)

    rng = np.random.default_rng(seed)
    f_reps: List[float] = []
    n_pieces = len(pieces)
    for _ in range(n_boot):
        idx = rng.integers(0, n_pieces, size=n_pieces) if n_pieces else []
        s = arr[idx].sum(axis=0) if n_pieces else np.zeros(3)
        btp, bnp, bnr = s
        f_reps.append(prf(int(btp), int(bnp - btp), int(bnr - btp))[2])
    ci = (None if not f_reps else
          [float(v) for v in np.percentile(f_reps, [2.5, 97.5])])

    return dict(
        variant=variant,
        tau_s=tau,
        n_pieces=n_pieces,
        n_pred_err=n_pred,
        n_ref_err=n_ref,
        loc_tp=tp,
        loc_precision=p,
        loc_recall=r,
        loc_f1=f,
        loc_f1_ci95=ci,
        n_boot=n_boot,
        seed=seed,
        crosscheck=dict(
            tp_ours_total=tp_ours_total,
            tp_mireval_total=tp,
            n_pieces_disagreeing=n_disagree,
            disagreeing_pieces=disagree_pieces,
            note="mir_eval rounds onset distances to 4 decimals "
                 "(N_DECIMALS=4); the decoupled matcher uses unrounded "
                 "distances with 1e-9 slack. Any disagreement is confined "
                 "to onset gaps in (tau+1e-9, tau+5e-5].",
        ),
    )


def run_self_check(verbose: bool = True) -> int:
    """Synthetic smoke test of the comparator machinery. This validates the
    SCRIPT, not the paper: it fabricates no corpus numbers."""
    import mir_eval  # noqa: F401  -- fail loudly if absent

    def log(m: str) -> None:
        if verbose:
            print(m)

    # Deterministic random corpora: third-party TP must equal our exact
    # matcher's TP on well-separated data, and stay within the documented
    # boundary window otherwise (both are maximum-cardinality matchers).
    rng = np.random.default_rng(20260718)
    for trial in range(20):
        events_p, events_r = [], []
        for pc in ("p0", "p1", "p2"):
            n = int(rng.integers(3, 25))
            # onsets on a 10 ms lattice, far from the 50 ms tau boundary
            for _ in range(n):
                t0 = round(float(rng.integers(0, 400)) * 0.01, 3)
                et = str(rng.choice(("missed", "extra")))
                events_r.append(Event(t0, int(rng.integers(50, 80)), et, pc))
                if rng.random() < 0.8:
                    jit = round(float(rng.integers(-3, 4)) * 0.01, 3)
                    events_p.append(Event(round(t0 + jit, 3),
                                          int(rng.integers(50, 80)),
                                          str(rng.choice(("missed", "extra"))),
                                          pc))
        res = score_variant(events_p, events_r, tau=0.05, variant="raw",
                            epsilon=DEFAULT_EPSILON_S, collapse="strict",
                            n_boot=50, seed=1)
        cc = res["crosscheck"]
        assert cc["n_pieces_disagreeing"] == 0, (trial, cc)
        assert cc["tp_ours_total"] == cc["tp_mireval_total"]
    log("SELF-CHECK A: 20 random 3-piece corpora, lattice onsets -- "
        "third-party TP == exact-matcher TP on every piece. OK")

    # Collapse population parity: postcollapse variant must see the same
    # event counts our pipeline's collapse produces.
    ref = [Event(1.0, 61, "missed"), Event(1.0, 60, "extra")]
    pred = [Event(1.0, 60, "extra")]
    res = score_variant(pred, ref, tau=0.05, variant="postcollapse",
                        epsilon=DEFAULT_EPSILON_S, collapse="strict",
                        n_boot=10, seed=1)
    assert res["n_ref_err"] == 1 and res["n_pred_err"] == 1  # collapsed
    assert res["loc_tp"] == 1 and res["loc_f1"] == 1.0
    log("SELF-CHECK B: wrong-note collapse parity (A2 worked example) -- "
        "ref collapses to 1 wrong, third-party Loc-F = 1.0. OK")

    # Empty-side behavior.
    res = score_variant([], ref, tau=0.05, variant="raw",
                        epsilon=DEFAULT_EPSILON_S, collapse="strict",
                        n_boot=10, seed=1)
    assert res["loc_tp"] == 0 and res["loc_f1"] == 0.0
    log("SELF-CHECK C: empty prediction side -- TP=0, F=0, no crash. OK")

    log("SELF-CHECK PASSED (comparator_mireval_onset v%s)." % __version__)
    return 0


def _to_seconds(v: float) -> float:
    """Same unit convention as decoupled_scorer: >1 = ms, <=1 = s."""
    return v / 1000.0 if v > 1.0 else v


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="comparator_mireval_onset.py",
        description="Third-party (mir_eval) onset-only localization "
                    "comparator on the Table II split (IR-11).")
    ap.add_argument("--pred_dir", help="directory of predicted multi-track MIDI")
    ap.add_argument("--gt_meta", help="ground-truth metadata JSON")
    ap.add_argument("--tau", nargs="+", type=float, default=[50.0],
                    help="onset tolerances; >1 = ms, <=1 = s (default: 50)")
    ap.add_argument("--eps", type=float, default=DEFAULT_EPSILON_S,
                    help="collapse radius for the postcollapse variant "
                         "(default 0.05 s)")
    ap.add_argument("--collapse", choices=("strict", "pitch_aware"),
                    default="strict")
    ap.add_argument("--bootstrap", type=int, default=1000, metavar="N",
                    help="per-piece bootstrap replicates (default 1000, "
                         "matching Table II)")
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--out", default="comparator_mireval.json")
    ap.add_argument("--self-check", action="store_true",
                    help="run the synthetic smoke test and exit")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.self_check:
        return run_self_check(verbose=True)
    if not args.pred_dir or not args.gt_meta:
        build_arg_parser().error("--pred_dir and --gt_meta are required "
                                 "(or use --self-check)")

    import mir_eval

    taus = [_to_seconds(t) for t in args.tau]
    eps = _to_seconds(args.eps)

    pred = load_pred_events(args.pred_dir)
    ref = load_ref_events(args.gt_meta)
    pred_pieces = {e.piece for e in pred}
    ref_pieces = {e.piece for e in ref}
    if pred_pieces != ref_pieces:
        raise RuntimeError(
            "piece-id mismatch between pred_dir and gt_meta: "
            "%d pred-only, %d ref-only -- refusing to score a different split"
            % (len(pred_pieces - ref_pieces), len(ref_pieces - pred_pieces)))
    pieces = sorted(ref_pieces)

    payload: Dict[str, object] = dict(
        config=dict(
            comparator_version=__version__,
            mir_eval_version=mir_eval.__version__,
            matcher="mir_eval.transcription.match_note_onsets "
                    "(onset-only, pitch-ignoring, maximum bipartite)",
            pred_dir=os.path.abspath(args.pred_dir),
            gt_meta=os.path.abspath(args.gt_meta),
            tau_s=taus,
            epsilon_s=eps,
            collapse=args.collapse,
        ),
        n_pieces=len(pieces),
        piece_ids=pieces,
        piece_fingerprint_sha256=_piece_fingerprint(pieces),
        results=[score_variant(pred, ref, tau=t, variant=v, epsilon=eps,
                               collapse=args.collapse, n_boot=args.bootstrap,
                               seed=args.seed)
                 for t in taus for v in ("postcollapse", "raw")],
    )

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("comparator_mireval_onset v%s: %d pieces (fingerprint %s)"
          % (__version__, len(pieces),
             payload["piece_fingerprint_sha256"][:16]))
    for r in payload["results"]:  # type: ignore[union-attr]
        print("  tau=%3dms %-12s Loc-F=%.4f CI=%s disagreeing-pieces=%d"
              % (int(round(r["tau_s"] * 1000)), r["variant"], r["loc_f1"],
                 r["loc_f1_ci95"], r["crosscheck"]["n_pieces_disagreeing"]))
    print("  wrote %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
