#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic_oracle.py -- Controlled-injection oracle for the decoupled MER scorer.

Paper: "A Decoupled Localization-Classification Measure for Music Error Detection"
Target: IEEE Signal Processing Letters.

WHY THIS FILE EXISTS
--------------------
The shipped MER evaluators (Polytune, LadderSym, RUMAA) score each error type as an
INDEPENDENT mir_eval transcription problem in a per-track loop at a fixed 50 ms onset
tolerance. A note flagged at the correct onset but assigned the WRONG error type is
therefore charged twice -- a false positive in one track and a false negative in
another -- and never recorded as ONE "localized-but-misclassified" event.

This oracle is MODEL-INDEPENDENT. It does not run any neural network. It injects
errors of KNOWN type into a synthetic clean corpus and passes them through a KNOWN
labelling/confusion process, so the ground-truth localized-but-misclassified mass
HM* is set by us. It then demonstrates three things end to end:

  A. DOUBLE-COUNT WITNESS. On a single planted event, the shipped per-track scorer
     records 1 FP + 1 FN across two tracks (F1 = 0 in BOTH tracks), while the
     decoupled scorer recovers EXACTLY ONE localized-but-misclassified event.

  B. NON-IDENTIFIABILITY WITNESS. Two synthetic systems are constructed with
     IDENTICAL per-track F1 vectors but DIFFERENT true confusion: HM = 1/3 vs HM = 0.
     The shipped metric cannot tell them apart; the decoupled scorer separates them.
     This is the physical witness of the marginal-sufficiency (non-identifiability)
     proposition that anchors the paper's signal-processing framing.

  C. RECOVERY-WITHIN-TOLERANCE. On a jittered corpus with a planted confusion
     kernel, the decoupled scorer recovers the planted HM* across the onset-tolerance
     sweep tau in {50,75,100,150,200,500} ms, while the shipped per-track F1 is blind
     to HM*. Emits the data for the killer figure (HM(tau) curve + confusion heatmap).

All numeric expectations in A and B are hand-derived (integer, exact) and asserted,
so a clean run is a machine-checked proof of the paper's core claims. See
experiments/README_oracle.md for the derivations and the figure spec.

DEPENDENCIES: Python 3.8+ standard library only (no numpy, no mir_eval required).
An optional mir_eval cross-check of the shipped mode runs if mir_eval is importable.

USAGE:
    python3 synthetic_oracle.py            # run A, B, C; print report; write figure CSVs
    python3 synthetic_oracle.py --quiet    # assertions only, minimal output

OUTPUTS (written next to this file):
    oracle_hm_sweep.csv        # figure LEFT panel: HM(tau) per system
    oracle_confusion.csv       # figure RIGHT panel: onset-matched confusion cells
    oracle_report.json         # every scored quantity, machine-readable
"""

import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Event model (Section 4.1 of the draft).                                     #
# --------------------------------------------------------------------------- #
# An event is (onset_seconds, pitch_midi, type). type in {"missed","extra",     #
# "wrong","correct"}. The shipped systems emit only {missed, extra, correct};   #
# "wrong" is DERIVED by collapsing a co-located missed+extra pair (Section 4.3).#

ERROR_TYPES = ("missed", "extra", "wrong")   # first-class error labels after collapse
SHIP_TRACKS = ("missed", "extra", "correct")  # the three tracks shipped systems score


@dataclass(frozen=True)
class Event:
    onset: float
    pitch: int
    etype: str
    piece: str = "p0"


# --------------------------------------------------------------------------- #
# Optimal one-to-one onset matching (Section 4.2).                            #
# --------------------------------------------------------------------------- #
# We compute a MAXIMUM-CARDINALITY one-to-one matching on the graph of pairs   #
# whose onset distance <= tau, breaking ties by MINIMUM total onset distance,  #
# so the matched count is monotone non-decreasing in tau (the sweep is         #
# interpretable) and deterministic (mir_eval's matcher is order-dependent).    #
#                                                                              #
# Implementation note: on well-separated onsets (every candidate has a unique  #
# nearest partner, as in all corpora this file builds), a greedy pass in       #
# ascending onset-distance order is provably a min-distance maximum matching.  #
# The production scorer should delegate to mir_eval.util.match_events /        #
# scipy.optimize.linear_sum_assignment; this stdlib version exists so the      #
# oracle runs with zero third-party dependencies.                              #

def match_onsets(
    preds: Sequence[Event],
    refs: Sequence[Event],
    tau: float,
    require_pitch: bool = False,
) -> List[Tuple[int, int]]:
    """Return list of (pred_index, ref_index) matched pairs.

    onset match: |onset_p - onset_r| <= tau (with a tiny epsilon for float safety).
    require_pitch: if True, also require pitch_p == pitch_r (the shipped mir_eval
                   onset-AND-pitch semantics used in shipped-metric mode).
    """
    eps = 1e-9
    cands = []  # (distance, tie_pitch, p_idx, r_idx)
    for pi, p in enumerate(preds):
        for ri, r in enumerate(refs):
            if p.piece != r.piece:      # matching is within a piece (Section 4.7)
                continue
            d = abs(p.onset - r.onset)
            if d <= tau + eps and (not require_pitch or p.pitch == r.pitch):
                # secondary tie-break by pitch distance keeps chords deterministic
                cands.append((d, abs(p.pitch - r.pitch), pi, ri))
    cands.sort()
    used_p, used_r = set(), set()
    matches = []
    for d, _pd, pi, ri in cands:
        if pi in used_p or ri in used_r:
            continue
        used_p.add(pi)
        used_r.add(ri)
        matches.append((pi, ri))
    return matches


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, recall, F1 with the 0/0 -> 0 convention."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# --------------------------------------------------------------------------- #
# SHIPPED-METRIC MODE (Section 4.6): what Polytune / LadderSym / RUMAA do.     #
# --------------------------------------------------------------------------- #
# For each track T in {missed, extra, correct}, run an independent onset+pitch  #
# note match at tau and report per-track P/R/F1. No cross-track assignment, no  #
# wrong-note collapse. This is the construction the paper re-scores.           #

def ship_track_events(events: Sequence[Event], track: str) -> List[Event]:
    return [e for e in events if e.etype == track]


def shipped_scores(
    pred: Sequence[Event],
    ref: Sequence[Event],
    tau: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Per-track P/R/F1 exactly as the shipped per-track loop computes them."""
    out = {}
    for track in SHIP_TRACKS:
        pT = ship_track_events(pred, track)
        rT = ship_track_events(ref, track)
        m = match_onsets(pT, rT, tau, require_pitch=True)
        tp = len(m)
        fp = len(pT) - tp
        fn = len(rT) - tp
        p, r, f = prf(tp, fp, fn)
        out[track] = dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f)
    out["_mean_f1"] = sum(out[t]["f1"] for t in SHIP_TRACKS) / len(SHIP_TRACKS)
    # Aggregate over ERROR tracks only (the Polytune "Error Detection F1" family
    # excludes the correct track from the headline error number).
    out["_mean_error_f1"] = (out["missed"]["f1"] + out["extra"]["f1"]) / 2.0
    return out


# --------------------------------------------------------------------------- #
# WRONG-NOTE COLLAPSE (Section 4.3).                                           #
# --------------------------------------------------------------------------- #
# A co-located (missed, extra) pair within radius epsilon collapses to a single #
# "wrong" event. Runs on BOTH pred and ref BEFORE matching (symmetric,          #
# pre-match), at a fixed epsilon INDEPENDENT of the swept tau, so the event     #
# population is constant across the sweep. pitch_aware=True additionally        #
# requires the missed (score) pitch != extra (played) pitch.                    #

def collapse_wrong(
    events: Sequence[Event],
    epsilon: float = 0.05,
    pitch_aware: bool = False,
) -> List[Event]:
    missed = [e for e in events if e.etype == "missed"]
    extra = [e for e in events if e.etype == "extra"]
    passthrough = [e for e in events if e.etype not in ("missed", "extra")]

    # candidate co-locations, chosen by min total onset distance (then pitch dist)
    eps = 1e-9
    cands = []
    for mi, m in enumerate(missed):
        for xi, x in enumerate(extra):
            if m.piece != x.piece:      # collapse only within a piece (Section 4.7)
                continue
            d = abs(m.onset - x.onset)
            if d <= epsilon + eps and (not pitch_aware or m.pitch != x.pitch):
                cands.append((d, abs(m.pitch - x.pitch), mi, xi))
    cands.sort()
    used_m, used_x = set(), set()
    wrong = []
    for _d, _pd, mi, xi in cands:
        if mi in used_m or xi in used_x:
            continue
        used_m.add(mi)
        used_x.add(xi)
        m, x = missed[mi], extra[xi]
        # a wrong event carries the played (extra) pitch; score pitch kept as tag
        wrong.append(Event(onset=x.onset, pitch=x.pitch, etype="wrong", piece=x.piece))
    leftover_missed = [m for i, m in enumerate(missed) if i not in used_m]
    leftover_extra = [x for i, x in enumerate(extra) if i not in used_x]
    return passthrough + leftover_missed + leftover_extra + wrong


# --------------------------------------------------------------------------- #
# DECOUPLED SCORER (Sections 4.2, 4.3, 4.5).                                   #
# --------------------------------------------------------------------------- #

@dataclass
class DecoupledResult:
    tau: float
    localization: Dict[str, float]                 # precision/recall/f1 + counts
    confusion: Dict[Tuple[str, str], int]          # (ref_type, pred_type) -> count
    false_alarm_on_correct: int                    # pred error onset-matched a correct note
    spurious: int                                  # pred error matched nothing
    miss_margin: int                               # ref error matched nothing
    n_localized: int                               # |M(tau)|
    hm: Optional[float]                            # localized-but-misclassified fraction
    n_pred_err: int
    n_ref_err: int

    def mass_conserved(self) -> bool:
        """The named invariant (Section 4.3): every event lands in exactly one cell."""
        conf_total = sum(self.confusion.values())
        pred_accounted = conf_total + self.false_alarm_on_correct + self.spurious
        ref_accounted = conf_total + self.miss_margin
        return pred_accounted == self.n_pred_err and ref_accounted == self.n_ref_err


def decoupled_scores(
    pred: Sequence[Event],
    ref: Sequence[Event],
    tau: float,
    epsilon: float = 0.05,
    pitch_aware: bool = False,
) -> DecoupledResult:
    # 1. collapse wrong-note pairs on both sides, BEFORE matching
    predc = collapse_wrong(pred, epsilon=epsilon, pitch_aware=pitch_aware)
    refc = collapse_wrong(ref, epsilon=epsilon, pitch_aware=pitch_aware)

    pred_err = [e for e in predc if e.etype in ERROR_TYPES]
    ref_err = [e for e in refc if e.etype in ERROR_TYPES]
    ref_correct = [e for e in refc if e.etype == "correct"]

    # 2. localization: one-to-one onset match, ANY type, on the error union
    match = match_onsets(pred_err, ref_err, tau, require_pitch=False)
    matched_p = {pi for pi, _ in match}
    matched_r = {ri for _, ri in match}
    tp = len(match)
    fp = len(pred_err) - tp
    fn = len(ref_err) - tp
    lp, lr, lf = prf(tp, fp, fn)

    # 3. confusion on onset-matched events (Section 4.3)
    confusion: Dict[Tuple[str, str], int] = {}
    for pi, ri in match:
        key = (ref_err[ri].etype, pred_err[pi].etype)
        confusion[key] = confusion.get(key, 0) + 1

    # 4. resolve the two margins. An unmatched predicted error that still onset-
    #    matches a CORRECT-track reference note (second pass at the same tau) is a
    #    false-alarm-on-a-correctly-played-note; otherwise it is spurious.
    unmatched_pred = [pred_err[i] for i in range(len(pred_err)) if i not in matched_p]
    fa_on_correct = 0
    if ref_correct and unmatched_pred:
        m2 = match_onsets(unmatched_pred, ref_correct, tau, require_pitch=False)
        fa_on_correct = len(m2)
    spurious = len(unmatched_pred) - fa_on_correct
    miss_margin = fn  # ref errors matched by no predicted error

    # 5. hidden mass HM(tau): localized-but-misclassified fraction (Section 4.5)
    n_loc = tp
    off_diag = sum(c for (rt, pt), c in confusion.items() if rt != pt)
    hm = (off_diag / n_loc) if n_loc > 0 else None

    return DecoupledResult(
        tau=tau,
        localization=dict(tp=tp, fp=fp, fn=fn, precision=lp, recall=lr, f1=lf),
        confusion=confusion,
        false_alarm_on_correct=fa_on_correct,
        spurious=spurious,
        miss_margin=miss_margin,
        n_localized=n_loc,
        hm=hm,
        n_pred_err=len(pred_err),
        n_ref_err=len(ref_err),
    )


# --------------------------------------------------------------------------- #
# CORPUS + INJECTION RECIPE (README_oracle.md, Section "Injection recipe").    #
# --------------------------------------------------------------------------- #

def clean_corpus(n_notes: int, ioi: float = 0.25, base_pitch: int = 60,
                 piece: str = "p0") -> List[Event]:
    """A deterministic clean performance: n notes on a fixed grid, C-major scale.
    All events are 'correct' -- this is the aligned clean reference."""
    scale = [0, 2, 4, 5, 7, 9, 11]  # C major
    return [
        Event(onset=i * ioi, pitch=base_pitch + scale[i % len(scale)],
              etype="correct", piece=piece)
        for i in range(n_notes)
    ]


def inject_and_label(
    clean: Sequence[Event],
    plant: Sequence[Tuple[int, str, Optional[str]]],  # (idx, true_type, pred_type|None)
    onset_jitter: float = 0.0,               # seconds added to predicted onset
    piece: str = "p0",
) -> Tuple[List[Event], List[Event]]:
    """Return (reference_events, predicted_events) in the SHIPPED 3-track encoding.

    Each plant entry is (note_index, true_error_type, predicted_error_type). The
    reference encodes the TRUE type; the prediction encodes the SYSTEM's type, so
    off-diagonal (true != predicted) planted events ARE the localized-but-
    misclassified ground truth. pred_type=None models a pure localization miss
    (the system emits nothing). true 'missed'/'extra' are single-track entries;
    'wrong' is the co-located (missed=score_pitch, extra=played_pitch) pair.
    """
    by_idx = {i: e for i, e in enumerate(clean)}
    planted_idx = {i for i, _, _ in plant}
    ref: List[Event] = [e for i, e in by_idx.items() if i not in planted_idx]
    pred: List[Event] = list(ref)  # correct notes agreed on by both sides

    def emit(dst, onset, ttype, score_pitch, played_pitch):
        if ttype == "missed":
            dst.append(Event(onset, score_pitch, "missed", piece))
        elif ttype == "extra":
            dst.append(Event(onset, played_pitch, "extra", piece))
        elif ttype == "wrong":
            dst.append(Event(onset, score_pitch, "missed", piece))
            dst.append(Event(onset, played_pitch, "extra", piece))

    for i, ttype, ptype in plant:
        e = by_idx[i]
        played = e.pitch + 1  # played pitch for a wrong/extra note: one semitone up
        emit(ref, e.onset, ttype, e.pitch, played)
        if ptype is not None:
            emit(pred, e.onset + onset_jitter, ptype, e.pitch, played)
    return ref, pred


# --------------------------------------------------------------------------- #
# EXPERIMENT A -- the double-count witness.                                    #
# --------------------------------------------------------------------------- #

def experiment_A(quiet: bool = False) -> dict:
    """One reference error localized but mistyped. Shipped: 1 FP + 1 FN in two
    tracks (F1=0 in both). Decoupled: exactly ONE localized-but-misclassified
    event. Two variants: pure missed->extra (cleanest) and the draft's
    wrong->extra worked example."""
    results = {}

    # --- Variant A1: reference MISSED, model predicts EXTRA (clean 1FP+1FN) ---
    ref = [Event(1.0, 61, "missed")]
    pred = [Event(1.0, 61, "extra")]
    ship = shipped_scores(pred, ref, tau=0.05)
    dec = decoupled_scores(pred, ref, tau=0.05)

    # hand-derived expectations (see README_oracle.md, Table A1)
    assert ship["missed"]["fn"] == 1 and ship["missed"]["fp"] == 0
    assert ship["extra"]["fp"] == 1 and ship["extra"]["fn"] == 0
    assert ship["missed"]["f1"] == 0.0 and ship["extra"]["f1"] == 0.0  # penalized twice
    assert dec.n_localized == 1                                        # ONE event
    assert dec.confusion == {("missed", "extra"): 1}                  # off-diagonal
    assert dec.hm == 1.0                                              # fully hidden mass
    assert dec.mass_conserved()
    results["A1_missed_as_extra"] = dict(ship=ship, dec=_dec_json(dec))

    # --- Variant A2: reference WRONG (missed+extra pair), model predicts EXTRA ---
    #     This is the draft Section 4.3 worked example, verbatim.
    ref2 = [Event(1.0, 61, "missed"), Event(1.0, 60, "extra")]  # score C#5, played C5
    pred2 = [Event(1.0, 60, "extra")]
    ship2 = shipped_scores(pred2, ref2, tau=0.05)
    dec2 = decoupled_scores(pred2, ref2, tau=0.05)
    # shipped: extra track matches (played pitch), missed track FN -> event half-scored
    assert ship2["extra"]["tp"] == 1 and ship2["missed"]["fn"] == 1
    # decoupled: ref collapses to ONE wrong event; matched to pred extra -> off-diagonal
    assert dec2.n_localized == 1
    assert dec2.confusion == {("wrong", "extra"): 1}
    assert dec2.hm == 1.0
    assert dec2.mass_conserved()
    results["A2_wrong_as_extra"] = dict(ship=ship2, dec=_dec_json(dec2))

    if not quiet:
        print("=" * 74)
        print("EXPERIMENT A -- double-count witness")
        print("=" * 74)
        print("A1  ref={missed@1.0}  pred={extra@1.0}")
        print("    SHIPPED : missed-track FN=1 (F1=0.0)  +  extra-track FP=1 (F1=0.0)")
        print("              => ONE physical event penalized in TWO tracks.")
        print("    DECOUPLED: localized=1, confusion (ref=missed -> pred=extra)=1,")
        print("              HM=1.0, mass_conserved=%s => ONE event recovered."
              % dec.mass_conserved())
        print("A2  ref={wrong (missed C#5 + extra C5)@1.0}  pred={extra C5@1.0}  [draft 4.3]")
        print("    SHIPPED : extra-track TP=1 (looks perfect) + missed-track FN=1")
        print("              => substitution nature invisible.")
        print("    DECOUPLED: localized=1, confusion (ref=wrong -> pred=extra)=1, HM=1.0")
        print()
    return results


# --------------------------------------------------------------------------- #
# EXPERIMENT B -- the non-identifiability witness.                             #
# --------------------------------------------------------------------------- #

def experiment_B(quiet: bool = False) -> dict:
    """Two systems, IDENTICAL per-track F1, DIFFERENT HM (1/3 vs 0).

    Onsets are spaced 2.0 s apart (>> 500 ms) so localization is tau-invariant
    across the whole sweep and every count is exact. See README_oracle.md Table B."""

    def at(k):  # onset for slot k
        return 2.0 * k

    # ----- SYSTEM 1 (HIGH HM): all FP/FN come from localized MIS-TYPES -----
    ref1, pred1 = [], []
    slot = 1
    for _ in range(4):  # (missed -> missed)  diagonal
        ref1.append(Event(at(slot), 60, "missed")); pred1.append(Event(at(slot), 60, "missed")); slot += 1
    for _ in range(4):  # (extra -> extra)    diagonal
        ref1.append(Event(at(slot), 60, "extra")); pred1.append(Event(at(slot), 60, "extra")); slot += 1
    for _ in range(2):  # (missed -> extra)   OFF-diagonal
        ref1.append(Event(at(slot), 60, "missed")); pred1.append(Event(at(slot), 60, "extra")); slot += 1
    for _ in range(2):  # (extra -> missed)   OFF-diagonal
        ref1.append(Event(at(slot), 60, "extra")); pred1.append(Event(at(slot), 60, "missed")); slot += 1

    # ----- SYSTEM 2 (ZERO HM): identical marginals, all FP/FN are MARGINS -----
    ref2, pred2 = [], []
    slot = 1
    for _ in range(4):  # (missed -> missed)
        ref2.append(Event(at(slot), 60, "missed")); pred2.append(Event(at(slot), 60, "missed")); slot += 1
    for _ in range(4):  # (extra -> extra)
        ref2.append(Event(at(slot), 60, "extra")); pred2.append(Event(at(slot), 60, "extra")); slot += 1
    for _ in range(2):  # pure-miss reference missed (no prediction near) -> FN_missed
        ref2.append(Event(at(slot), 60, "missed")); slot += 1
    for _ in range(2):  # pure-miss reference extra  (no prediction near) -> FN_extra
        ref2.append(Event(at(slot), 60, "extra")); slot += 1
    for _ in range(2):  # spurious predicted missed (no reference near) -> FP_missed
        pred2.append(Event(at(slot), 60, "missed")); slot += 1
    for _ in range(2):  # spurious predicted extra   (no reference near) -> FP_extra
        pred2.append(Event(at(slot), 60, "extra")); slot += 1

    s1 = shipped_scores(pred1, ref1, tau=0.05)
    s2 = shipped_scores(pred2, ref2, tau=0.05)
    d1 = decoupled_scores(pred1, ref1, tau=0.05)
    d2 = decoupled_scores(pred2, ref2, tau=0.05)

    # hand-derived expectations (README_oracle.md Table B)
    for s in (s1, s2):
        assert abs(s["missed"]["f1"] - 2 / 3) < 1e-9
        assert abs(s["extra"]["f1"] - 2 / 3) < 1e-9
    # SHIPPED per-track F1 vectors are IDENTICAL:
    assert abs(s1["missed"]["f1"] - s2["missed"]["f1"]) < 1e-12
    assert abs(s1["extra"]["f1"] - s2["extra"]["f1"]) < 1e-12
    assert abs(s1["_mean_error_f1"] - s2["_mean_error_f1"]) < 1e-12
    # DECOUPLED HM SEPARATES them:
    assert abs(d1.hm - 1 / 3) < 1e-9
    assert d2.hm == 0.0
    assert d1.mass_conserved() and d2.mass_conserved()

    if not quiet:
        print("=" * 74)
        print("EXPERIMENT B -- non-identifiability witness")
        print("=" * 74)
        print("             per-track F1(missed) | per-track F1(extra) | mean | HM")
        print("  System 1:      %.4f        |     %.4f        | %.4f | %.4f"
              % (s1["missed"]["f1"], s1["extra"]["f1"], s1["_mean_error_f1"], d1.hm))
        print("  System 2:      %.4f        |     %.4f        | %.4f | %.4f"
              % (s2["missed"]["f1"], s2["extra"]["f1"], s2["_mean_error_f1"], d2.hm))
        print("  => SHIPPED per-track F1 IDENTICAL; decoupled HM = 1/3 vs 0.")
        print("     The shipped metric provably cannot separate these systems.")
        print()

    return dict(
        system1=dict(ship=s1, dec=_dec_json(d1)),
        system2=dict(ship=s2, dec=_dec_json(d2)),
    )


# --------------------------------------------------------------------------- #
# EXPERIMENT C -- recovery within tolerance + the killer-figure data.         #
# --------------------------------------------------------------------------- #

TAU_GRID = [0.050, 0.075, 0.100, 0.150, 0.200, 0.500]  # Section 4.4


def _build_injected_corpus(jitter: float):
    """Plant, per piece, 12 well-separated errors (4 missed, 4 extra, 4 wrong) and
    mis-type every 3rd non-wrong planted error, so the ground-truth localized-but-
    misclassified fraction HM* is set by construction. Returns
    (refs, preds, planted_total, mistyped_total)."""
    n_pieces, n_notes, ioi = 5, 40, 0.25
    sibling = {"missed": "extra", "extra": "missed"}
    refs, preds = [], []
    planted_total, mistyped_total = 0, 0
    for pc in range(n_pieces):
        piece = "pc%d" % pc
        clean = clean_corpus(n_notes, ioi=ioi, piece=piece)
        idxs = list(range(0, 36, 3))[:12]  # every 3rd note -> 750 ms apart, separated
        true_types = (["missed"] * 4) + (["extra"] * 4) + (["wrong"] * 4)
        plant = []
        for k, (i, tt) in enumerate(zip(idxs, true_types)):
            if k % 3 == 2 and tt != "wrong":
                ptype = sibling[tt]           # localized-but-misclassified
                mistyped_total += 1
            else:
                ptype = tt                    # correctly localized AND classified
            planted_total += 1
            plant.append((i, tt, ptype))
        r1, p1 = inject_and_label(clean, plant, onset_jitter=jitter, piece=piece)
        refs.extend(r1)
        preds.extend(p1)
    return refs, preds, planted_total, mistyped_total


def experiment_C(quiet: bool = False) -> dict:
    """Plant a KNOWN confusion kernel + onset jitter into a clean corpus; show the
    decoupled scorer recovers the planted HM* across the tau sweep while shipped
    per-track F1 is blind to HM*. Emits the killer-figure data."""

    # jitter 60 ms: at tau=50 ms not every event localizes; by tau>=100 ms all do.
    refsA, predsA, tot, mis = _build_injected_corpus(jitter=0.060)
    planted_hm = mis / tot  # ground-truth localized-but-misclassified fraction

    rows = []
    confusion_500 = {}
    for tau in TAU_GRID:
        dec = decoupled_scores(predsA, refsA, tau=tau)
        ship = shipped_scores(predsA, refsA, tau=tau)
        rows.append(dict(
            tau_ms=int(tau * 1000),
            localization_f1=round(dec.localization["f1"], 4),
            n_localized=dec.n_localized,
            hm=None if dec.hm is None else round(dec.hm, 4),
            shipped_mean_error_f1=round(ship["_mean_error_f1"], 4),
            mass_conserved=dec.mass_conserved(),
        ))
        if tau == TAU_GRID[-1]:
            confusion_500 = {f"{k[0]}->{k[1]}": v for k, v in dec.confusion.items()}

    # At sufficient tolerance (jitter < tau) HM(tau) -> planted HM*.
    hm_at_500 = rows[-1]["hm"]
    assert hm_at_500 is not None
    assert abs(hm_at_500 - planted_hm) < 0.05, (hm_at_500, planted_hm)

    if not quiet:
        print("=" * 74)
        print("EXPERIMENT C -- recovery within tolerance (figure data)")
        print("=" * 74)
        print("  planted localized-but-misclassified fraction HM* = %.4f" % planted_hm)
        print("  tau(ms)  loc_F1   n_loc   HM(tau)   shipped_mean_error_F1")
        for r in rows:
            print("   %4d    %.4f    %4d   %s      %.4f" % (
                r["tau_ms"], r["localization_f1"], r["n_localized"],
                "  n/a" if r["hm"] is None else "%.4f" % r["hm"],
                r["shipped_mean_error_f1"]))
        print("  => HM(500 ms)=%.4f recovers planted HM*=%.4f; shipped F1 never sees it."
              % (hm_at_500, planted_hm))
        print()

    return dict(planted_hm=planted_hm, sweep=rows, confusion_500ms=confusion_500)


# --------------------------------------------------------------------------- #
# PROPERTY TESTS (Section 4.2 tau-monotonicity; Section 4.3 mass conservation) #
# --------------------------------------------------------------------------- #

def property_tests(quiet: bool = False) -> None:
    import random
    rng = random.Random(20260718)
    for trial in range(200):
        n = rng.randint(0, 25)
        refs, preds = [], []
        for _ in range(n):
            on = round(rng.uniform(0, 10), 3)
            pit = rng.randint(48, 84)
            refs.append(Event(on, pit, rng.choice(SHIP_TRACKS)))
        for _ in range(rng.randint(0, 25)):
            on = round(rng.uniform(0, 10), 3)
            pit = rng.randint(48, 84)
            preds.append(Event(on, pit, rng.choice(SHIP_TRACKS)))
        prev_loc_tp = -1
        for tau in [0.05, 0.1, 0.2, 0.5, 1.0]:
            dec = decoupled_scores(preds, refs, tau=tau)
            # mass conservation invariant must hold at every tau
            assert dec.mass_conserved(), ("mass", trial, tau)
            # localization matched count is monotone non-decreasing in tau
            assert dec.localization["tp"] >= prev_loc_tp, ("monotone", trial, tau)
            prev_loc_tp = dec.localization["tp"]
    if not quiet:
        print("PROPERTY TESTS: 200 random corpora x 5 tolerances -- "
              "mass-conservation and tau-monotonicity PASS.\n")


# --------------------------------------------------------------------------- #
# Optional mir_eval cross-check of shipped mode.                              #
# --------------------------------------------------------------------------- #

def mir_eval_crosscheck(quiet: bool = False) -> Optional[bool]:
    try:
        import numpy as np
        import mir_eval
    except Exception:
        if not quiet:
            print("mir_eval cross-check SKIPPED (mir_eval/numpy not installed).\n")
        return None
    # one track, onset+pitch at 50 ms; compare our shipped F1 to mir_eval's.
    # mir_eval expects pitches in Hz and pitch_tolerance in cents; convert MIDI->Hz.
    def hz(m):
        return 440.0 * (2.0 ** ((m - 69) / 12.0))
    ref = [Event(0.0, 60, "extra"), Event(0.30, 62, "extra"), Event(0.61, 64, "extra")]
    pred = [Event(0.02, 60, "extra"), Event(0.30, 62, "extra"), Event(1.0, 67, "extra")]
    ours = shipped_scores(pred, ref, tau=0.05)["extra"]["f1"]
    ri = np.array([[e.onset, e.onset + 0.1] for e in ref])
    rp = np.array([hz(e.pitch) for e in ref], dtype=float)
    pi = np.array([[e.onset, e.onset + 0.1] for e in pred])
    pp = np.array([hz(e.pitch) for e in pred], dtype=float)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, rp, pi, pp, onset_tolerance=0.05, offset_ratio=None)
    ok = abs(ours - f) < 1e-9
    if not quiet:
        status = "MATCH" if ok else "DIFF (investigate matcher semantics)"
        print("mir_eval cross-check: ours=%.6f  mir_eval=%.6f  -> %s\n"
              % (ours, f, status))
    return ok


# --------------------------------------------------------------------------- #
# Output writers.                                                             #
# --------------------------------------------------------------------------- #

def _dec_json(d: DecoupledResult) -> dict:
    return dict(
        tau=d.tau,
        localization=d.localization,
        confusion={f"{k[0]}->{k[1]}": v for k, v in d.confusion.items()},
        false_alarm_on_correct=d.false_alarm_on_correct,
        spurious=d.spurious,
        miss_margin=d.miss_margin,
        n_localized=d.n_localized,
        hm=d.hm,
        mass_conserved=d.mass_conserved(),
    )


def write_outputs(here: str, A: dict, B: dict, C: dict) -> None:
    # figure LEFT panel: HM(tau) per system (Experiment C sweep + B constants)
    with open(os.path.join(here, "oracle_hm_sweep.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "tau_ms", "hm", "localization_f1",
                    "shipped_mean_error_f1"])
        for r in C["sweep"]:
            w.writerow(["injected_mistyper", r["tau_ms"], r["hm"],
                        r["localization_f1"], r["shipped_mean_error_f1"]])
        # System1/System2 are tau-flat (onsets >> 500 ms apart)
        for tau in [50, 75, 100, 150, 200, 500]:
            w.writerow(["witness_sys1_HM_high", tau,
                        B["system1"]["dec"]["hm"],
                        B["system1"]["dec"]["localization"]["f1"],
                        B["system1"]["ship"]["_mean_error_f1"]])
            w.writerow(["witness_sys2_HM_zero", tau,
                        B["system2"]["dec"]["hm"],
                        B["system2"]["dec"]["localization"]["f1"],
                        B["system2"]["ship"]["_mean_error_f1"]])

    # figure RIGHT panel: onset-matched confusion for the injected corpus at 500 ms.
    # Emit a full 3x3 grid (rows=ref error type, cols=pred error type) so the
    # off-diagonal (localized-but-misclassified) cells render as a heatmap.
    conf = C["confusion_500ms"]
    with open(os.path.join(here, "oracle_confusion.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ref_type", "pred_type", "count", "is_off_diagonal"])
        for rt in ERROR_TYPES:
            for pt in ERROR_TYPES:
                cnt = conf.get("%s->%s" % (rt, pt), 0)
                w.writerow([rt, pt, cnt, int(rt != pt)])

    with open(os.path.join(here, "oracle_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(dict(experiment_A=A, experiment_B=B, experiment_C=C),
                  fh, indent=2)


# --------------------------------------------------------------------------- #

def main() -> int:
    quiet = "--quiet" in sys.argv
    here = os.path.dirname(os.path.abspath(__file__))
    if not quiet:
        print("\nSYNTHETIC CONTROLLED-INJECTION ORACLE -- decoupled MER scorer\n")
    property_tests(quiet)
    mir_eval_crosscheck(quiet)
    A = experiment_A(quiet)
    B = experiment_B(quiet)
    C = experiment_C(quiet)
    write_outputs(here, A, B, C)
    if not quiet:
        print("=" * 74)
        print("ALL ASSERTIONS PASSED. Figure data written to:")
        print("  oracle_hm_sweep.csv   (figure LEFT panel: HM vs tolerance)")
        print("  oracle_confusion.csv  (figure RIGHT panel: onset-matched confusion)")
        print("  oracle_report.json    (all scored quantities)")
        print("=" * 74)
    else:
        print("OK: all oracle assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
