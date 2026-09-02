#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decoupled_scorer.py -- PRODUCTION decoupled localization/classification scorer
for Music Error Detection (MER).

Paper: "A Decoupled Localization-Classification Measure for Music Error Detection"
Target: IEEE Signal Processing Letters. Phase 2 artifact.

This module is the production harness built on the verified stdlib reference
implementation ``experiments/synthetic_oracle.py``. Every number the oracle
emits (``experiments/oracle_report.json``) is a regression anchor for this
module (see ``test_decoupled_scorer.py``).

DIFFERENCES FROM THE ORACLE (deliberate, spec-mandated)
-------------------------------------------------------
The oracle's matcher is a GREEDY ascending-distance pass, exact only on
well-separated onsets. This module replaces it with a provably correct
TWO-STAGE assignment (scipy ``linear_sum_assignment`` on a padded cost
matrix): the PRIMARY objective is maximum matching cardinality, the SECONDARY
objective is minimum total onset distance. Both objectives are EXACT (see
``_two_stage_assignment`` for the proof sketch); a documented, vanishingly
small tertiary perturbation keeps chord ties deterministic. Consequences:

* Loc-TP(tau) is the true maximum-cardinality one-to-one match count, so it is
  provably monotone non-decreasing in tau (the feasible edge set only grows).
* On dense/chordal data the greedy oracle pass can return FEWER matches than
  optimal; this module never does (regression-tested against a crafted case
  where greedy yields 1 match and the optimum is 2).
* Results are deterministic and independent of input event order: the scorer
  canonically sorts events before matching.

LOCKED METRIC KNOBS (proposal/LOCK-DECISIONS.md section 5)
----------------------------------------------------------
* Wrong-note collapse: PRIMARY strict co-location (any pitch pair; chords by
  max-cardinality min-onset-distance); SENSITIVITY pitch-aware (require
  score pitch != played pitch; chords by max-cardinality min-semitone-distance).
* Collapse radius eps: PRIMARY 50 ms, INDEPENDENT of the swept tau, applied
  symmetrically to BOTH sides BEFORE matching (constant event population
  across the sweep). SENSITIVITY eps = 0.
* Onset-tolerance grid: {50, 75, 100, 150, 200, 500} ms + mean-Loc-F summary.
* Localization matcher: max-cardinality bipartite, tie-break min total onset
  distance, deterministic.
* HM aggregation: corpus-pooled (micro) primary; per-piece bootstrap 95% CI
  diagnostic; HM undefined (None, never 0) when |M(tau)| = 0.
* Correct track: excluded from the localization match; false-alarm-on-correct
  resolved by a SECOND pass at the same tau into a mass-conserved margin.
* Backward-compatible SHIPPED mode: tau = 50 ms, per-track, coupled
  onset+pitch match, NO collapse; reproduces shipped per-track F1 to float
  tolerance (regression-tested against
  ``mir_eval.transcription.precision_recall_f1_overlap``).

REAL-DATA LOADER -- CONFIRMED SCHEMA
------------------------------------
The loader targets a directory of predicted
multi-track MIDI files (tracks named Extra / Missing-Removed / Correct) plus a
ground-truth metadata JSON mapping each piece id to reference MIDIs keyed
``extra_notes_midi`` / ``removed_notes_midi`` / ``correct_notes_midi``. The
"Missed" class is stored under ``removed_notes`` and is mapped
``removed_notes -> missed`` here; ``wrong`` is DERIVED by the collapse and is
never read from disk.  The on-disk schema (exact track names, metadata layout,
path conventions) was confirmed against the real Polytune/LadderSym artifacts
(see ``bridge_checks.py``, check C15). The loader is additionally unit-tested
on tiny generated MIDI files.

No side effects on import. CLI:

    python decoupled_scorer.py \
        --pred_dir <pred_dir> --gt_meta <metadata.json> \
        --tau 50 75 100 150 200 500 --eps 0.05 --collapse strict \
        --out results.json [--shipped] [--bootstrap N] [--seed S]

    python decoupled_scorer.py --self-check   # built-in regression vs oracle

tau / eps unit convention (matches the documented command line, which mixes
``--tau 50 ... 500`` with ``--eps 0.05``): a value > 1 is milliseconds, a
value <= 1 is seconds.

Dependencies: numpy, scipy (core scoring); pretty_midi (loader only, imported
lazily); mir_eval (optional cross-check only, imported lazily).
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

__version__ = "1.1.0"

__all__ = [
    "Event",
    "ERROR_TYPES",
    "SHIP_TRACKS",
    "TAU_GRID_S",
    "DEFAULT_EPSILON_S",
    "match_events",
    "collapse_wrong",
    "decoupled_scores",
    "shipped_scores",
    "score_consistency_filter",
    "sweep",
    "bootstrap_piece_ci",
    "load_pred_events",
    "load_ref_events",
    "DecoupledResult",
    "run_self_check",
    "main",
]

# First-class error labels AFTER the wrong-note collapse.
ERROR_TYPES = ("missed", "extra", "wrong")
# The three tracks shipped systems (Polytune / LadderSym / RUMAA) score.
SHIP_TRACKS = ("missed", "extra", "correct")
# Event types accepted on input. "wrong" is normally derived by the collapse,
# but pre-collapsed events are passed through unchanged (oracle semantics).
VALID_ETYPES = ("missed", "extra", "correct", "wrong")

# Locked tolerance grid (seconds) + locked collapse radius.
TAU_GRID_S = (0.050, 0.075, 0.100, 0.150, 0.200, 0.500)
DEFAULT_EPSILON_S = 0.050

# Float-safety slack on |d| <= radius comparisons, identical to the oracle's.
# The shipped mode instead uses slack 0 plus mir_eval's 4-decimal distance
# rounding (mir_eval.transcription.N_DECIMALS) for bit-exact compatibility.
FLOAT_SLACK = 1e-9


# --------------------------------------------------------------------------- #
# Event model (draft section 4.1).                                            #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Event:
    """A single MER event.

    Attributes:
        onset_s:    onset time in seconds.
        pitch_midi: MIDI pitch number (integer semitones).
        etype:      one of VALID_ETYPES. Named ``etype`` (not ``type``) to
                    avoid shadowing the builtin; the paper's "type".
        piece:      piece identifier; matching and collapse are within-piece
                    (draft section 4.7).
    """
    onset_s: float
    pitch_midi: int
    etype: str
    piece: str = "p0"


def _validate_events(events: Sequence[Event], what: str) -> None:
    for e in events:
        if e.etype not in VALID_ETYPES:
            raise ValueError(
                "%s event has unknown etype %r (valid: %s)"
                % (what, e.etype, ", ".join(VALID_ETYPES)))


def _canonical(events: Sequence[Event]) -> List[Event]:
    """Canonical total order; makes every downstream result independent of
    the caller's event ordering (determinism guarantee)."""
    return sorted(events, key=lambda e: (e.piece, e.onset_s, e.pitch_midi, e.etype))


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, recall, F1 with the 0/0 -> 0 convention (oracle-identical)."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# --------------------------------------------------------------------------- #
# Exact two-stage assignment core (draft section 4.2).                        #
# --------------------------------------------------------------------------- #

def _two_stage_assignment(cost: np.ndarray, feasible: np.ndarray) -> List[Tuple[int, int]]:
    """Exact (max-cardinality, then min-total-cost) one-to-one matching.

    Args:
        cost:     (na, nb) nonnegative costs for FEASIBLE pairs.
        feasible: (na, nb) boolean mask of admissible pairs.

    Returns: list of (row, col) matched feasible pairs.

    Correctness. Let C = sum(cost[feasible]) + 1, and score every infeasible
    cell C. A full rectangular assignment of size m = min(na, nb) that uses f
    feasible pairs costs (m - f) * C + S, where S <= sum(cost[feasible]) < C.
    Hence total cost is strictly decreasing in f before S can matter:
    minimizing it FIRST maximizes the number of feasible pairs (true maximum
    cardinality, since any maximum matching extends to a full assignment) and
    THEN minimizes the total feasible cost S. Both objectives are exact; no
    tolerance is involved. scipy's Hungarian solver is deterministic, so ties
    among optimal matchings resolve to a fixed canonical choice.
    """
    if not feasible.any():
        return []
    big = float(cost[feasible].sum()) + 1.0
    padded = np.where(feasible, cost, big)
    rows, cols = linear_sum_assignment(padded)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if feasible[i, j]]


def _split_components(a_onsets: np.ndarray, b_onsets: np.ndarray,
                      radius: float, slack: float
                      ) -> List[Tuple[List[int], List[int]]]:
    """Split the bipartite onset-proximity graph into connected components.

    Sort the union of onsets; a gap > radius + slack between consecutive
    union onsets separates components (no feasible edge can cross it, since
    edge feasibility requires |a - b| <= radius + slack). Keeps assignment
    matrices small on long pieces without changing the optimum.
    """
    tagged = ([(float(o), 0, i) for i, o in enumerate(a_onsets)]
              + [(float(o), 1, i) for i, o in enumerate(b_onsets)])
    tagged.sort()
    comps: List[Tuple[List[int], List[int]]] = []
    cur_a: List[int] = []
    cur_b: List[int] = []
    prev = None
    for onset, side, idx in tagged:
        if prev is not None and onset - prev > radius + slack:
            if cur_a and cur_b:
                comps.append((cur_a, cur_b))
            cur_a, cur_b = [], []
        (cur_a if side == 0 else cur_b).append(idx)
        prev = onset
    if cur_a and cur_b:
        comps.append((cur_a, cur_b))
    return comps


def match_events(preds: Sequence[Event], refs: Sequence[Event], tau: float,
                 require_pitch: bool = False,
                 slack: float = FLOAT_SLACK,
                 round_decimals: Optional[int] = None) -> List[Tuple[int, int]]:
    """TRUE maximum-cardinality one-to-one onset matching within tau.

    Objective hierarchy (exact for 1 and 2; see ``_two_stage_assignment``):
      1. maximize the number of matched pairs (max cardinality);
      2. minimize the total onset distance of the matched pairs;
      3. deterministic chord tie-break by pitch proximity, implemented as a
         perturbation of total magnitude < 1e-9 s -- it can only decide
         between matchings whose total onset distances differ by less than
         that (physically indistinguishable onsets).

    Matching is within-piece. ``require_pitch`` adds the coupled onset+pitch
    constraint used by the shipped mode (feasible only if pitches are equal).
    ``slack`` is the float-safety slack on the tau boundary (0 in shipped
    mode for bit-exact mir_eval semantics). ``round_decimals`` rounds onset
    DISTANCES before the tau comparison, reproducing
    ``mir_eval.transcription`` boundary semantics (its ``N_DECIMALS = 4``);
    None disables rounding (decoupled/oracle semantics).

    Returns (pred_index, ref_index) pairs into the given sequences.
    Cardinality is monotone non-decreasing in tau: the feasible edge set only
    grows with tau, and stage 1 is an exact maximum over that set.
    """
    # rounding can admit pairs with true distance up to tau + 0.5*10^-k, so
    # the component split threshold must cover that extra reach.
    split_slack = slack
    if round_decimals is not None:
        split_slack = max(split_slack, slack + 0.51 * 10.0 ** (-round_decimals))
    by_piece_p: Dict[str, List[int]] = defaultdict(list)
    by_piece_r: Dict[str, List[int]] = defaultdict(list)
    for i, e in enumerate(preds):
        by_piece_p[e.piece].append(i)
    for j, e in enumerate(refs):
        by_piece_r[e.piece].append(j)

    pairs: List[Tuple[int, int]] = []
    for piece in sorted(set(by_piece_p) & set(by_piece_r)):
        pi = by_piece_p[piece]
        ri = by_piece_r[piece]
        p_on = np.array([preds[i].onset_s for i in pi], dtype=float)
        r_on = np.array([refs[j].onset_s for j in ri], dtype=float)
        p_pt = np.array([preds[i].pitch_midi for i in pi], dtype=float)
        r_pt = np.array([refs[j].pitch_midi for j in ri], dtype=float)
        for ca, cb in _split_components(p_on, r_on, tau, split_slack):
            d = np.abs(p_on[ca][:, None] - r_on[cb][None, :])
            pd = np.abs(p_pt[ca][:, None] - r_pt[cb][None, :])
            d_cmp = d if round_decimals is None else np.around(d, round_decimals)
            feas = d_cmp <= tau + slack
            if require_pitch:
                feas &= (pd == 0)
            # tertiary determinism perturbation (see docstring, item 3)
            tie_w = 1e-9 / (127.0 * (min(len(ca), len(cb)) + 1.0))
            cost = d + tie_w * pd
            for a, b in _two_stage_assignment(cost, feas):
                pairs.append((pi[ca[a]], ri[cb[b]]))
    return sorted(pairs)


# --------------------------------------------------------------------------- #
# Symmetric PRE-match wrong-note collapse (draft section 4.3).                #
# --------------------------------------------------------------------------- #

def collapse_wrong(events: Sequence[Event],
                   epsilon: float = DEFAULT_EPSILON_S,
                   mode: str = "strict") -> List[Event]:
    """Collapse co-located (missed, extra) pairs within ``epsilon`` into
    single ``wrong`` events. Applied to ONE side; the scorer applies it to
    both sides symmetrically BEFORE matching, at a radius INDEPENDENT of tau.

    Locked pairing rules (LOCK-DECISIONS.md section 5):
      * mode="strict" (PRIMARY): any pitch pair; chords resolved by
        max-cardinality, then min total onset distance (exact), then a
        deterministic pitch-proximity perturbation.
      * mode="pitch_aware" (SENSITIVITY): additionally require
        score (missed) pitch != played (extra) pitch; chords resolved by
        max-cardinality, then min total SEMITONE distance (exact), then a
        deterministic onset-proximity tie-break (exact when semitone totals
        tie, since semitone totals are integers and the onset term's total
        contribution is < 1/2).

    A collapsed ``wrong`` event carries the played (extra) onset and pitch,
    exactly as in the oracle. Non-(missed|extra) events pass through.
    """
    if mode not in ("strict", "pitch_aware", "none"):
        raise ValueError("collapse mode must be 'strict', 'pitch_aware', or 'none', got %r" % mode)
    if mode == "none":
        # Collapse-free two-class scoring: the functional Prop. 1 bounds.
        return list(events)
    missed = [e for e in events if e.etype == "missed"]
    extra = [e for e in events if e.etype == "extra"]
    passthrough = [e for e in events if e.etype not in ("missed", "extra")]

    by_piece_m: Dict[str, List[int]] = defaultdict(list)
    by_piece_x: Dict[str, List[int]] = defaultdict(list)
    for i, e in enumerate(missed):
        by_piece_m[e.piece].append(i)
    for j, e in enumerate(extra):
        by_piece_x[e.piece].append(j)

    used_m: set = set()
    used_x: set = set()
    wrong: List[Event] = []
    for piece in sorted(set(by_piece_m) & set(by_piece_x)):
        mi = by_piece_m[piece]
        xi = by_piece_x[piece]
        m_on = np.array([missed[i].onset_s for i in mi], dtype=float)
        x_on = np.array([extra[j].onset_s for j in xi], dtype=float)
        m_pt = np.array([missed[i].pitch_midi for i in mi], dtype=float)
        x_pt = np.array([extra[j].pitch_midi for j in xi], dtype=float)
        for ca, cb in _split_components(m_on, x_on, epsilon, FLOAT_SLACK):
            d = np.abs(m_on[ca][:, None] - x_on[cb][None, :])
            pd = np.abs(m_pt[ca][:, None] - x_pt[cb][None, :])
            feas = d <= epsilon + FLOAT_SLACK
            n_min = min(len(ca), len(cb))
            if mode == "pitch_aware":
                feas &= (pd > 0)
                # primary(cardinality) -> secondary(total semitones, exact:
                # integer totals) -> tertiary(total onset dist, contribution
                # bounded < 1/2 so it cannot flip an integer comparison).
                w_on = 0.5 / ((epsilon + FLOAT_SLACK) * (n_min + 1.0) + 1e-12)
                cost = pd + w_on * d
            else:
                tie_w = 1e-9 / (127.0 * (n_min + 1.0))
                cost = d + tie_w * pd
            for a, b in _two_stage_assignment(cost, feas):
                gm, gx = mi[ca[a]], xi[cb[b]]
                used_m.add(gm)
                used_x.add(gx)
                x = extra[gx]
                wrong.append(Event(x.onset_s, x.pitch_midi, "wrong", x.piece))
    leftover_m = [e for i, e in enumerate(missed) if i not in used_m]
    leftover_x = [e for j, e in enumerate(extra) if j not in used_x]
    return passthrough + leftover_m + leftover_x + wrong


# --------------------------------------------------------------------------- #
# Decoupled scorer (draft sections 4.2, 4.3, 4.5).                            #
# --------------------------------------------------------------------------- #

@dataclass
class DecoupledResult:
    """All scored quantities at one tau. Mass-conservation invariant:
    every post-collapse predicted error is in exactly one confusion cell or
    one of {false_alarm_on_correct, spurious}; every post-collapse reference
    error is in exactly one confusion cell or miss_margin."""
    tau: float
    epsilon: float
    collapse: str
    localization: Dict[str, float]            # tp/fp/fn/precision/recall/f1
    confusion: Dict[str, Dict[str, int]]      # full [ref_type][pred_type] grid
    false_alarm_on_correct: int
    spurious: int
    miss_margin: int
    n_localized: int                          # |M(tau)|
    hm: Optional[float]                       # None (never 0) when |M(tau)| = 0
    n_pred_err: int
    n_ref_err: int

    def off_diagonal(self) -> int:
        return sum(self.confusion[rt][pt]
                   for rt in ERROR_TYPES for pt in ERROR_TYPES if rt != pt)

    def confusion_total(self) -> int:
        return sum(self.confusion[rt][pt]
                   for rt in ERROR_TYPES for pt in ERROR_TYPES)

    def mass_conserved(self) -> bool:
        """Every collapsed error event is accounted for exactly once.

        v1.1.0: this is now a real check. Through v1.0.0 `spurious` was
        computed as `len(unmatched_pred) - fa_on_correct` and `miss_margin`
        as `fn`, so both identities below reduced to `tp + (n - tp) == n`
        and held for ANY input — the guard could never fire (verifier
        finding R1, 2026-07-20). The three prediction-side classes are now
        counted from independent sources (matcher output, second-pass FA
        matcher, direct scan of the residue), so a disagreement between the
        two matchers — double-counting a prediction as both matched and a
        false alarm, or dropping one — now trips this.
        """
        conf_total = self.confusion_total()
        pred_ok = (conf_total + self.false_alarm_on_correct + self.spurious
                   == self.n_pred_err)
        ref_ok = conf_total + self.miss_margin == self.n_ref_err
        return pred_ok and ref_ok

    def confusion_sparse(self) -> Dict[str, int]:
        """Oracle-report-compatible sparse form ('ref->pred': count, nonzero)."""
        return {"%s->%s" % (rt, pt): self.confusion[rt][pt]
                for rt in ERROR_TYPES for pt in ERROR_TYPES
                if self.confusion[rt][pt]}

    def to_jsonable(self) -> dict:
        return dict(
            tau_s=self.tau,
            tau_ms=int(round(self.tau * 1000)),
            epsilon_s=self.epsilon,
            collapse=self.collapse,
            localization=dict(self.localization),
            confusion=self.confusion,
            confusion_sparse=self.confusion_sparse(),
            false_alarm_on_correct=self.false_alarm_on_correct,
            spurious=self.spurious,
            miss_margin=self.miss_margin,
            n_localized=self.n_localized,
            hm=self.hm,
            n_pred_err=self.n_pred_err,
            n_ref_err=self.n_ref_err,
            mass_conserved=self.mass_conserved(),
        )


def decoupled_scores(pred: Sequence[Event], ref: Sequence[Event], tau: float,
                     epsilon: float = DEFAULT_EPSILON_S,
                     collapse: str = "strict") -> DecoupledResult:
    """Decoupled localization + classification scoring at one tau.

    Pipeline (oracle-identical semantics, exact matcher):
      1. symmetric PRE-match wrong-note collapse on both sides at ``epsilon``
         (independent of tau -> constant event population across the sweep);
      2. localization: exact max-cardinality one-to-one onset match on the
         pooled error union, type-ignored;
      3. rectangular confusion C[ref_type][pred_type] over {extra,missed,wrong}
         on matched events; ``correct`` is a reference-side-only class;
      4. margins: unmatched predicted errors get a SECOND pass at the same tau
         against reference CORRECT notes (false-alarm-on-correct) else are
         spurious; unmatched reference errors are the miss margin;
      5. HM(tau) = off-diagonal / |M(tau)|; None (never 0) when |M(tau)| = 0.

    Raises RuntimeError if the mass-conservation invariant is violated
    (cannot happen by construction; enforced as a production guard).
    """
    _validate_events(pred, "predicted")
    _validate_events(ref, "reference")
    predc = _canonical(collapse_wrong(pred, epsilon=epsilon, mode=collapse))
    refc = _canonical(collapse_wrong(ref, epsilon=epsilon, mode=collapse))

    pred_err = [e for e in predc if e.etype in ERROR_TYPES]
    ref_err = [e for e in refc if e.etype in ERROR_TYPES]
    ref_correct = [e for e in refc if e.etype == "correct"]

    match = match_events(pred_err, ref_err, tau, require_pitch=False)
    matched_p = {pi for pi, _ in match}
    tp = len(match)
    fp = len(pred_err) - tp
    fn = len(ref_err) - tp
    lp, lr, lf = prf(tp, fp, fn)

    confusion: Dict[str, Dict[str, int]] = {
        rt: {pt: 0 for pt in ERROR_TYPES} for rt in ERROR_TYPES}
    for pi, ri in match:
        confusion[ref_err[ri].etype][pred_err[pi].etype] += 1

    unmatched_pred = [pred_err[i] for i in range(len(pred_err)) if i not in matched_p]
    fa_on_correct = 0
    fa_pred_idx: set = set()
    if ref_correct and unmatched_pred:
        fa_match = match_events(unmatched_pred, ref_correct, tau,
                                require_pitch=False)
        fa_on_correct = len(fa_match)
        fa_pred_idx = {pi for pi, _ in fa_match}
    # Count spurious DIRECTLY (v1.1.0) rather than as
    # len(unmatched_pred) - fa_on_correct. The subtraction made the
    # mass-conservation check below a tautology (verifier finding R1,
    # 2026-07-20): every term was derived from the same two quantities, so
    # the identity held for any input and the guard was dead code. Counting
    # the three partition classes from independent sources — matcher output,
    # second-pass FA matcher, and a direct scan of the residue — makes the
    # invariant a genuine test of the accounting.
    spurious = sum(1 for i in range(len(unmatched_pred)) if i not in fa_pred_idx)

    off_diag = sum(confusion[rt][pt]
                   for rt in ERROR_TYPES for pt in ERROR_TYPES if rt != pt)
    hm = (off_diag / tp) if tp > 0 else None

    result = DecoupledResult(
        tau=tau, epsilon=epsilon, collapse=collapse,
        localization=dict(tp=tp, fp=fp, fn=fn, precision=lp, recall=lr, f1=lf),
        confusion=confusion,
        false_alarm_on_correct=fa_on_correct,
        spurious=spurious,
        miss_margin=fn,
        n_localized=tp,
        hm=hm,
        n_pred_err=len(pred_err),
        n_ref_err=len(ref_err),
    )
    if not result.mass_conserved():
        raise RuntimeError(
            "mass-conservation invariant violated at tau=%g "
            "(pred_err=%d, ref_err=%d, confusion=%d, fa=%d, spurious=%d, miss=%d)"
            % (tau, result.n_pred_err, result.n_ref_err, result.confusion_total(),
               fa_on_correct, spurious, fn))
    return result


# --------------------------------------------------------------------------- #
# Backward-compatible SHIPPED mode (draft section 4.6).                       #
# --------------------------------------------------------------------------- #

def shipped_scores(pred: Sequence[Event], ref: Sequence[Event],
                   tau: float = 0.050) -> Dict[str, Dict[str, float]]:
    """Per-track P/R/F1 exactly as the shipped per-track loop computes them:
    for each track in {missed, extra, correct}, an INDEPENDENT coupled
    onset+pitch match at tau (default 50 ms), NO collapse.

    SCOPE OF THE BIT-EXACTNESS CLAIM (corrected 2026-07-20). The guarantee
    below applies to SHIPPED MODE ONLY. Decoupled mode matches with
    ``round_decimals=None`` and slack 1e-9, whereas mir_eval rounds onset
    distances to 4 decimals (``N_DECIMALS = 4``); the two therefore disagree
    for onset gaps with ``|delta| in (tau + 1e-9, tau + 5e-5]``. That window is
    empirically vacuous on MAESTRO-E but is a real semantic difference and must
    not be described as bit-exact. See proposal/ANALYTIC-BRIDGE.md (R6).

    Boundary semantics are bit-exact with
    ``mir_eval.transcription.precision_recall_f1_overlap(...,
    onset_tolerance=tau, offset_ratio=None)`` for integer-MIDI pitches:
    slack 0 and onset distances rounded to 4 decimals before the tau
    comparison (mir_eval's ``N_DECIMALS = 4``). The TP count of any
    maximum-cardinality matching on the same feasibility graph is unique, so
    P/R/F agree to float tolerance even though the matched pairs may differ.
    """
    _validate_events(pred, "predicted")
    _validate_events(ref, "reference")
    out: Dict[str, Dict[str, float]] = {}
    for track in SHIP_TRACKS:
        pT = _canonical([e for e in pred if e.etype == track])
        rT = _canonical([e for e in ref if e.etype == track])
        m = match_events(pT, rT, tau, require_pitch=True, slack=0.0,
                         round_decimals=4)
        tp = len(m)
        fp = len(pT) - tp
        fn = len(rT) - tp
        p, r, f = prf(tp, fp, fn)
        out[track] = dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f)
    out["_mean_f1"] = sum(out[t]["f1"] for t in SHIP_TRACKS) / len(SHIP_TRACKS)
    # Polytune "Error Detection F1" family excludes the correct track.
    out["_mean_error_f1"] = (out["missed"]["f1"] + out["extra"]["f1"]) / 2.0
    return out


def score_consistency_filter(pred: Sequence[Event], ref: Sequence[Event],
                             anchor: float = 0.050) -> Tuple[List[Event], int]:
    """Drop every predicted ``missed`` claim that names no score note.

    A deletion claim (pitch p at onset t) is kept only if the score holds a
    note of pitch p with an onset within ``anchor`` of t, in the same piece.
    The score is the reference ``correct`` plus ``missed`` (removed) tracks:
    every note the performer was meant to play, whether played or omitted.
    No reference *error* label is consulted, so a score-informed system can
    apply the same rule at inference time. Every other event type passes
    through unchanged. Returns (kept events, number dropped). The anchor
    semantics (closed window, no slack) mirror the unfounded-claim
    adjudication in setup/collapse_validation_ei.py.
    """
    _validate_events(pred, "predicted")
    _validate_events(ref, "reference")
    score: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for e in ref:
        if e.etype in ("correct", "missed"):
            score[(e.piece, e.pitch_midi)].append(e.onset_s)
    for onsets in score.values():
        onsets.sort()
    kept: List[Event] = []
    dropped = 0
    for e in pred:
        if e.etype != "missed":
            kept.append(e)
            continue
        onsets = score.get((e.piece, e.pitch_midi))
        i = bisect.bisect_left(onsets, e.onset_s - anchor) if onsets else 0
        if onsets and i < len(onsets) and onsets[i] <= e.onset_s + anchor:
            kept.append(e)
        else:
            dropped += 1
    return kept, dropped


def mir_eval_track_prf(pred: Sequence[Event], ref: Sequence[Event],
                       tau: float = 0.050) -> Tuple[float, float, float]:
    """Direct mir_eval cross-check for ONE track's event lists (all events are
    scored regardless of etype). Converts MIDI pitch to Hz and uses
    ``precision_recall_f1_overlap(onset_tolerance=tau, offset_ratio=None)``
    per piece, aggregating matched counts across pieces (mir_eval has no
    piece concept). Lazy import: mir_eval is optional at runtime."""
    import mir_eval  # lazy

    def hz(m: float) -> float:
        return 440.0 * (2.0 ** ((m - 69) / 12.0))

    pieces = sorted({e.piece for e in pred} | {e.piece for e in ref})
    tp = fp = fn = 0
    for pc in pieces:
        p = [e for e in pred if e.piece == pc]
        r = [e for e in ref if e.piece == pc]
        if not p or not r:
            fp += len(p)
            fn += len(r)
            continue
        ri = np.array([[e.onset_s, e.onset_s + 0.1] for e in r])
        rp = np.array([hz(e.pitch_midi) for e in r], dtype=float)
        pi = np.array([[e.onset_s, e.onset_s + 0.1] for e in p])
        pp = np.array([hz(e.pitch_midi) for e in p], dtype=float)
        matching = mir_eval.transcription.match_notes(
            ri, rp, pi, pp, onset_tolerance=tau, offset_ratio=None)
        tp += len(matching)
        fp += len(p) - len(matching)
        fn += len(r) - len(matching)
    return prf(tp, fp, fn)


# --------------------------------------------------------------------------- #
# Tolerance sweep + summary (locked grid).                                    #
# --------------------------------------------------------------------------- #

def sweep(pred: Sequence[Event], ref: Sequence[Event],
          taus: Sequence[float] = TAU_GRID_S,
          epsilon: float = DEFAULT_EPSILON_S,
          collapse: str = "strict") -> Dict[str, object]:
    """Decoupled scores over the tolerance grid + mean-Loc-F summary.

    Because the collapse radius is independent of tau, the event population
    (n_pred_err, n_ref_err) is constant across the sweep, and because the
    matcher is an exact maximum, Loc-TP(tau) is monotone non-decreasing."""
    results = [decoupled_scores(pred, ref, tau=t, epsilon=epsilon,
                                collapse=collapse) for t in taus]
    # NOTE (v1.1.0): this averages over the TAU GRID, not over pieces. It was
    # emitted as `mean_loc_f1`, which reads as a macro-per-piece mean and
    # invited exactly that misreading next to Loc-F@50 (verifier finding R2,
    # 2026-07-20). Renamed to make the averaging axis explicit. The grid is
    # non-uniform and includes tau=500 ms, a tolerance well outside anything
    # defensible for MER, so this is a sweep-shape summary only — do not
    # report it as a headline localization number.
    mean_loc_f1_over_tau_grid = (
        sum(r.localization["f1"] for r in results) / len(results)
        if results else 0.0)
    return dict(taus_s=list(taus), results=results,
                mean_loc_f1_over_tau_grid=mean_loc_f1_over_tau_grid)


# --------------------------------------------------------------------------- #
# Per-piece bootstrap 95% CI (LOCK section 5: diagnostic aggregation).        #
# --------------------------------------------------------------------------- #

def bootstrap_piece_ci(pred: Sequence[Event], ref: Sequence[Event], tau: float,
                       epsilon: float = DEFAULT_EPSILON_S,
                       collapse: str = "strict",
                       n_boot: int = 2000,
                       seed: int = 20260718,
                       alpha: float = 0.05) -> Dict[str, object]:
    """Per-piece nonparametric bootstrap CI for HM(tau) and Loc-F(tau).

    Pieces are the resampling unit (draft section 4.7). Because matching and
    collapse are within-piece, corpus-pooled (micro) counts equal the sum of
    per-piece counts; this is asserted as a safety check. Replicates with
    |M(tau)| = 0 leave HM undefined and are EXCLUDED from the HM percentile
    interval (their count is reported as ``n_hm_undefined``). The point
    estimate is the corpus-pooled micro value (primary aggregation); the
    bootstrap distribution is the per-piece diagnostic.
    """
    pieces = sorted({e.piece for e in pred} | {e.piece for e in ref})
    stats = []  # per piece: (off_diag, n_localized, tp, fp, fn)
    for pc in pieces:
        r = decoupled_scores([e for e in pred if e.piece == pc],
                             [e for e in ref if e.piece == pc],
                             tau=tau, epsilon=epsilon, collapse=collapse)
        stats.append((r.off_diagonal(), r.n_localized,
                      r.localization["tp"], r.localization["fp"],
                      r.localization["fn"]))
    arr = np.array(stats, dtype=float) if stats else np.zeros((0, 5))

    pooled = decoupled_scores(pred, ref, tau=tau, epsilon=epsilon,
                              collapse=collapse)
    # within-piece decomposition safety check (RuntimeError, not assert: must
    # survive `python -O`, matching the guard discipline in decoupled_scores)
    if int(arr[:, 2].sum()) != pooled.localization["tp"]:
        raise RuntimeError(
            "per-piece TP does not sum to pooled TP (within-piece invariant broken)")
    point_hm = pooled.hm
    point_loc_f1 = pooled.localization["f1"]

    rng = np.random.default_rng(seed)
    hm_reps: List[float] = []
    loc_reps: List[float] = []
    n_undef = 0
    n_pieces = len(pieces)
    for _ in range(n_boot):
        idx = rng.integers(0, n_pieces, size=n_pieces) if n_pieces else []
        s = arr[idx].sum(axis=0) if n_pieces else np.zeros(5)
        od, nloc, tp, fp, fn = s
        if nloc > 0:
            hm_reps.append(od / nloc)
        else:
            n_undef += 1
        loc_reps.append(prf(int(tp), int(fp), int(fn))[2])

    def pct_ci(reps: List[float]) -> Optional[Tuple[float, float]]:
        if not reps:
            return None
        lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (float(lo), float(hi))

    return dict(
        tau_s=tau,
        n_pieces=n_pieces,
        n_boot=n_boot,
        seed=seed,
        point_hm=point_hm,
        point_loc_f1=point_loc_f1,
        hm_ci95=pct_ci(hm_reps),
        loc_f1_ci95=pct_ci(loc_reps),
        n_hm_undefined=n_undef,
    )


# --------------------------------------------------------------------------- #
# Real-data loader (schema per setup/03_inference_commands.md).               #
# PHASE-3 FLAG: reconfirm the on-disk schema against real artifacts before   #
# the pilot; unit-tested on tiny generated MIDI only.                         #
# --------------------------------------------------------------------------- #

# Predicted multi-track MIDI: instrument-name substring -> event type.
# Order matters: first match wins. "Missing"/"Removed" both map to missed.
TRACK_NAME_RULES: Tuple[Tuple[str, str], ...] = (
    ("extra", "extra"),
    ("remov", "missed"),   # removed_notes -> missed (critical loader note)
    ("miss", "missed"),    # matches both "Missing" and "Missed"
    ("correct", "correct"),
)

# Ground-truth metadata keys -> event type. "wrong" is DERIVED, never read.
GT_KEY_TO_ETYPE: Mapping[str, str] = {
    "extra_notes_midi": "extra",
    "removed_notes_midi": "missed",   # removed_notes -> missed
    "correct_notes_midi": "correct",
}


def classify_track_name(name: str) -> Optional[str]:
    """Map a predicted-MIDI instrument/track name to an event type, or None."""
    low = (name or "").lower()
    for sub, etype in TRACK_NAME_RULES:
        if sub in low:
            return etype
    return None


def _midi_notes(path: str):
    """Yield (start_s, pitch_midi, instrument_name) from a MIDI file."""
    import pretty_midi  # lazy: loader-only dependency
    pm = pretty_midi.PrettyMIDI(path)
    for inst in pm.instruments:
        for note in inst.notes:
            yield float(note.start), int(note.pitch), inst.name


def load_pred_events(pred_dir: str) -> List[Event]:
    """Load predicted events from a directory of multi-track MIDI files.

    One file per piece; piece id = file stem. Each MIDI track's name selects
    the event type via TRACK_NAME_RULES (Extra / Missing-Removed / Correct).
    Raises ValueError on a track name that matches no rule -- a loud schema
    guard, since the real Polytune/LadderSym track names must be reconfirmed
    in Phase 3.
    """
    names = sorted(n for n in os.listdir(pred_dir)
                   if n.lower().endswith((".mid", ".midi")))
    if not names:
        raise FileNotFoundError("no .mid/.midi files found in %r" % pred_dir)
    events: List[Event] = []
    for name in names:
        piece = os.path.splitext(name)[0]
        path = os.path.join(pred_dir, name)
        for start, pitch, tname in _midi_notes(path):
            etype = classify_track_name(tname)
            if etype is None:
                raise ValueError(
                    "unrecognized track name %r in %s -- update "
                    "TRACK_NAME_RULES after reconfirming the Phase-3 schema"
                    % (tname, path))
            events.append(Event(start, pitch, etype, piece))
    return events


def load_ref_events(gt_meta_path: str) -> List[Event]:
    """Load reference events from the ground-truth metadata JSON.

    Schema (per setup/03_inference_commands.md; reconfirm in Phase 3):
        { "<piece_id>": { "extra_notes_midi":   "<path.mid>",
                          "removed_notes_midi": "<path.mid>",
                          "correct_notes_midi": "<path.mid>" }, ... }

    Paths are resolved relative to the metadata file's directory when not
    absolute. ``removed_notes_midi`` maps to etype "missed". ``wrong`` is
    derived later by the collapse and is never read from disk. A piece may
    omit a key (no events of that type). Every note in every track of the
    referenced MIDI receives the key's event type.
    """
    with open(gt_meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if not isinstance(meta, dict):
        raise ValueError(
            "gt_meta must be a JSON object mapping piece id -> track paths "
            "(got %s) -- reconfirm the Phase-3 schema" % type(meta).__name__)
    base = os.path.dirname(os.path.abspath(gt_meta_path))
    events: List[Event] = []
    for piece_id in sorted(meta):
        entry = meta[piece_id]
        if not isinstance(entry, dict):
            raise ValueError("gt_meta entry for %r is not an object -- "
                             "reconfirm the Phase-3 schema" % piece_id)
        for key, etype in GT_KEY_TO_ETYPE.items():
            if key not in entry or entry[key] is None:
                continue
            path = entry[key]
            if not os.path.isabs(path):
                path = os.path.join(base, path)
            for start, pitch, _tname in _midi_notes(path):
                events.append(Event(start, pitch, etype, piece_id))
    return events


# --------------------------------------------------------------------------- #
# Built-in self-check: regression vs the oracle's hand-derived numbers.       #
# --------------------------------------------------------------------------- #

def _greedy_reference_match(preds: Sequence[Event], refs: Sequence[Event],
                            tau: float) -> List[Tuple[int, int]]:
    """The oracle's greedy ascending-distance pass, reproduced verbatim for
    the self-check demonstration ONLY (exact just on well-separated onsets)."""
    cands = []
    for pi, p in enumerate(preds):
        for ri, r in enumerate(refs):
            if p.piece != r.piece:
                continue
            d = abs(p.onset_s - r.onset_s)
            if d <= tau + FLOAT_SLACK:
                cands.append((d, abs(p.pitch_midi - r.pitch_midi), pi, ri))
    cands.sort()
    used_p, used_r, out = set(), set(), []
    for _d, _pd, pi, ri in cands:
        if pi in used_p or ri in used_r:
            continue
        used_p.add(pi)
        used_r.add(ri)
        out.append((pi, ri))
    return out


def run_self_check(verbose: bool = True) -> int:
    """Machine-checked regression of the paper's core claims. Returns 0 on
    success; raises AssertionError on any failure."""
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    # --- A1: double-count witness -----------------------------------------
    ref = [Event(1.0, 61, "missed")]
    pred = [Event(1.0, 61, "extra")]
    ship = shipped_scores(pred, ref, tau=0.05)
    dec = decoupled_scores(pred, ref, tau=0.05)
    assert ship["missed"]["fn"] == 1 and ship["extra"]["fp"] == 1
    assert ship["missed"]["f1"] == 0.0 and ship["extra"]["f1"] == 0.0
    assert dec.n_localized == 1 and dec.confusion["missed"]["extra"] == 1
    assert dec.hm == 1.0 and dec.mass_conserved()
    log("A1 double-count witness: shipped 1 FP + 1 FN in two tracks; "
        "decoupled = ONE localized-but-misclassified event. OK")

    # --- A2: wrong-as-extra worked example --------------------------------
    ref2 = [Event(1.0, 61, "missed"), Event(1.0, 60, "extra")]
    pred2 = [Event(1.0, 60, "extra")]
    dec2 = decoupled_scores(pred2, ref2, tau=0.05)
    assert dec2.n_localized == 1 and dec2.confusion["wrong"]["extra"] == 1
    assert dec2.hm == 1.0 and dec2.mass_conserved()
    log("A2 wrong-as-extra worked example: ref collapses to one wrong; "
        "confusion (wrong->extra)=1, HM=1.0. OK")

    # --- B: non-identifiability witness (HM = 1/3 vs 0) -------------------
    def slot(k: int) -> float:
        return 2.0 * k

    ref1, pred1, ref2b, pred2b = [], [], [], []
    k = 1
    for _ in range(4):
        ref1.append(Event(slot(k), 60, "missed")); pred1.append(Event(slot(k), 60, "missed"))
        ref2b.append(Event(slot(k), 60, "missed")); pred2b.append(Event(slot(k), 60, "missed")); k += 1
    for _ in range(4):
        ref1.append(Event(slot(k), 60, "extra")); pred1.append(Event(slot(k), 60, "extra"))
        ref2b.append(Event(slot(k), 60, "extra")); pred2b.append(Event(slot(k), 60, "extra")); k += 1
    for _ in range(2):
        ref1.append(Event(slot(k), 60, "missed")); pred1.append(Event(slot(k), 60, "extra"))
        ref2b.append(Event(slot(k), 60, "missed")); k += 1
    for _ in range(2):
        ref1.append(Event(slot(k), 60, "extra")); pred1.append(Event(slot(k), 60, "missed"))
        ref2b.append(Event(slot(k), 60, "extra")); k += 1
    for _ in range(2):
        pred2b.append(Event(slot(k), 60, "missed")); k += 1
    for _ in range(2):
        pred2b.append(Event(slot(k), 60, "extra")); k += 1

    s1 = shipped_scores(pred1, ref1)
    s2 = shipped_scores(pred2b, ref2b)
    d1 = decoupled_scores(pred1, ref1, tau=0.05)
    d2 = decoupled_scores(pred2b, ref2b, tau=0.05)
    assert abs(s1["_mean_error_f1"] - 2 / 3) < 1e-12
    assert abs(s1["_mean_error_f1"] - s2["_mean_error_f1"]) < 1e-12
    assert abs(d1.hm - 1 / 3) < 1e-12 and d2.hm == 0.0
    assert d1.mass_conserved() and d2.mass_conserved()
    log("B  non-identifiability witness: identical shipped mean-error-F1 = "
        "%.4f; HM = 1/3 vs 0. OK" % s1["_mean_error_f1"])

    # --- Max-cardinality vs greedy (the point of the rebuild) --------------
    g_pred = [Event(0.52, 60, "extra"), Event(0.45, 60, "extra")]
    g_ref = [Event(0.50, 60, "missed"), Event(0.90, 60, "missed")]
    greedy_n = len(_greedy_reference_match(g_pred, g_ref, tau=0.40))
    exact_n = len(match_events(g_pred, g_ref, tau=0.40))
    assert greedy_n == 1 and exact_n == 2, (greedy_n, exact_n)
    log("MATCHER: crafted dense case -- greedy=1 match, exact "
        "max-cardinality=2. Production matcher strictly better. OK")

    # --- mini property check ----------------------------------------------
    rng = np.random.default_rng(20260718)
    for _ in range(30):
        def rand_events(n: int) -> List[Event]:
            return [Event(float(np.round(rng.uniform(0, 1.5), 3)),
                          int(rng.integers(58, 66)),
                          str(rng.choice(SHIP_TRACKS)), "p0")
                    for _ in range(n)]
        rf = rand_events(int(rng.integers(0, 15)))
        pf = rand_events(int(rng.integers(0, 15)))
        prev = -1
        for t in TAU_GRID_S:
            r = decoupled_scores(pf, rf, tau=t)
            assert r.mass_conserved()
            assert r.localization["tp"] >= prev
            prev = r.localization["tp"]
    log("PROPERTIES: 30 random dense corpora x %d tolerances -- "
        "mass conservation + tau-monotonicity. OK" % len(TAU_GRID_S))

    log("SELF-CHECK PASSED (decoupled_scorer v%s)." % __version__)
    return 0


# --------------------------------------------------------------------------- #
# CLI (contract: experiments/setup/03_inference_commands.md).                 #
# --------------------------------------------------------------------------- #

def _to_seconds(v: float) -> float:
    """Documented unit convention: values > 1 are milliseconds, else seconds
    (the contract command line mixes ``--tau 50 ... 500`` and ``--eps 0.05``)."""
    return v / 1000.0 if v > 1.0 else v


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="decoupled_scorer.py",
        description="Decoupled localization/classification MER scorer "
                    "(IEEE SPL, Phase 2).")
    ap.add_argument("--pred_dir", help="directory of predicted multi-track MIDI")
    ap.add_argument("--gt_meta", help="ground-truth metadata JSON")
    ap.add_argument("--tau", nargs="+", type=float,
                    default=[t * 1000 for t in TAU_GRID_S],
                    help="onset tolerances; >1 = ms, <=1 = s "
                         "(default: 50 75 100 150 200 500)")
    ap.add_argument("--eps", type=float, default=DEFAULT_EPSILON_S,
                    help="wrong-note collapse radius; >1 = ms, <=1 = s "
                         "(default 0.05 s; independent of tau)")
    ap.add_argument("--collapse", choices=("strict", "pitch_aware", "none"),
                    default="strict", help="collapse rule (default strict)")
    ap.add_argument("--out", default="results.json", help="output JSON path")
    ap.add_argument("--shipped", action="store_true",
                    help="also emit the backward-compatible shipped-mode "
                         "per-track P/R/F1 at 50 ms")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="per-piece bootstrap replicates for 95%% CIs (0 = off)")
    ap.add_argument("--seed", type=int, default=20260718,
                    help="bootstrap RNG seed")
    ap.add_argument("--score_filter", type=float, default=0.0, metavar="ANCHOR",
                    help="drop predicted missed claims naming no score note "
                         "within ANCHOR (>1 = ms, <=1 = s; 0 = off) before "
                         "scoring; every mode below sees the filtered claims")
    ap.add_argument("--self-check", action="store_true",
                    help="run the built-in oracle regression and exit")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.self_check:
        return run_self_check(verbose=True)
    if not args.pred_dir or not args.gt_meta:
        build_arg_parser().error("--pred_dir and --gt_meta are required "
                                 "(or use --self-check)")

    taus = [_to_seconds(t) for t in args.tau]
    eps = _to_seconds(args.eps)

    pred = load_pred_events(args.pred_dir)
    ref = load_ref_events(args.gt_meta)
    pieces = sorted({e.piece for e in pred} | {e.piece for e in ref})

    score_filter: Optional[Dict[str, object]] = None
    if args.score_filter > 0:
        anchor = _to_seconds(args.score_filter)
        n_missed_before = sum(1 for e in pred if e.etype == "missed")
        pred, n_dropped = score_consistency_filter(pred, ref, anchor)
        score_filter = dict(anchor_s=anchor, n_missed_before=n_missed_before,
                            n_missed_after=n_missed_before - n_dropped,
                            n_dropped=n_dropped)

    sw = sweep(pred, ref, taus=taus, epsilon=eps, collapse=args.collapse)
    results: List[DecoupledResult] = sw["results"]  # type: ignore[assignment]

    payload: Dict[str, object] = dict(
        config=dict(
            scorer_version=__version__,
            pred_dir=os.path.abspath(args.pred_dir),
            gt_meta=os.path.abspath(args.gt_meta),
            tau_s=taus,
            epsilon_s=eps,
            collapse=args.collapse,
            note="PHASE-3: on-disk schema (track names, metadata layout) "
                 "must be reconfirmed against real Polytune/LadderSym "
                 "artifacts before the pilot.",
        ),
        n_pieces=len(pieces),
        n_pred_events=len(pred),
        n_ref_events=len(ref),
        score_filter=score_filter,
        decoupled=dict(
            per_tau=[r.to_jsonable() for r in results],
            mean_loc_f1_over_tau_grid=sw["mean_loc_f1_over_tau_grid"],
        ),
    )
    if args.shipped:
        payload["shipped_50ms"] = shipped_scores(pred, ref, tau=0.050)
    if args.bootstrap > 0:
        payload["bootstrap"] = {
            str(int(round(t * 1000))): bootstrap_piece_ci(
                pred, ref, tau=t, epsilon=eps, collapse=args.collapse,
                n_boot=args.bootstrap, seed=args.seed)
            for t in taus}

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("decoupled_scorer v%s: %d pieces, %d pred events, %d ref events"
          % (__version__, len(pieces), len(pred), len(ref)))
    print("  tau(ms)  loc_F1   n_loc   HM        mass_conserved")
    for r in results:
        print("   %4d    %.4f   %5d   %s     %s"
              % (int(round(r.tau * 1000)), r.localization["f1"], r.n_localized,
                 "  n/a " if r.hm is None else "%.4f" % r.hm,
                 r.mass_conserved()))
    print("  mean Loc-F over tau grid (not per-piece): %.4f"
          % sw["mean_loc_f1_over_tau_grid"])
    print("  wrote %s" % os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
