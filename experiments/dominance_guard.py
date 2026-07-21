#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dominance_guard.py -- per-piece diagonal-dominance preflight for the MER
evaluation pipeline (run BEFORE scoring a sweep).

WHAT IT GUARDS
--------------
The bridged predictions (``bridge_predictions.py``) assign class by track
POSITION (0 -> Extra, 1 -> Removed, 2 -> Correct). If that positional
convention were ever wrong -- a model version that emits tracks in a
different order, a !=3-track file that slipped through, a stale bridge --
the symptom downstream is silent label permutation, not a crash. This guard
detects it the same way the verifier did: build, per piece, the 3x3
match-count matrix (pred track = row, GT label = column) and require the
diagonal to dominate. Under a permuted mapping the mass moves off-diagonal
and the guard trips loudly.

MATCH SEMANTICS (identical to the verifier's)
---------------------------------------------
Match counts come from ``mir_eval.transcription.match_notes`` -- the exact
matcher ``precision_recall_f1_overlap`` calls internally -- at the default
50 ms onset tolerance, default 50-cent pitch tolerance (exact semitone for
integer-MIDI pitches converted to Hz), ``offset_ratio=None`` (offsets
ignored). Cell [r][c] = size of the maximum onset+pitch matching between
pred track r's notes and GT label c's notes.

PER-PIECE ASSERT (hard, nonzero exit on any failure)
----------------------------------------------------
For each ERROR row r in {Extra, Removed}: the diagonal cell M[r][r] must be
  * the row maximum among the ERROR columns, AND
  * the maximum of its column (over all three rows).
The Correct row is checked too but only WARNED on: the Correct track is
huge (thousands of notes vs hundreds) and cannot realistically be flipped
by a permutation without the error rows tripping first; and legitimate
error/correct confusion should not block a sweep. Similarly, an error row's
Correct-column cell exceeding its diagonal is a warning, not a failure.

OUTPUT
------
Per-piece pass/fail table and the grand (summed) matrix on stdout; full
machine-readable report via ``--out`` (default ``guard_report.json``).
Exit status: 0 = all pieces pass; 1 = at least one piece failed the hard
assert; 2 = bad invocation / unreadable inputs (message names the file).

CLI (paths as produced by bridge_predictions.py + the smoke gt meta):

    python dominance_guard.py \
        --pred_dir <flat dir of bridged <id>.mid> \
        --gt_meta <json: id -> {extra_notes_midi, removed_notes_midi,
                                correct_notes_midi}> \
        [--onset_tol 50] [--out guard_report.json]

``--onset_tol`` follows the house unit convention (decoupled_scorer.py):
a value > 1 is milliseconds, a value <= 1 is seconds. Default 50 ms.

No side effects on import. Deterministic (pieces and tracks processed in
sorted/fixed order; ``match_notes`` is deterministic).

Dependencies: numpy; pretty_midi and mir_eval (imported lazily, loader/
matcher only) -- same stack the verifier used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__version__ = "1.0.0"

__all__ = [
    "ROW_NAMES",
    "GT_KEYS",
    "GuardError",
    "load_pred_tracks",
    "load_gt_tracks",
    "match_count",
    "piece_matrix",
    "check_matrix",
    "main",
]

# Row convention: pred track name -> row index (bridge_predictions.TRACK_NAMES).
ROW_NAMES: Tuple[str, str, str] = ("Extra", "Removed", "Correct")
# Column convention: gt_meta key -> column index (same class order).
GT_KEYS: Tuple[str, str, str] = ("extra_notes_midi", "removed_notes_midi",
                                 "correct_notes_midi")
ERROR_IDX: Tuple[int, int] = (0, 1)   # Extra, Removed
CORRECT_IDX: int = 2

DEFAULT_ONSET_TOL_S = 0.050

# One (onset_s, pitch_midi) note list per class.
NoteList = List[Tuple[float, int]]


class GuardError(RuntimeError):
    """Unreadable/malformed input. The message always names the file."""


# --------------------------------------------------------------------------- #
# Loaders (schema exactly as bridge_predictions.py / smoke_gt_meta.json).      #
# --------------------------------------------------------------------------- #

def _midi_notes(path: str) -> NoteList:
    """All (start_s, pitch) notes in a MIDI file, all instruments pooled."""
    import pretty_midi  # lazy: loader-only dependency
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception as exc:
        raise GuardError("cannot read MIDI %s: %s" % (path, exc))
    return [(float(n.start), int(n.pitch))
            for inst in pm.instruments for n in inst.notes]


def load_pred_tracks(path: str) -> Dict[str, NoteList]:
    """Load a bridged prediction file into {row name: notes}.

    Hard requirements (GuardError otherwise, naming the file): the file
    contains exactly the three instruments Extra / Removed / Correct, each
    exactly once. Anything else means the file was not produced by (or has
    drifted from) bridge_predictions.py, and positional trust is void.
    """
    import pretty_midi  # lazy
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception as exc:
        raise GuardError("cannot read predicted MIDI %s: %s" % (path, exc))
    tracks: Dict[str, NoteList] = {}
    for inst in pm.instruments:
        name = (inst.name or "").strip()
        if name not in ROW_NAMES:
            raise GuardError(
                "%s: unexpected track name %r (expected one of %s) -- run "
                "bridge_predictions.py first" % (path, name, list(ROW_NAMES)))
        if name in tracks:
            raise GuardError("%s: duplicate track name %r" % (path, name))
        tracks[name] = [(float(n.start), int(n.pitch)) for n in inst.notes]
    missing = [n for n in ROW_NAMES if n not in tracks]
    if missing:
        raise GuardError("%s: missing track(s) %s" % (path, missing))
    return tracks


def load_gt_tracks(entry: Mapping[str, object], piece: str,
                   base: str) -> Dict[str, NoteList]:
    """Load one gt_meta entry into {gt key: notes}.

    All three keys are required: a dominance check with a missing column is
    vacuous, so absence is treated as a schema error, loudly.
    Relative paths resolve against the metadata file's directory (same rule
    as decoupled_scorer.load_ref_events).
    """
    if not isinstance(entry, Mapping):
        raise GuardError("gt_meta entry for %r is not an object" % piece)
    out: Dict[str, NoteList] = {}
    for key in GT_KEYS:
        path = entry.get(key)
        if not path or not isinstance(path, str):
            raise GuardError("gt_meta entry for %r lacks key %r (all three "
                             "label MIDIs are required)" % (piece, key))
        if not os.path.isabs(path):
            path = os.path.join(base, path)
        out[key] = _midi_notes(path)
    return out


# --------------------------------------------------------------------------- #
# Match-count matrix (verifier semantics).                                     #
# --------------------------------------------------------------------------- #

def _hz(pitches: Sequence[int]) -> np.ndarray:
    return 440.0 * (2.0 ** ((np.asarray(pitches, dtype=float) - 69.0) / 12.0))


def match_count(pred: NoteList, ref: NoteList,
                onset_tol: float = DEFAULT_ONSET_TOL_S) -> int:
    """|maximum matching| between two note lists, verifier semantics.

    ``mir_eval.transcription.match_notes`` (the matcher inside
    ``precision_recall_f1_overlap``) with ``onset_tolerance=onset_tol``,
    default pitch tolerance (50 cents = exact semitone for integer MIDI),
    ``offset_ratio=None``. Offsets are ignored under offset_ratio=None; a
    fixed dummy duration keeps the interval arrays valid.
    """
    if not pred or not ref:
        return 0
    import mir_eval  # lazy: matcher-only dependency
    ref_on = np.array([n[0] for n in ref], dtype=float)
    est_on = np.array([n[0] for n in pred], dtype=float)
    ref_int = np.stack([ref_on, ref_on + 0.1], axis=1)
    est_int = np.stack([est_on, est_on + 0.1], axis=1)
    matching = mir_eval.transcription.match_notes(
        ref_int, _hz([n[1] for n in ref]),
        est_int, _hz([n[1] for n in pred]),
        onset_tolerance=onset_tol, offset_ratio=None)
    return len(matching)


def piece_matrix(pred: Dict[str, NoteList], gt: Dict[str, NoteList],
                 onset_tol: float = DEFAULT_ONSET_TOL_S) -> List[List[int]]:
    """3x3 match-count matrix: rows = pred tracks (ROW_NAMES order),
    columns = GT labels (GT_KEYS order)."""
    return [[match_count(pred[row], gt[key], onset_tol) for key in GT_KEYS]
            for row in ROW_NAMES]


def check_matrix(m: Sequence[Sequence[int]]) -> Dict[str, object]:
    """Apply the dominance rules (module docstring) to one 3x3 matrix.

    Returns dict(passed=bool, failures=[...], warnings=[...]). Failures come
    only from the two error rows; the Correct row and the error-row-vs-
    Correct-column comparison produce warnings.
    """
    failures: List[str] = []
    warnings: List[str] = []
    for r in ERROR_IDX:
        diag = m[r][r]
        # HARD: diagonal is the row maximum among ERROR columns...
        for c in ERROR_IDX:
            if c != r and m[r][c] > diag:
                failures.append(
                    "%s row: off-diagonal error cell [%s][%s]=%d exceeds "
                    "diagonal %d" % (ROW_NAMES[r], ROW_NAMES[r], ROW_NAMES[c],
                                     m[r][c], diag))
        # HARD: ...and the column maximum over all rows.
        for r2 in range(3):
            if r2 != r and m[r2][r] > diag:
                failures.append(
                    "%s column: cell [%s][%s]=%d exceeds diagonal %d"
                    % (ROW_NAMES[r], ROW_NAMES[r2], ROW_NAMES[r],
                       m[r2][r], diag))
        # WARN: error mass landing on GT-correct beyond the diagonal.
        if m[r][CORRECT_IDX] > diag:
            warnings.append(
                "%s row: Correct-column count %d exceeds diagonal %d "
                "(warn-only)" % (ROW_NAMES[r], m[r][CORRECT_IDX], diag))
    # WARN-ONLY: Correct-track dominance (huge; cannot realistically flip).
    cd = m[CORRECT_IDX][CORRECT_IDX]
    for k in ERROR_IDX:
        if m[CORRECT_IDX][k] > cd:
            warnings.append(
                "Correct row: cell [Correct][%s]=%d exceeds diagonal %d "
                "(warn-only)" % (ROW_NAMES[k], m[CORRECT_IDX][k], cd))
        if m[k][CORRECT_IDX] > cd:
            warnings.append(
                "Correct column: cell [%s][Correct]=%d exceeds diagonal %d "
                "(warn-only)" % (ROW_NAMES[k], m[k][CORRECT_IDX], cd))
    return dict(passed=not failures, failures=failures, warnings=warnings)


# --------------------------------------------------------------------------- #
# CLI.                                                                          #
# --------------------------------------------------------------------------- #

def _to_seconds(v: float) -> float:
    """House unit convention (decoupled_scorer.py): >1 = ms, <=1 = s."""
    return v / 1000.0 if v > 1.0 else v


def _fmt_matrix(m: Sequence[Sequence[int]], indent: str = "    ") -> str:
    head = indent + "%-9s" % "" + "".join("%9s" % n for n in ROW_NAMES)
    rows = [indent + "%-9s" % ROW_NAMES[r]
            + "".join("%9d" % m[r][c] for c in range(3))
            for r in range(3)]
    return "\n".join([head] + rows)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dominance_guard.py",
        description="Per-piece diagonal-dominance check on bridged MER "
                    "predictions; run before scoring a sweep.")
    ap.add_argument("--pred_dir", required=True,
                    help="flat directory of bridged <id>.mid predictions")
    ap.add_argument("--gt_meta", required=True,
                    help="JSON: id -> {extra_notes_midi, removed_notes_midi, "
                         "correct_notes_midi}")
    ap.add_argument("--onset_tol", type=float, default=50.0,
                    help="onset tolerance; >1 = ms, <=1 = s (default 50 ms)")
    ap.add_argument("--out", default="guard_report.json",
                    help="machine-readable report path")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tol = _to_seconds(args.onset_tol)

    try:
        with open(args.gt_meta, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if not isinstance(meta, dict):
            raise GuardError("%s: gt_meta must be a JSON object" % args.gt_meta)
        pred_files = {os.path.splitext(f)[0]: os.path.join(args.pred_dir, f)
                      for f in os.listdir(args.pred_dir)
                      if f.lower().endswith((".mid", ".midi"))}
        if not pred_files:
            raise GuardError("no .mid/.midi files in %s" % args.pred_dir)
        missing_meta = sorted(set(pred_files) - set(meta))
        if missing_meta:
            raise GuardError("pieces missing from gt_meta: %s" % missing_meta)
    except (OSError, ValueError, GuardError) as exc:
        print("GUARD ERROR: %s" % exc, file=sys.stderr)
        return 2

    base = os.path.dirname(os.path.abspath(args.gt_meta))
    grand = np.zeros((3, 3), dtype=int)
    per_piece: Dict[str, dict] = {}
    n_fail = 0

    print("dominance_guard v%s: %d piece(s), onset_tol=%d ms, "
          "pitch=exact semitone, offset_ratio=None"
          % (__version__, len(pred_files), int(round(tol * 1000))))
    for pid in sorted(pred_files):
        try:
            pred = load_pred_tracks(pred_files[pid])
            gt = load_gt_tracks(meta[pid], pid, base)
        except GuardError as exc:
            print("GUARD ERROR: %s" % exc, file=sys.stderr)
            return 2
        m = piece_matrix(pred, gt, tol)
        verdict = check_matrix(m)
        grand += np.asarray(m, dtype=int)
        n_fail += not verdict["passed"]
        per_piece[pid] = dict(
            matrix=m,
            pred_counts={n: len(pred[n]) for n in ROW_NAMES},
            gt_counts={k: len(gt[k]) for k in GT_KEYS},
            **verdict,
        )
        print("piece %s: %s" % (pid, "PASS" if verdict["passed"] else "FAIL"))
        print(_fmt_matrix(m))
        for msg in verdict["failures"]:
            print("    FAIL: %s" % msg)
        for msg in verdict["warnings"]:
            print("    warn: %s" % msg)

    grand_list = grand.tolist()
    grand_verdict = check_matrix(grand_list)  # informational; per-piece gates
    print("grand matrix (%d pieces summed): %s"
          % (len(per_piece), "dominant" if grand_verdict["passed"]
             else "NOT dominant"))
    print(_fmt_matrix(grand_list))

    all_pass = n_fail == 0
    report = dict(
        config=dict(
            guard_version=__version__,
            pred_dir=os.path.abspath(args.pred_dir),
            gt_meta=os.path.abspath(args.gt_meta),
            onset_tolerance_s=tol,
            pitch_tolerance="mir_eval default (50 cents; exact semitone "
                            "for integer MIDI)",
            offset_ratio=None,
            row_order=list(ROW_NAMES),
            col_order=list(GT_KEYS),
        ),
        per_piece=per_piece,
        grand_matrix=grand_list,
        grand_check=grand_verdict,
        n_pieces=len(per_piece),
        n_pass=len(per_piece) - n_fail,
        n_fail=n_fail,
        all_pass=all_pass,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("verdict: %d/%d pieces pass; wrote %s"
          % (len(per_piece) - n_fail, len(per_piece),
             os.path.abspath(args.out)))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
