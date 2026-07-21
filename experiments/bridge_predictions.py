#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bridge_predictions.py -- committed, asserting index->name bridge for
Polytune/LadderSym predicted MIDI (MER evaluation pipeline, Phase 3).

Polytune / LadderSym inference writes each piece as one multi-track MIDI in
which the error class is encoded ONLY by track INDEX
(0 -> Extra, 1 -> Removed, 2 -> Correct).

Track NAMES carry no class information and must never be relied on. The
shipped repos' own eval prints them as "Acoustic Grand Piano", but the
prediction files as written to disk read back UNNAMED (empty track name) —
verified against real Gilbreth artifacts 2026-07-20. The bridge is purely
positional, and track_census.py enforces the exactly-3-note-bearing-tracks
contract that makes the positional read safe.
The decoupled scorer (``experiments/decoupled_scorer.py``,
``load_pred_events``) selects the event type by track NAME, so the two
conventions must be bridged. This module rewrites the track names
positionally and HARD-VERIFIES that the rewrite changed nothing else.

It is the committed replacement for the unsaved one-off that produced
``~/mer/run/preds_flat`` during the smoke run (the one-off survives only as
the heredoc in ``wsl_smoke/t7_flatten_preds.sh``): the same mido-based
byte-level track-name edit, plus the asserts the one-off lacked.

GUARANTEES -- hard asserts, fail-loud, nonzero exit NAMING the offending file
-----------------------------------------------------------------------------
 1. Every input file has EXACTLY 3 note-bearing tracks. This is the critical
    guard: a !=3-track file would silently SHIFT the positional
    index -> class mapping and mislabel every downstream event, with no
    other symptom.
 2. Per track (positionally), the multiset of (onset_tick, pitch) note
    onsets is EQUAL before and after the rewrite -- recomputed from the
    bytes actually re-read from disk, not from the in-memory object.
 3. Output total note-onset count equals the input count (implied by 2;
    asserted independently as a belt-and-braces check).
 4. The re-read output tracks carry exactly the intended names, and
    ticks_per_beat is unchanged (tick multisets are only comparable at a
    fixed resolution).

Input layouts (auto-detected, may be mixed):
  * nested: ``<in_dir>/.../<id>/<pattern>`` -- pattern defaults to
    ``mix.mid``; the piece id is the name of the directory containing the
    matched file (matches the Polytune smoke layout
    ``out/<tag>/<subset>/<id>/mix.mid``).
  * flat:   ``<in_dir>/<id>.mid`` (or ``.midi``).

Output: ``<out_dir>/<id>.mid``, flat. Refuses duplicate piece ids and
refuses to overwrite an input file in place.

No side effects on import. Deterministic: inputs are processed in sorted
order and mido serialization is deterministic.

CLI:
    python bridge_predictions.py --in_dir <dir> --out_dir <dir> \
        [--pattern mix.mid]

Exit status: 0 = all files bridged and verified; 1 = contract violation
(message on stderr names the file); 2 = bad invocation / no inputs.

Dependencies: mido only (byte-level edit; the same tool the one-off used).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import mido

__version__ = "1.0.0"

__all__ = [
    "TRACK_NAMES",
    "BridgeError",
    "note_tracks",
    "onset_multiset",
    "discover_inputs",
    "bridge_file",
    "main",
]

# Positional index -> class name. THE central convention this module exists
# to enforce; must stay in sync with decoupled_scorer.TRACK_NAME_RULES
# ("extra" -> extra, "remov" -> missed, "correct" -> correct).
TRACK_NAMES: Tuple[str, str, str] = ("Extra", "Removed", "Correct")


class BridgeError(RuntimeError):
    """Any per-file contract violation. The message always names the file."""


# --------------------------------------------------------------------------- #
# MIDI inspection helpers (pure functions on mido objects).                    #
# --------------------------------------------------------------------------- #

def note_tracks(mid: mido.MidiFile) -> List[mido.MidiTrack]:
    """The note-bearing tracks of ``mid``, in file order.

    A track is note-bearing iff it contains any ``note_on`` message
    (velocity-0 note_ons included -- same criterion as the original one-off,
    so the positional mapping is identical). Tempo/meta-only tracks are
    passed through untouched.
    """
    return [t for t in mid.tracks if any(m.type == "note_on" for m in t)]


def onset_multiset(track: mido.MidiTrack) -> Counter:
    """Multiset of (absolute_onset_tick, pitch) note onsets in ``track``.

    A note onset is a ``note_on`` with velocity > 0 (velocity 0 is a
    note-off by convention). Absolute ticks are cumulative delta times, so
    the multiset is invariant under any rewrite that preserves timing.
    """
    tick = 0
    onsets: Counter = Counter()
    for msg in track:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            onsets[(tick, msg.note)] += 1
    return onsets


def _track_names(track: mido.MidiTrack) -> List[str]:
    return [m.name for m in track if m.type == "track_name"]


# --------------------------------------------------------------------------- #
# Input discovery.                                                             #
# --------------------------------------------------------------------------- #

def discover_inputs(in_dir: str, pattern: str = "mix.mid") -> List[Tuple[str, str]]:
    """Return sorted (piece_id, path) pairs found under ``in_dir``.

    Nested hits (``<in_dir>/**/<pattern>``) take the parent directory name as
    the piece id; flat hits (``<in_dir>/<id>.mid[i]``) take the file stem.
    Raises BridgeError on duplicate ids (two inputs would silently race for
    the same output file) or if nothing is found.
    """
    if not os.path.isdir(in_dir):
        raise BridgeError("input directory does not exist: %s" % in_dir)
    pairs: Dict[str, str] = {}

    def add(pid: str, path: str) -> None:
        if pid in pairs and os.path.abspath(pairs[pid]) != os.path.abspath(path):
            raise BridgeError(
                "duplicate piece id %r: %s vs %s -- refusing (outputs would "
                "silently overwrite each other)" % (pid, pairs[pid], path))
        pairs[pid] = path

    for path in sorted(glob.glob(os.path.join(in_dir, "**", pattern),
                                 recursive=True)):
        add(os.path.basename(os.path.dirname(os.path.abspath(path))), path)
    for ext in ("*.mid", "*.midi"):
        for path in sorted(glob.glob(os.path.join(in_dir, ext))):
            add(os.path.splitext(os.path.basename(path))[0], path)

    if not pairs:
        raise BridgeError(
            "no inputs found in %s (looked for **/%s and top-level *.mid/*.midi)"
            % (in_dir, pattern))
    return sorted(pairs.items())


# --------------------------------------------------------------------------- #
# The bridge itself.                                                           #
# --------------------------------------------------------------------------- #

def bridge_file(src: str, dst: str,
                names: Sequence[str] = TRACK_NAMES) -> Dict[str, object]:
    """Bridge one file src -> dst and hard-verify the result.

    Returns a small summary dict (original names, per-track note counts) on
    success; raises BridgeError naming ``src``/``dst`` on ANY violation of
    the module contract (see module docstring, guarantees 1-4).
    """
    if os.path.abspath(src) == os.path.abspath(dst):
        raise BridgeError("%s: refusing to rewrite in place (out_dir must "
                          "differ from the input location)" % src)

    mid = mido.MidiFile(src)
    in_tracks = note_tracks(mid)
    # --- Guarantee 1: exactly 3 note tracks, else the positional mapping    #
    # --- (index 0/1/2 -> Extra/Removed/Correct) silently shifts.            #
    if len(in_tracks) != len(names):
        raise BridgeError(
            "%s: expected exactly %d note-bearing tracks, found %d -- a "
            "!=%d-track file would silently shift the positional "
            "index->class mapping; refusing"
            % (src, len(names), len(in_tracks), len(names)))

    before = [onset_multiset(t) for t in in_tracks]
    original_names = [_track_names(t) for t in in_tracks]

    for track, newname in zip(in_tracks, names):
        had = False
        for msg in track:
            if msg.type == "track_name":
                msg.name = newname
                had = True
        if not had:
            track.insert(0, mido.MetaMessage("track_name", name=newname, time=0))
    mid.save(dst)

    # --- Verify against the bytes actually written, not the live object. --- #
    out = mido.MidiFile(dst)
    if out.ticks_per_beat != mid.ticks_per_beat:  # Guarantee 4 (resolution)
        raise BridgeError(
            "%s: ticks_per_beat changed %d -> %d on rewrite of %s"
            % (dst, mid.ticks_per_beat, out.ticks_per_beat, src))
    out_tracks = note_tracks(out)
    if len(out_tracks) != len(names):
        raise BridgeError(
            "%s: output has %d note-bearing tracks, expected %d (input %s)"
            % (dst, len(out_tracks), len(names), src))
    for i, (track, newname) in enumerate(zip(out_tracks, names)):
        got = set(_track_names(track))
        if got != {newname}:  # Guarantee 4 (names)
            raise BridgeError(
                "%s: output note-track %d is named %r, expected exactly {%r} "
                "(input %s)" % (dst, i, sorted(got), newname, src))

    after = [onset_multiset(t) for t in out_tracks]
    for i, (b, a) in enumerate(zip(before, after)):  # Guarantee 2
        if a != b:
            gained = sorted((a - b).elements())[:5]
            lost = sorted((b - a).elements())[:5]
            raise BridgeError(
                "%s: track %d (%r) (onset_tick, pitch) multiset CHANGED by "
                "the rewrite of %s -- e.g. gained %s, lost %s"
                % (dst, i, names[i], src, gained, lost))
    n_in = sum(sum(c.values()) for c in before)
    n_out = sum(sum(c.values()) for c in after)
    if n_in != n_out:  # Guarantee 3 (unreachable if 2 held; kept anyway)
        raise BridgeError("%s: output note count %d != input %d (input %s)"
                          % (dst, n_out, n_in, src))

    return dict(
        src=src, dst=dst,
        original_names=original_names,
        note_counts=[sum(c.values()) for c in after],
        n_notes=n_out,
    )


# --------------------------------------------------------------------------- #
# CLI.                                                                          #
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bridge_predictions.py",
        description="Rewrite Polytune/LadderSym index-encoded prediction "
                    "tracks to named Extra/Removed/Correct tracks, with "
                    "hard content-preservation asserts.")
    ap.add_argument("--in_dir", required=True,
                    help="directory of predicted MIDI (nested <id>/mix.mid "
                         "or flat <id>.mid)")
    ap.add_argument("--out_dir", required=True,
                    help="output directory for flat <id>.mid files")
    ap.add_argument("--pattern", default="mix.mid",
                    help="filename matched in nested layouts (default mix.mid)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        inputs = discover_inputs(args.in_dir, args.pattern)
    except BridgeError as exc:
        print("BRIDGE FAIL: %s" % exc, file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    total_notes = 0
    for pid, src in inputs:
        dst = os.path.join(args.out_dir, pid + ".mid")
        try:
            info = bridge_file(src, dst)
        except BridgeError as exc:
            print("BRIDGE FAIL: %s" % exc, file=sys.stderr)
            return 1
        orig = sorted({n for per_track in info["original_names"]
                       for n in per_track}) or ["<unnamed>"]
        print("%s: 3 tracks -> %s  notes=%s  (orig names: %s)  VERIFIED"
              % (pid, "/".join(TRACK_NAMES), info["note_counts"],
                 ", ".join(repr(n) for n in orig)))
        total_notes += int(info["n_notes"])  # type: ignore[arg-type]
    print("bridge_predictions v%s: %d file(s) bridged into %s, %d note "
          "onsets, all asserts passed."
          % (__version__, len(inputs), os.path.abspath(args.out_dir),
             total_notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
