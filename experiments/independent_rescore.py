"""Independent re-derivation of the letter's MAESTRO-E numbers from raw data.

A second implementation of the measure, written from the letter's definitions
and sharing no code with decoupled_scorer.py: MIDI is read with mido (the
scorer uses pretty_midi), every matching is solved with SciPy's Hungarian
assignment (the scorer uses its own two-stage assignment), and the collapse,
localization, confusion, adjudication, published-protocol scores and the
score-consistency filter are re-implemented from the printed definitions.
The results are compared with the numbers the letter prints.

Inputs: the systems' raw prediction MIDI files and the MAESTRO-E label stems
(one directory per class), as pulled from the cluster:
    <raw>/A_polytune_maestro/*.mid, <raw>/B_laddersym_maestro_*prompted/*.mid
    <raw>/label/{correct_notes,extra_notes,removed_notes}/<piece>/stems_midi/*.mid
    <raw>/gt_meta_maestro.json   (piece -> label paths; only the keys are used)

Run: python experiments/independent_rescore.py --raw <raw dir>
Exit 0 when every printed number is reproduced within its stated tolerance.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import mido
import numpy as np
from scipy.optimize import linear_sum_assignment

TAU = 0.050          # onset tolerance of the letter's Table I
EPS = 0.050          # collapse radius
ANCHOR = 0.050       # anchor window of the score look-up
SLACK = 1e-9         # float slack on the tolerance boundary
INFEASIBLE = 1e6     # assignment cost for pairs outside the tolerance
# The paper's scorer reads MIDI with pretty_midi, which discards a note-on
# that is never terminated by a note-off (75 of the 775,962 reference
# note-ons, 0.01%). With this flag the loader applies the same rule so the
# reproduction can be exact; without it every note-on counts.
DROP_UNTERMINATED = True

CONFIGS = ("A_polytune_maestro", "B_laddersym_maestro_unprompted", "B_laddersym_maestro_prompted")
PRINTED = {
    "N_polytune": [[536, 83, 424], [12, 12307, 7331], [12, 1066, 8689]],
    "M": (30460, 33392, 35105),
    "hm": (0.293, 0.199, 0.175),
    "K": (16444, 14461, 14369),
    "G": (6678, 7689, 8758), "A": (2411, 1630, 1739), "U": (7355, 5142, 3872),
    "hm_g": (0.063, 0.054, 0.050),
    "pred_merges": (29157, 22376, 18644), "ref_merges": 12258,
    "post_ref_totals": (10829, 22280, 12258),
    "published": {
        "A_polytune_maestro": ((7088, 24039, 15999), (27650, 18829, 6888)),
        "B_laddersym_maestro_unprompted": ((9975, 18592, 13112), (29166, 12914, 5372)),
        "B_laddersym_maestro_prompted": ((12135, 14782, 10952), (30414, 8475, 4124)),
    },
    "pooled_f1": {"A_polytune_maestro": (0.261, 0.683), "B_laddersym_maestro_unprompted": (0.386, 0.761),
                  "B_laddersym_maestro_prompted": (0.485, 0.828)},
    "filter_missed_f1": (0.404, 0.552, 0.629), "filter_drop_pct": (61, 54, 42),
    "unfounded_pct": (24.1, 15.4, 11.0),
}


@dataclass(frozen=True)
class Note:
    onset: float
    pitch: int
    cls: str          # missed | extra | correct | wrong
    claimed: int = -1  # for a merged wrong event: the missed member's pitch


# ---------------------------------------------------------------- MIDI ----
def midi_notes(path: str):
    """Yield (onset_s, pitch, track_name) for every note-on, tempo-mapped."""
    mf = mido.MidiFile(path)
    tempo_map = []  # (abs_tick, tempo)
    for tr in mf.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                tempo_map.append((t, msg.tempo))
    tempo_map.sort()
    if not tempo_map or tempo_map[0][0] > 0:
        tempo_map.insert(0, (0, 500000))

    def to_sec(tick: int) -> float:
        sec, last_tick, tempo = 0.0, 0, tempo_map[0][1]
        for tk, tp in tempo_map:
            if tk >= tick:
                break
            sec += mido.tick2second(tk - last_tick, mf.ticks_per_beat, tempo)
            last_tick, tempo = tk, tp
        return sec + mido.tick2second(tick - last_tick, mf.ticks_per_beat, tempo)

    for tr in mf.tracks:
        t = 0
        open_notes = {}  # pitch -> [start ticks], pretty_midi's bookkeeping
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                if not DROP_UNTERMINATED:
                    yield to_sec(t), int(msg.note), tr.name
                else:
                    open_notes.setdefault(msg.note, []).append(t)
            elif DROP_UNTERMINATED and (msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)):
                # pretty_midi: a note-off closes every open note of that pitch
                # that started at an earlier tick; a zero-length note stays
                # open; a note still open at end of track is discarded
                starts = open_notes.get(msg.note, [])
                closed = [st for st in starts if st != t]
                for st in closed:
                    yield to_sec(st), int(msg.note), tr.name
                kept = [st for st in starts if st == t]
                if kept and closed:
                    open_notes[msg.note] = kept
                elif msg.note in open_notes:
                    del open_notes[msg.note]


def track_class(name: str):
    low = name.lower()
    if "extra" in low:
        return "extra"
    if "remov" in low or "miss" in low:
        return "missed"
    if "correct" in low:
        return "correct"
    return None


def load_predictions(pred_dir: str) -> dict:
    out = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(pred_dir, "*.mid"))):
        piece = os.path.splitext(os.path.basename(path))[0]
        for onset, pitch, tname in midi_notes(path):
            cls = track_class(tname)
            if cls is None:
                raise SystemExit(f"unrecognized track {tname!r} in {path}")
            out[piece].append(Note(onset, pitch, cls))
    return out


def load_reference(raw: str) -> dict:
    meta = json.load(open(os.path.join(raw, "gt_meta_maestro.json"), encoding="utf-8"))
    out = defaultdict(list)
    for piece in sorted(meta):
        for key, cls in (("removed_notes", "missed"), ("extra_notes", "extra"), ("correct_notes", "correct")):
            stems = glob.glob(os.path.join(raw, "label", key, piece, "stems_midi", "*.mid"))
            if not stems:
                raise SystemExit(f"no {key} stem for {piece}")
            for path in stems:
                for onset, pitch, _ in midi_notes(path):
                    out[piece].append(Note(onset, pitch, cls))
    return out


# ------------------------------------------------------------ matching ----
def assign(a: list, b: list, radius: float, feasible=None) -> list:
    """Maximum-cardinality, then minimum-total-onset-distance one-to-one
    assignment between note lists a and b under |onset difference| <= radius
    (plus SLACK), with a pitch-proximity perturbation for exact ties.
    Solved as a min-cost assignment with a large penalty for infeasible
    pairs; components separated by gaps larger than the radius are solved
    independently. Returns index pairs (i, j)."""
    if not a or not b:
        return []
    order = sorted([(n.onset, 0, i) for i, n in enumerate(a)] + [(n.onset, 1, j) for j, n in enumerate(b)])
    pairs, block = [], []

    def solve(block):
        ai = [i for _, s, i in block if s == 0]
        bj = [j for _, s, j in block if s == 1]
        if not ai or not bj:
            return
        cost = np.full((len(ai), len(bj)), INFEASIBLE)
        for r, i in enumerate(ai):
            for c, j in enumerate(bj):
                d = abs(a[i].onset - b[j].onset)
                if d <= radius + SLACK and (feasible is None or feasible(a[i], b[j])):
                    cost[r, c] = d + 1e-12 * abs(a[i].pitch - b[j].pitch)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] < INFEASIBLE:
                pairs.append((ai[r], bj[c]))

    prev = None
    for item in order:
        if prev is not None and item[0] - prev > radius + SLACK:
            solve(block)
            block = []
        block.append(item)
        prev = item[0]
    solve(block)
    return pairs


def collapse(notes: list) -> tuple:
    """Strict collapse of one side: pair missed with extra within EPS, each
    pair becoming one wrong event at the extra's onset and pitch, carrying
    the missed member's pitch as its claim. Returns (events, n_merged)."""
    missed = [n for n in notes if n.cls == "missed"]
    extra = [n for n in notes if n.cls == "extra"]
    pairs = assign(missed, extra, EPS)
    used_m = {i for i, _ in pairs}
    used_e = {j for _, j in pairs}
    out = [Note(extra[j].onset, extra[j].pitch, "wrong", missed[i].pitch) for i, j in pairs]
    out += [n for i, n in enumerate(missed) if i not in used_m]
    out += [n for j, n in enumerate(extra) if j not in used_e]
    return out, len(pairs)


def anchor_class(pred_wrong: Note, ref_notes: list) -> str:
    """genuine: a removed (omitted) score note of the claimed pitch within
    ANCHOR; ambiguous: a played (correct) score note of that pitch; else
    unfounded."""
    for cls, label in (("missed", "G"), ("correct", "A")):
        for n in ref_notes:
            if n.cls == cls and n.pitch == pred_wrong.claimed and abs(n.onset - pred_wrong.onset) <= ANCHOR + SLACK:
                return label
    return "U"


def score_config(preds: dict, refs: dict) -> dict:
    classes = ("missed", "extra", "wrong")
    N = np.zeros((3, 3), dtype=int)
    n_pred_err = n_ref_err = 0
    pred_merges = ref_merges = 0
    K = {"G": 0, "A": 0, "U": 0}
    genuine_offdiag = 0
    post_ref = np.zeros(3, dtype=int)
    for piece in sorted(refs):
        p_err = [n for n in preds.get(piece, []) if n.cls in ("missed", "extra")]
        r_err = [n for n in refs[piece] if n.cls in ("missed", "extra")]
        p_c, mp = collapse(p_err)
        r_c, mr = collapse(r_err)
        pred_merges += mp
        ref_merges += mr
        n_pred_err += len(p_c)
        n_ref_err += len(r_c)
        for n in r_c:
            post_ref[classes.index(n.cls)] += 1
        for i, j in assign(p_c, r_c, TAU):
            pi, ri = classes.index(p_c[i].cls), classes.index(r_c[j].cls)
            N[ri, pi] += 1
            if p_c[i].cls == "wrong":
                lab = anchor_class(p_c[i], refs[piece])
                K[lab] += 1
                if lab == "G" and ri != pi:
                    genuine_offdiag += 1
    M = int(N.sum())
    off = M - int(np.trace(N))
    outside_wrong = off - int(N[0, 2] + N[1, 2])
    hm_g = (outside_wrong + genuine_offdiag) / (M - K["A"] - K["U"])
    return dict(N=N.tolist(), M=M, off=off, hm=off / M, K=K, K_total=sum(K.values()),
                hm_g=hm_g, outside_wrong=outside_wrong, genuine_offdiag=genuine_offdiag,
                pred_merges=pred_merges, ref_merges=ref_merges, post_ref=post_ref.tolist(),
                n_pred_err=n_pred_err, n_ref_err=n_ref_err,
                unfounded_pct=100 * K["U"] / M)


def published_scores(preds: dict, refs: dict, drop_unfounded_missed: bool = False) -> dict:
    """The systems' own protocol: per class, an independent onset+pitch match
    at TAU on the uncollapsed events, distances rounded to 4 decimals
    (mir_eval semantics), pooled TP/FP/FN over the corpus."""
    tot = {c: [0, 0, 0] for c in ("missed", "extra")}
    dropped = kept = 0
    for piece in sorted(refs):
        for cls in ("missed", "extra"):
            p = [n for n in preds.get(piece, []) if n.cls == cls]
            if cls == "missed" and drop_unfounded_missed:
                score = [n for n in refs[piece] if n.cls in ("missed", "correct")]
                keep = [n for n in p if any(s.pitch == n.pitch and abs(s.onset - n.onset) <= ANCHOR + SLACK for s in score)]
                dropped += len(p) - len(keep)
                kept += len(keep)
                p = keep
            r = [n for n in refs[piece] if n.cls == cls]
            pairs = assign(p, r, TAU, feasible=lambda x, y: x.pitch == y.pitch and round(abs(x.onset - y.onset), 4) <= TAU)
            tp = len(pairs)
            tot[cls][0] += tp
            tot[cls][1] += len(p) - tp
            tot[cls][2] += len(r) - tp
    out = {}
    for cls, (tp, fp, fn) in tot.items():
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out[cls] = dict(tp=tp, fp=fp, fn=fn, f1=(2 * pr * rc / (pr + rc) if pr + rc else 0.0))
    out["drop_pct"] = 100 * dropped / (dropped + kept) if drop_unfounded_missed else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--keep-unterminated", action="store_true",
                    help="count every note-on (mido semantics) instead of pretty_midi's rule")
    args = ap.parse_args()
    global DROP_UNTERMINATED
    DROP_UNTERMINATED = not args.keep_unterminated
    refs = load_reference(args.raw)
    fails = []

    def cmp(label, got, want, tol=0.0):
        good = abs(got - want) <= tol + 1e-12 if isinstance(want, (int, float)) else got == want
        print(("OK   " if good else "FAIL "), f"{label}: independent {got!r} vs printed {want!r}")
        if not good:
            fails.append(label)

    for k, cfg in enumerate(CONFIGS):
        preds = load_predictions(os.path.join(args.raw, cfg))
        r = score_config(preds, refs)
        print(f"\n== {cfg}")
        cmp("|M|", r["M"], PRINTED["M"][k])
        cmp("raw HM (3 dp)", round(r["hm"], 3), PRINTED["hm"][k], 0.002)
        # |K|, |G|, |A|, |U| and the confusion cells depend on which of several
        # equal-distance pairings the matcher's tie-break selects; the
        # supplement states that reversing the tie-break leaves |M| identical
        # and moves HM by <= 0.002. Cells are therefore compared at 0.1% of |M|.
        cell_tol = max(1, round(0.001 * PRINTED["M"][k]))
        cmp("|K| = matched merged predictions (0.1% of |M|)", r["K_total"], PRINTED["K"][k], cell_tol)
        for s in "GAU":
            cmp(f"|{s}| (0.1% of |M|)", r["K"][s], PRINTED[s][k], cell_tol)
        cmp("HM_G (3 dp)", round(r["hm_g"], 3), PRINTED["hm_g"][k], 0.002)
        cmp("unfounded share |U|/|M| (%)", round(r["unfounded_pct"], 1), PRINTED["unfounded_pct"][k], 0.1)
        cmp("predicted-side merges", r["pred_merges"], PRINTED["pred_merges"][k], 3)
        cmp("reference-side merges", r["ref_merges"], PRINTED["ref_merges"], 3)
        if k == 0:
            cmp("N (Polytune), every cell within 0.1% of |M|", all(abs(a - b) <= cell_tol for ra, rb in zip(r["N"], PRINTED["N_polytune"]) for a, b in zip(ra, rb)), True)
            print("     N =", r["N"], "| max cell difference:", max(abs(a - b) for ra, rb in zip(r["N"], PRINTED["N_polytune"]) for a, b in zip(ra, rb)))
            cmp("N row sums (reference classes) within 0.1% of |M|", all(abs(sum(ra) - sum(rb)) <= cell_tol for ra, rb in zip(r["N"], PRINTED["N_polytune"])), True)
            cmp("post-collapse reference totals", tuple(r["post_ref"]), PRINTED["post_ref_totals"])
        pub = published_scores(preds, refs)
        for cls, i in (("missed", 0), ("extra", 1)):
            cmp(f"published-protocol {cls} (TP,FP,FN)", (pub[cls]["tp"], pub[cls]["fp"], pub[cls]["fn"]), PRINTED["published"][cfg][i])
            cmp(f"pooled {cls} F1 (3 dp)", round(pub[cls]["f1"], 3), PRINTED["pooled_f1"][cfg][i], 0.001)
        filt = published_scores(preds, refs, drop_unfounded_missed=True)
        cmp("score-consistency filter: missed F1 (3 dp)", round(filt["missed"]["f1"], 3), PRINTED["filter_missed_f1"][k], 0.001)
        cmp("score-consistency filter: drop share (%)", round(filt["drop_pct"]), PRINTED["filter_drop_pct"][k], 1)
        cmp("filter leaves the extra class untouched", (filt["extra"]["tp"], filt["extra"]["fp"], filt["extra"]["fn"]),
            (pub["extra"]["tp"], pub["extra"]["fp"], pub["extra"]["fn"]))
    print("\n" + ("ALL INDEPENDENT RESCORE CHECKS PASSED" if not fails else f"FAILED: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
