"""MAESTRO-EI: labelled error injector with a per-injection manifest.

Design follows research/notes/synthetic-error-corpus-generation.md:
- Error-type mix targets the AUTHENTIC beginner counts (75/51/35 of 161):
  substitution 46.6%, insertion 31.7%, omission 21.7% -- against the released
  MAESTRO-E generator's implied 25/50/25 over note-level draws.
- Correct notes are copied VERBATIM (no timing mutation), so `correct_notes`
  labels are clean; the release's 300 ms jitter on correct notes is a
  documented divergence, not replicated.
- The manifest records, per injection: score-note index, onsets, pitches,
  drawn parameters, and for substitutions an explicit PAIRING EDGE
  (removed_id -> inserted_id). Negative controls are unpaired
  deletion+insertion pairs placed >= 2 s apart, flagged; a scorer's merge rule
  must NOT pair them.
- Selection: a FIXED fraction lambda = 0.25 of notes per piece (the release's
  mean), deterministic per piece via a seeded RNG (sha1 of piece id + SEED).

Layout mirrors MAESTRO-E so the existing inference driver runs unchanged:
  <out>/score/<piece>/mix.mid      (copied)
  <out>/mistake/<piece>/mix.mid    (injected)
  <out>/label/{correct,extra,removed}_notes/<piece>.mid
  <out>/manifest/<piece>.json
"""
import argparse, hashlib, json, os, shutil, sys
import numpy as np
import pretty_midi

SEED = 20260831
MIX = {"substitution": 75 / 161, "insertion": 51 / 161, "omission": 35 / 161}
LAMBDA = 0.25
NEG_CONTROL_FRAC = 0.05          # of substitution count, as unpaired del+ins
SUB_DELTA_P = {1: 0.856, 2: 0.144}   # |semitone| distribution, per release measurement


def rng_for(piece: str) -> np.random.Generator:
    h = int(hashlib.sha1((piece + str(SEED)).encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h)


def write_track(notes, path, program=0):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    inst.notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    pm.instruments.append(inst)
    pm.write(path)


def inject_piece(score_mid, piece, out):
    rng = rng_for(piece)
    pm = pretty_midi.PrettyMIDI(score_mid)
    notes = [n for inst in pm.instruments for n in inst.notes]
    notes.sort(key=lambda n: (n.start, n.pitch))
    N = len(notes)
    k = max(1, int(round(LAMBDA * N)))
    chosen = sorted(rng.choice(N, size=min(k, N), replace=False).tolist())
    types = rng.choice(list(MIX), size=len(chosen), p=list(MIX.values()))

    correct, extra, removed = [], [], []
    manifest = {"piece": piece, "seed": SEED, "lambda": LAMBDA,
                "mix": MIX, "n_score_notes": N, "injections": [],
                "pairing": [], "negative_controls": []}
    touched = set()
    NoteT = pretty_midi.Note

    def clone(n, **kw):
        d = dict(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end)
        d.update(kw)
        return NoteT(**d)

    ins_id = 0
    for idx, t in zip(chosen, types):
        n = notes[idx]
        touched.add(idx)
        if t == "substitution":
            mag = int(rng.choice([1, 2], p=[SUB_DELTA_P[1], SUB_DELTA_P[2]]))
            sgn = int(rng.choice([-1, 1]))
            newp = int(np.clip(n.pitch + sgn * mag, 21, 108))
            removed.append(clone(n))
            extra.append(clone(n, pitch=newp))
            manifest["injections"].append(dict(
                id=f"sub{ins_id}", type="substitution", score_idx=idx,
                onset=n.start, pitch_removed=n.pitch, pitch_inserted=newp,
                delta=sgn * mag))
            manifest["pairing"].append(
                dict(removed_id=f"sub{ins_id}", inserted_id=f"sub{ins_id}"))
        elif t == "omission":
            removed.append(clone(n))
            manifest["injections"].append(dict(
                id=f"om{ins_id}", type="omission", score_idx=idx,
                onset=n.start, pitch_removed=n.pitch))
        else:  # insertion near this note, note itself stays correct
            touched.discard(idx)
            mag = int(rng.choice([1, 2], p=[SUB_DELTA_P[1], SUB_DELTA_P[2]]))
            sgn = int(rng.choice([-1, 1]))
            dt = float(rng.uniform(-0.10, 0.10))
            newp = int(np.clip(n.pitch + sgn * mag, 21, 108))
            ins = clone(n, pitch=newp, start=max(0.0, n.start + dt),
                        end=max(0.05, n.start + dt) + (n.end - n.start))
            extra.append(ins)
            manifest["injections"].append(dict(
                id=f"ins{ins_id}", type="insertion", anchor_idx=idx,
                onset=ins.start, pitch_inserted=newp, dt=dt, delta=sgn * mag))
        ins_id += 1

    # negative controls: deletion + insertion >= 2 s apart, NOT a pair
    n_sub = sum(1 for i in manifest["injections"] if i["type"] == "substitution")
    n_neg = max(1, int(round(NEG_CONTROL_FRAC * n_sub)))
    untouched = [i for i in range(N) if i not in touched]
    rng.shuffle(untouched)
    made = 0
    for a_idx in untouched:
        if made >= n_neg:
            break
        cands = [j for j in untouched
                 if j != a_idx and j not in touched
                 and abs(notes[j].start - notes[a_idx].start) >= 2.0]
        if not cands:
            continue
        b_idx = cands[int(rng.integers(len(cands)))]
        a, b = notes[a_idx], notes[b_idx]
        touched.add(a_idx)
        removed.append(clone(a))
        mag = int(rng.choice([1, 2], p=[SUB_DELTA_P[1], SUB_DELTA_P[2]]))
        sgn = int(rng.choice([-1, 1]))
        newp = int(np.clip(b.pitch + sgn * mag, 21, 108))
        ins = clone(b, pitch=newp)
        extra.append(ins)
        manifest["negative_controls"].append(dict(
            id=f"neg{made}", removed_idx=a_idx, removed_onset=a.start,
            removed_pitch=a.pitch, inserted_onset=ins.start,
            inserted_pitch=newp, separation_s=abs(b.start - a.start)))
        made += 1

    correct = [clone(notes[i]) for i in range(N) if i not in touched]

    # invariants
    assert len(correct) + len(removed) == N, (piece, len(correct), len(removed), N)
    subs = {i["id"] for i in manifest["injections"] if i["type"] == "substitution"}
    edges = {e["removed_id"] for e in manifest["pairing"]}
    assert edges == subs, "pairing edges must be exactly the substitutions"

    os.makedirs(os.path.join(out, "mistake", piece), exist_ok=True)
    os.makedirs(os.path.join(out, "score", piece), exist_ok=True)
    for d in ("correct_notes", "extra_notes", "removed_notes"):
        os.makedirs(os.path.join(out, "label", d), exist_ok=True)
    os.makedirs(os.path.join(out, "manifest"), exist_ok=True)

    write_track(correct + extra, os.path.join(out, "mistake", piece, "mix.mid"))
    shutil.copy(score_mid, os.path.join(out, "score", piece, "mix.mid"))
    # score audio can be reused verbatim (score side is unchanged)
    src_wav = os.path.join(os.path.dirname(score_mid), "mix.wav")
    if os.path.exists(src_wav):
        shutil.copy(src_wav, os.path.join(out, "score", piece, "mix.wav"))
    # MAESTRO-E label layout: label/<type>/<piece>/stems_midi/mix.mid
    for typ, ns in (("correct_notes", correct), ("extra_notes", extra),
                    ("removed_notes", removed)):
        d = os.path.join(out, "label", typ, piece, "stems_midi")
        os.makedirs(d, exist_ok=True)
        write_track(ns, os.path.join(d, "mix.mid"))
    with open(os.path.join(out, "manifest", piece + ".json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return dict(piece=piece, n=N,
                sub=len(subs), om=sum(1 for i in manifest["injections"]
                                      if i["type"] == "omission"),
                ins=sum(1 for i in manifest["injections"]
                        if i["type"] == "insertion"),
                neg=made)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="run/data/MAESTRO-E/score")
    ap.add_argument("--out", default="run/data/MAESTRO-EI")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    pieces = sorted(os.listdir(a.src))
    if a.limit:
        pieces = pieces[:a.limit]
    stats, tot = [], dict(sub=0, om=0, ins=0, neg=0, n=0)
    for p in pieces:
        mid = os.path.join(a.src, p, "mix.mid")
        if not os.path.exists(mid):
            print("skip (no mix.mid):", p); continue
        st = inject_piece(mid, p, a.out)
        stats.append(st)
        for k in tot:
            tot[k] += st[k]
        print(st)
    k = tot["sub"] + tot["om"] + tot["ins"]
    summary = dict(pieces=len(stats), totals=tot,
                   achieved_mix=dict(substitution=tot["sub"] / k,
                                     insertion=tot["ins"] / k,
                                     omission=tot["om"] / k))
    with open(os.path.join(a.out, "corpus_summary.json"), "w") as fh:
        json.dump(dict(summary=summary, per_piece=stats), fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
