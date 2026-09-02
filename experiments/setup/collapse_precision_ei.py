"""Reference-side collapse precision on MAESTRO-EI label streams, tolerance-matched
against the manifest: of all pairs the collapse merges on the reference side, how many
are manifest substitutions (played pitch == pitch_inserted, |onset diff| <= tol)?"""
import json, os, sys, bisect
from collections import defaultdict
sys.path.insert(0, "repo/experiments"); sys.path.insert(0, ".")
import decoupled_scorer as ds
import pretty_midi
ROOT = "run/data/MAESTRO-EI"; EPS = 0.05; TOL = 0.005
def load(path, piece, etype):
    pm = pretty_midi.PrettyMIDI(path)
    return [ds.Event(piece=piece, etype=etype, onset_s=float(n.start), pitch_midi=int(n.pitch))
            for inst in pm.instruments for n in inst.notes]
def near(idx, key, onset, tol):
    xs = idx.get(key)
    if not xs: return False
    i = bisect.bisect_left(xs, onset - tol)
    return i < len(xs) and xs[i] <= onset + tol
pieces = [p[:-5] for p in sorted(os.listdir(os.path.join(ROOT, "manifest")))]
n_merged = n_sub_hit = n_subs = n_neg_hit = 0; mind = []
for pc in pieces:
    man = json.load(open(os.path.join(ROOT, "manifest", pc + ".json")))
    sidx = defaultdict(list); nidx = defaultdict(list)
    for i in man["injections"]:
        if i["type"] == "substitution": sidx[i["pitch_inserted"]].append(i["onset"])
    for n in man["negative_controls"]: nidx[n["inserted_pitch"]].append(n["inserted_onset"])
    for d in (sidx, nidx):
        for k in d: d[k].sort()
    n_subs += sum(len(v) for v in sidx.values())
    ex = load(os.path.join(ROOT, "label", "extra_notes", pc, "stems_midi", "mix.mid"), pc, "extra")
    rm = load(os.path.join(ROOT, "label", "removed_notes", pc, "stems_midi", "mix.mid"), pc, "missed")
    out = ds.collapse_wrong(ex + rm, epsilon=EPS, mode="strict")
    for e in out:
        if e.etype != "wrong": continue
        n_merged += 1
        if near(sidx, e.pitch_midi, e.onset_s, TOL): n_sub_hit += 1
        if near(nidx, e.pitch_midi, e.onset_s, TOL): n_neg_hit += 1
        xs = sidx.get(e.pitch_midi)
        if xs:
            j = bisect.bisect_left(xs, e.onset_s); c = [abs(xs[k]-e.onset_s) for k in (j-1, j) if 0 <= k < len(xs)]
            if c: mind.append(min(c))
mind.sort()
res = dict(_provenance=dict(run="collapse_precision_ei.py on gilbreth 2026-09-01", corpus=ROOT, epsilon_s=EPS, mode="strict", match_tol_s=TOL),
           n_reference_merges=n_merged, n_manifest_substitutions=n_subs, merges_that_are_substitutions=n_sub_hit,
           precision=n_sub_hit / n_merged, recall=n_sub_hit / n_subs, decoys_merged=n_neg_hit,
           onset_gap_quantiles_s={"p50": mind[len(mind)//2], "p90": mind[int(0.9*len(mind))], "p99": mind[int(0.99*len(mind))], "max": mind[-1]})
json.dump(res, open("run/results/maestro_ei_collapse_precision.json", "w"), indent=1)
print(json.dumps(res, indent=1))
