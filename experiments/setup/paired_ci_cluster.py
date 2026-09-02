"""Per-piece adjudication counts for all three configurations and PAIRED
bootstrap differences between adjacent systems (same pieces resampled).

RUNS ON GILBRETH from /scratch/gilbreth/dcharapa/mer (anaenv/bin/python).
Extends uk_ci_cluster.py: stores per-piece [M, off, off_unmerged, K, G, A, U,
offG] so that raw HM, HM_G, |U|/|M|, |U|/|K| get paired-difference intervals
rather than marginal-overlap comparisons.
"""
import json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, 'repo/experiments')
import collapse_validation as CV
import decoupled_scorer as ds

RES, SEED = 10000, 20260718
ref = ds.load_ref_events('run/gt_meta_maestro.json')
idx_played = CV.index_by_type(ref, 'correct')
idx_unplay = CV.index_by_type(ref, 'missed')
rc_raw, _ = CV.collapse_with_prov(ref)
rc = ds._canonical(rc_raw)
re_ = [e for e in rc if e.etype in CV.ERR]

per_sys = {}
for name in CV.CFGS:
    pred = ds.load_pred_events('run/preds/' + name)
    pc_raw, prov_p = CV.collapse_with_prov(pred)
    wrong_slice = pc_raw[len(pc_raw) - len(prov_p):]
    pmap = {id(e): (prov_p[i][0], prov_p[i][2]) for i, e in enumerate(wrong_slice)}
    pc = ds._canonical(pc_raw)
    pe = [e for e in pc if e.etype in CV.ERR]
    pairs = ds.match_events(pe, re_, CV.TAU, require_pitch=False)
    per = defaultdict(lambda: dict(M=0, off=0, off_unm=0, K=0, G=0, A=0, U=0, offG=0))
    for pi, ri in pairs:
        p, r = pe[pi], re_[ri]
        d = per[p.piece]; d['M'] += 1
        offd = p.etype != r.etype
        if offd: d['off'] += 1
        if p.etype == 'wrong' and id(p) in pmap:
            d['K'] += 1
            m_pitch, m_on = pmap[id(p)]
            if CV.anchored(idx_unplay, p.piece, m_pitch, m_on):
                d['G'] += 1
                if offd: d['offG'] += 1
            elif CV.anchored(idx_played, p.piece, m_pitch, m_on):
                d['A'] += 1
            else:
                d['U'] += 1
        elif offd:
            d['off_unm'] += 1
    per_sys[name] = dict(per)

pieces = sorted(set.intersection(*[set(v) for v in per_sys.values()]))
keys = ['M', 'off', 'off_unm', 'K', 'G', 'A', 'U', 'offG']
arr = {n: np.array([[per_sys[n][p][k] for k in keys] for p in pieces], float) for n in CV.CFGS}
ki = {k: i for i, k in enumerate(keys)}

def metrics(S):
    M, off, off_unm, K, G, A, U, offG = (S[:, ki[k]] for k in keys)
    return dict(raw_hm=off / M, hm_g=(off_unm + offG) / (M - A - U),
                unfounded=U / M, u_over_k=U / K)

rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(pieces), size=(RES, len(pieces)))
boot = {n: metrics(arr[n][idx].sum(axis=1)) for n in CV.CFGS}
point = {n: {k: float(v) for k, v in metrics(arr[n].sum(axis=0, keepdims=True)).items()} for n in CV.CFGS}
out = {"_protocol": "paired per-piece cluster bootstrap, %d resamples, seed %d, "
                    "same piece indices for every system" % (RES, SEED),
       "n_pieces": len(pieces), "point": point, "paired_diff_ci95": {}, "per_piece_keys": keys,
       "per_piece": {n: {p: [int(x) for x in arr[n][i]] for i, p in enumerate(pieces)} for n in CV.CFGS}}
names = list(CV.CFGS)
for a, b in [(names[0], names[1]), (names[1], names[2]), (names[0], names[2])]:
    out["paired_diff_ci95"]["%s - %s" % (a, b)] = {
        k: [float(np.percentile(boot[a][k] - boot[b][k], 2.5)),
            float(np.percentile(boot[a][k] - boot[b][k], 97.5)),
            float(point[a][k] - point[b][k])] for k in boot[a]}
json.dump(out, open('run/results/paired_ci.json', 'w'), indent=1)
for pair, d in out["paired_diff_ci95"].items():
    print(pair)
    for k, (lo, hi, pt) in d.items():
        print("   %-10s diff %.4f  95%% [%.4f, %.4f]  %s" % (k, pt, lo, hi, "excludes 0" if lo > 0 or hi < 0 else "includes 0"))
