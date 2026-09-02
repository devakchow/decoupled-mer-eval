"""How much of U depends on the EXACT-pitch anchor? Re-adjudicate merged wrong events
with (a) exact pitch (letter), (b) +/-1 semitone, (c) octave-equivalent pitch, all within
the 50 ms anchor; also the equal-pitch share of DIAGONAL matched pairs (pitch-blind
localization credits pitch-mismatched same-class pairs).
RUNS ON GILBRETH from /scratch/gilbreth/dcharapa/mer (anaenv/bin/python)."""
import json, sys, bisect
from collections import defaultdict
sys.path.insert(0, "repo/experiments"); sys.path.insert(0, ".")
import collapse_validation as CV
import decoupled_scorer as ds

ref = ds.load_ref_events('run/gt_meta_maestro.json')
idx_played = CV.index_by_type(ref, 'correct')
idx_unplay = CV.index_by_type(ref, 'missed')
rc_raw, _ = CV.collapse_with_prov(ref)
re_ = [e for e in ds._canonical(rc_raw) if e.etype in CV.ERR]

def anchored_any(idx, piece, pitches, onset):
    return any(CV.anchored(idx, piece, p, onset) for p in pitches)

out = {"_protocol": "re-adjudication of merged predicted-wrong events under pitch-tolerant anchors "
                    "(50 ms window unchanged); diagonal equal-pitch share of matched pairs"}
for name in CV.CFGS:
    pred = ds.load_pred_events('run/preds/' + name)
    pc_raw, prov_p = CV.collapse_with_prov(pred)
    wrong_slice = pc_raw[len(pc_raw) - len(prov_p):]
    pmap = {id(e): (prov_p[i][0], prov_p[i][2]) for i, e in enumerate(wrong_slice)}
    pe = [e for e in ds._canonical(pc_raw) if e.etype in CV.ERR]
    pairs = ds.match_events(pe, re_, CV.TAU, require_pitch=False)
    K = 0; U = {"exact": 0, "semitone": 0, "octave": 0}
    diag_n = diag_eq = 0
    for pi, ri in pairs:
        p, r = pe[pi], re_[ri]
        if p.etype == r.etype:
            diag_n += 1; diag_eq += int(p.pitch_midi == r.pitch_midi)
        if p.etype == 'wrong' and id(p) in pmap:
            K += 1
            m_pitch, m_on = pmap[id(p)]
            variants = {"exact": [m_pitch],
                        "semitone": [m_pitch - 1, m_pitch, m_pitch + 1],
                        "octave": [q for q in range(0, 128) if (q - m_pitch) % 12 == 0]}
            for k, ps in variants.items():
                if not anchored_any(idx_unplay, p.piece, ps, m_on) and \
                   not anchored_any(idx_played, p.piece, ps, m_on):
                    U[k] += 1
    M = len(pairs)
    out[name] = {"M": M, "K": K, "U": U,
                 "unfounded_share": {k: v / M for k, v in U.items()},
                 "diag_pairs": diag_n, "diag_equal_pitch_frac": diag_eq / diag_n}
    print(name, {k: round(v / M, 4) for k, v in U.items()}, "diag eq-pitch %.4f" % (diag_eq / diag_n))
json.dump(out, open('run/results/anchor_pitch_sensitivity.json', 'w'), indent=1)
print("wrote")
