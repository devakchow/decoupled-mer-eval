"""MAESTRO-EI manifest-grounded validation, collapse ACTIVE, no model.

1. RECOVERY ON REAL MUSIC: degrade the labels by flipping a known fraction q
   of non-substitution label events to the sibling class. Every flip is one
   localized-but-misclassified event by construction; nothing else is. The
   measured off-diagonal must equal the planted flip count (collapse active,
   dense real piano).
2. NEGATIVE CONTROLS: the corpus plants unpaired deletion+insertion pairs
   >= 2 s apart; the collapse must merge NONE of them.
3. ADJUDICATION TRUTH: every substitution's merged wrong-event carries a
   missed-half naming its removed score note at the same onset; under the
   letter's 50 ms exact-pitch anchor its genuine rate must be ~1.
"""
import json, os, sys, bisect
from collections import defaultdict
import numpy as np
sys.path.insert(0, 'repo/experiments')
sys.path.insert(0, '.')
import decoupled_scorer as ds
import collapse_validation as CV
import pretty_midi

ROOT = 'run/data/MAESTRO-EI'
TAU, EPS = 0.050, 0.050
Q, SEED = 0.10, 20260831
Event = ds.Event


def load_track(path, piece, etype):
    pm = pretty_midi.PrettyMIDI(path)
    return [Event(piece=piece, etype=etype, onset_s=float(n.start),
                  pitch_midi=int(n.pitch))
            for inst in pm.instruments for n in inst.notes]


def near_onset(xs, onset, tol):
    # Label-MIDI onsets are tick-quantized (median 0.6 ms from the manifest
    # float), so membership must be tolerance-based, never exact.
    i = bisect.bisect_left(xs, onset - tol)
    return i < len(xs) and xs[i] <= onset + tol


def anchored(idx, piece, pitch, onset, tol=TAU):
    xs = idx.get((piece, pitch))
    if not xs:
        return False
    i = bisect.bisect_left(xs, onset - tol)
    return i < len(xs) and xs[i] <= onset + tol


def main():
    pieces = [p[:-5] for p in sorted(os.listdir(os.path.join(ROOT, 'manifest')))]
    rng = np.random.default_rng(SEED)
    refs, preds = [], []
    planted_flips = flippable = 0
    neg_controls = []
    removed_idx = defaultdict(list)          # (piece,pitch) -> removed onsets

    for pc in pieces:
        man = json.load(open(os.path.join(ROOT, 'manifest', pc + '.json')))
        ex = load_track(os.path.join(ROOT, 'label', 'extra_notes', pc, 'stems_midi', 'mix.mid'), pc, 'extra')
        rm = load_track(os.path.join(ROOT, 'label', 'removed_notes', pc, 'stems_midi', 'mix.mid'), pc, 'missed')
        refs.extend(ex); refs.extend(rm)
        for e in rm:
            removed_idx[(pc, e.pitch_midi)].append(e.onset_s)
        sub_onsets = sorted(i['onset'] for i in man['injections']
                            if i['type'] == 'substitution')
        for e in ex + rm:
            if not near_onset(sub_onsets, e.onset_s, 0.005):
                flippable += 1
                if rng.random() < Q:
                    planted_flips += 1
                    sib = 'extra' if e.etype == 'missed' else 'missed'
                    preds.append(Event(piece=pc, etype=sib,
                                       onset_s=e.onset_s, pitch_midi=e.pitch_midi))
                    continue
            preds.append(e)
        neg_controls.extend((pc, n) for n in man['negative_controls'])
    for k in removed_idx:
        removed_idx[k].sort()

    pc_raw, prov = CV.collapse_with_prov(preds)
    rc_raw, _ = CV.collapse_with_prov(refs)
    wrong_slice = pc_raw[len(pc_raw) - len(prov):]
    pmap = {}
    for i, e in enumerate(wrong_slice):
        m_pitch, x_pitch, m_on, x_on, piece = prov[i]
        pmap[id(e)] = (m_pitch, m_on)
    pcn = ds._canonical(pc_raw); rcn = ds._canonical(rc_raw)
    pe = [e for e in pcn if e.etype in ('missed', 'extra', 'wrong')]
    re_ = [e for e in rcn if e.etype in ('missed', 'extra', 'wrong')]
    pairs = ds.match_events(pe, re_, TAU, require_pitch=False)
    conf = defaultdict(int)
    gen_hit = gen_tot = 0
    for pi, ri in pairs:
        p, r = pe[pi], re_[ri]
        conf['%s->%s' % (r.etype, p.etype)] += 1
        if p.etype == 'wrong' and r.etype == 'wrong' and id(p) in pmap:
            gen_tot += 1
            m_pitch, m_on = pmap[id(p)]
            if anchored(removed_idx, p.piece, m_pitch, m_on):
                gen_hit += 1
    M = len(pairs)
    off = sum(v for k, v in conf.items() if k.split('->')[0] != k.split('->')[1])

    # fate accounting: every planted flip must be attributed
    flip_keys = set()
    rng2 = np.random.default_rng(SEED)
    for pc2 in pieces:
        man2 = json.load(open(os.path.join(ROOT, 'manifest', pc2 + '.json')))
        ex2 = load_track(os.path.join(ROOT, 'label', 'extra_notes', pc2, 'stems_midi', 'mix.mid'), pc2, 'extra')
        rm2 = load_track(os.path.join(ROOT, 'label', 'removed_notes', pc2, 'stems_midi', 'mix.mid'), pc2, 'missed')
        sub_on2 = sorted(i['onset'] for i in man2['injections']
                         if i['type'] == 'substitution')
        for e in ex2 + rm2:
            if not near_onset(sub_on2, e.onset_s, 0.005):
                if rng2.random() < Q:
                    sib = 'extra' if e.etype == 'missed' else 'missed'
                    flip_keys.add((pc2, round(e.onset_s, 6), e.pitch_midi, sib))
    assert len(flip_keys) == planted_flips, (len(flip_keys), planted_flips)
    pred_pos = {}
    for i2, e in enumerate(pe):
        pred_pos[(e.piece, round(e.onset_s, 6), e.pitch_midi, e.etype)] = i2
    matched_p = {pi2: ri2 for pi2, ri2 in pairs}
    fate = dict(matched_offdiag=0, matched_diag=0, merged=0, unmatched=0)
    for key in flip_keys:
        i2 = pred_pos.get(key)
        if i2 is None:
            fate['merged'] += 1            # consumed by the collapse
        elif i2 in matched_p:
            r2 = re_[matched_p[i2]]
            if r2.etype != pe[i2].etype:
                fate['matched_offdiag'] += 1
            else:
                fate['matched_diag'] += 1
        else:
            fate['unmatched'] += 1
    assert sum(fate.values()) == planted_flips

    paired_neg = 0
    wrong_by_piece = defaultdict(list)
    for e in pcn:
        if e.etype == 'wrong':
            wrong_by_piece[e.piece].append((e.onset_s, e.pitch_midi))
    for pc, n in neg_controls:
        for on, pit in wrong_by_piece.get(pc, ()):
            if abs(on - n['inserted_onset']) < 0.005 and pit == n['inserted_pitch']:
                paired_neg += 1
                break

    out = dict(
        _provenance=dict(run='validate_maestro_ei.py on gilbreth 2026-09-01 (tolerance-matched)',
                         corpus=ROOT, q_planted=Q, seed=SEED,
                         degraded_system='labels with q of non-substitution '
                                         'events class-flipped'),
        n_ref_events=len(refs), n_pred_events=len(preds),
        n_matched=M, off_diagonal=off,
        planted_flips=planted_flips, flippable=flippable,
        recovery=dict(measured_off_diagonal=off, planted=planted_flips,
                      flip_fate=fate),
        negative_controls=dict(total=len(neg_controls),
                               merged_by_collapse=paired_neg),
        adjudication=dict(wrong_wrong_merged=gen_tot, genuine=gen_hit,
                          genuine_rate=(gen_hit / gen_tot) if gen_tot else None))
    json.dump(out, open('validate_ei.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != '_provenance'}, indent=1))


if __name__ == '__main__':
    main()
