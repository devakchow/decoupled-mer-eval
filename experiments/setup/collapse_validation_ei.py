"""MAESTRO-EI adjudication: collapse_validation.py retargeted at the EI corpus.

Same provenance-clone machinery and gates; only the corpus, gt-meta, and
config names differ. Additionally adjudicates each merged wrong-event against
the MANIFEST pairing edges (exact ground truth), not just the score tracks.
"""
import json, sys, os, bisect
from collections import defaultdict
sys.path.insert(0, 'repo/experiments')
sys.path.insert(0, '.')
import decoupled_scorer as ds
import collapse_validation as CV

TAU, EPS = 0.050, 0.050
CFGS = ['A_polytune_maestro_ei', 'B_laddersym_maestro_ei_unprompted',
        'B_laddersym_maestro_ei_prompted']
GTM = 'run/gt_meta_maestro_ei.json'
MANIFEST_DIR = 'run/data/MAESTRO-EI/manifest'
ERR = ('missed', 'extra', 'wrong')


def load_manifest_edges():
    """(piece, pitch) -> sorted onsets of manifest-paired removed notes."""
    idx = defaultdict(list)
    for f in sorted(os.listdir(MANIFEST_DIR)):
        man = json.load(open(os.path.join(MANIFEST_DIR, f)))
        pc = man['piece']
        subs = {i['id']: i for i in man['injections'] if i['type'] == 'substitution'}
        for e in man['pairing']:
            i = subs[e['removed_id']]
            idx[(pc, i['pitch_removed'])].append(i['onset'])
    for k in idx:
        idx[k].sort()
    return idx


def anchored(idx, piece, pitch, onset, tol=TAU):
    xs = idx.get((piece, pitch))
    if not xs:
        return False
    i = bisect.bisect_left(xs, onset - tol)
    return i < len(xs) and xs[i] <= onset + tol


def analyse(name, man_idx):
    pred = ds.load_pred_events('run/preds/' + name)
    ref = ds.load_ref_events(GTM)
    pc_raw, prov_p = CV.collapse_with_prov(pred)
    rc_raw, _ = CV.collapse_with_prov(ref)
    n_wrong = sum(1 for e in pc_raw if e.etype == 'wrong')
    assert n_wrong == len(prov_p)
    wrong_slice = pc_raw[len(pc_raw) - len(prov_p):]
    pmap = {}
    for i, e in enumerate(wrong_slice):
        m_pitch, x_pitch, m_on, x_on, piece = prov_p[i]
        assert e.pitch_midi == x_pitch and e.onset_s == x_on and e.piece == piece
        pmap[id(e)] = (m_pitch, m_on)
    pcn = ds._canonical(pc_raw); rcn = ds._canonical(rc_raw)
    pe = [e for e in pcn if e.etype in ERR]
    re_ = [e for e in rcn if e.etype in ERR]
    pairs = ds.match_events(pe, re_, TAU, require_pitch=False)
    import glob
    cands = sorted(glob.glob('run/results*/%s_shipped.json' % name),
                   key=lambda p: ('v110' not in p, p))
    conf = defaultdict(int)
    for pi, ri in pairs:
        conf['%s->%s' % (re_[ri].etype, pe[pi].etype)] += 1
    gate = 'UNGATED (no shipped artifact found)'
    if cands:
        exp = json.load(open(cands[0]))
        t = [x for x in exp['decoupled']['per_tau'] if x['tau_ms'] == 50][0]
        assert dict(conf) == dict(t['confusion_sparse']), 'GATE FAILED ' + name
        assert len(pairs) == t['n_localized']
        gate = 'confusion identical to scorer artifact: ' + cands[0]
    out = {'_gate': gate, '_n_merges_pred': len(prov_p)}
    for cell in ('extra->wrong', 'wrong->wrong', 'missed->wrong'):
        rt, pt = cell.split('->')
        n = g = 0
        for pi, ri in pairs:
            if re_[ri].etype != rt or pe[pi].etype != pt:
                continue
            n += 1
            m_pitch, m_on = pmap[id(pe[pi])]
            if anchored(man_idx, pe[pi].piece, m_pitch, m_on):
                g += 1
        out[cell] = dict(n=n, manifest_genuine=g,
                         manifest_genuine_rate=(g / n) if n else None)
    return out


if __name__ == '__main__':
    man_idx = load_manifest_edges()
    res = {c: analyse(c, man_idx) for c in CFGS}
    json.dump(res, open('collapse_validation_ei.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
