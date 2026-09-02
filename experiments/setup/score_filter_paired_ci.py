"""Paired per-piece bootstrap: score-consistency filter vs the unfiltered claims.

For each configuration, per-piece shipped-mode (published-protocol) counts and
decoupled counts at tau = 50 ms are computed twice, on the raw predicted
claims and after decoupled_scorer.score_consistency_filter (50 ms anchor).
Pooled statistics (published missed/extra F1 and their mean, raw HM, F) are
resampled over pieces with the SAME piece draws for both arms, so every
difference carries a paired percentile 95% interval; per-piece wins on the
published missed F1 are counted too. 10^4 resamples, seed 20260718.
RUNS ON GILBRETH from /scratch/gilbreth/dcharapa/mer (envs/polytune/bin/python).
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "repo/experiments")
import decoupled_scorer as ds  # noqa: E402

CFGS = {
    "A_polytune_maestro": "run/gt_meta_maestro.json",
    "B_laddersym_maestro_unprompted": "run/gt_meta_maestro.json",
    "B_laddersym_maestro_prompted": "run/gt_meta_maestro.json",
    "A_polytune_maestro_ei": "run/gt_meta_maestro_ei.json",
    "B_laddersym_maestro_ei_unprompted": "run/gt_meta_maestro_ei.json",
    "B_laddersym_maestro_ei_prompted": "run/gt_meta_maestro_ei.json",
}
TAU, EPS, ANCHOR = 0.050, 0.050, 0.050
N_BOOT, SEED = 10_000, 20260718
FIELDS = ("m_tp", "m_fp", "m_fn", "e_tp", "e_fp", "e_fn",
          "l_tp", "l_fp", "l_fn", "n_loc", "off")


def by_piece(events):
    out = defaultdict(list)
    for e in events:
        out[e.piece].append(e)
    return out


def piece_rows(pred, ref):
    """One count vector per piece (ordered by piece id)."""
    pp, rp = by_piece(pred), by_piece(ref)
    rows = []
    for pc in sorted(set(pp) | set(rp)):
        p, r = pp.get(pc, []), rp.get(pc, [])
        sh = ds.shipped_scores(p, r, tau=TAU)
        dec = ds.decoupled_scores(p, r, tau=TAU, epsilon=EPS, collapse="strict")
        assert dec.mass_conserved()
        rows.append([sh["missed"]["tp"], sh["missed"]["fp"], sh["missed"]["fn"],
                     sh["extra"]["tp"], sh["extra"]["fp"], sh["extra"]["fn"],
                     dec.localization["tp"], dec.localization["fp"],
                     dec.localization["fn"], dec.n_localized, dec.off_diagonal()])
    return np.array(rows, dtype=float)


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0)


def pooled(c):
    """Pooled statistics from summed count rows (works on batched sums)."""
    m = f1(c[..., 0], c[..., 1], c[..., 2])
    e = f1(c[..., 3], c[..., 4], c[..., 5])
    loc = f1(c[..., 6], c[..., 7], c[..., 8])
    hm = c[..., 10] / np.where(c[..., 9] > 0, c[..., 9], 1)
    return dict(missed_f1=m, extra_f1=e, mean_error_f1=(m + e) / 2, loc_f=loc, hm=hm)


def main():
    rng = np.random.default_rng(SEED)
    out = {"_protocol": __doc__.strip(), "tau_s": TAU, "epsilon_s": EPS,
           "anchor_s": ANCHOR, "n_boot": N_BOOT, "seed": SEED}
    for name, gt in CFGS.items():
        pred = ds.load_pred_events("run/preds/" + name)
        ref = ds.load_ref_events(gt)
        filt, dropped = ds.score_consistency_filter(pred, ref, ANCHOR)
        base_rows, filt_rows = piece_rows(pred, ref), piece_rows(filt, ref)
        assert base_rows.shape == filt_rows.shape
        n = base_rows.shape[0]
        idx = rng.integers(0, n, size=(N_BOOT, n))
        b_sum, f_sum = base_rows[idx].sum(axis=1), filt_rows[idx].sum(axis=1)
        pb, pf = pooled(b_sum), pooled(f_sum)
        point_b, point_f = pooled(base_rows.sum(axis=0)), pooled(filt_rows.sum(axis=0))
        stats = {}
        for k in pb:
            diff = pf[k] - pb[k]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            stats[k] = dict(base=float(point_b[k]), filtered=float(point_f[k]),
                            diff=float(point_f[k] - point_b[k]),
                            diff_ci95=[float(lo), float(hi)],
                            filtered_ci95=[float(x) for x in np.percentile(pf[k], [2.5, 97.5])])
        pm_b = f1(base_rows[:, 0], base_rows[:, 1], base_rows[:, 2])
        pm_f = f1(filt_rows[:, 0], filt_rows[:, 1], filt_rows[:, 2])
        ph_b = base_rows[:, 10] / np.where(base_rows[:, 9] > 0, base_rows[:, 9], 1)
        ph_f = filt_rows[:, 10] / np.where(filt_rows[:, 9] > 0, filt_rows[:, 9], 1)
        out[name] = dict(
            n_pieces=int(n),
            n_missed_claims=int(sum(1 for e in pred if e.etype == "missed")),
            n_dropped=int(dropped),
            stats=stats,
            wins_missed_f1=dict(filtered=int((pm_f > pm_b).sum()),
                                base=int((pm_f < pm_b).sum()),
                                tie=int((pm_f == pm_b).sum())),
            wins_hm_lower=dict(filtered=int((ph_f < ph_b).sum()),
                               base=int((ph_f > ph_b).sum()),
                               tie=int((ph_f == ph_b).sum())),
        )
        print(name, "dropped %d/%d" % (dropped, out[name]["n_missed_claims"]),
              {k: (round(v["base"], 4), round(v["filtered"], 4),
                   [round(x, 4) for x in v["diff_ci95"]]) for k, v in stats.items()},
              flush=True)
    json.dump(out, open("run/results/score_filter_paired_ci.json", "w"), indent=1)
    print("wrote run/results/score_filter_paired_ci.json")


if __name__ == "__main__":
    main()
