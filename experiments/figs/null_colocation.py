"""Null-model test: is the matched off-diagonal (HM numerator) real temporal
alignment between a system's errors and the reference errors, or coincidental
co-location in dense passages?

Null: within each piece, circularly shift ALL predicted ERROR onsets by a random
offset (preserving each system's own error time-structure and count, destroying
its alignment to the reference). Recompute the onset matching and the matched
off-diagonal (localized-but-misclassified) and matched total. If the OBSERVED
matched total / off-diagonal greatly exceed the null distribution, the
co-locations are genuine detections, not chance -- which is exactly what HM and
the Fig.1 worked case need. If observed ~ null, HM is coincidence noise (fatal).

Deterministic: seeded per (piece, permutation) via an index-derived offset grid,
no Math.random. Run on cluster (preds + gt + scorer present).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import decoupled_scorer as ds  # noqa: E402


def off_and_total(pred_err, ref_err, tau):
    m = ds.match_events(pred_err, ref_err, tau, require_pitch=False)
    off = sum(1 for pi, ri in m if pred_err[pi].etype != ref_err[ri].etype)
    return off, len(m)


def shift_onsets(events, delta, span_lo, span):
    """Circularly shift error onsets within [span_lo, span_lo+span)."""
    out = []
    for e in events:
        no = span_lo + ((e.onset_s - span_lo + delta) % span)
        out.append(ds.Event(no, e.pitch_midi, e.etype, e.piece))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_meta", required=True)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--collapse", default="strict")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--out", default="null_colocation.json")
    args = ap.parse_args()

    preds = ds.load_pred_events(args.pred_dir)
    refs = ds.load_ref_events(args.gt_meta)
    P, R = defaultdict(list), defaultdict(list)
    for e in preds:
        P[e.piece].append(e)
    for e in refs:
        R[e.piece].append(e)

    obs_off = obs_tot = 0
    null_off = [0] * args.n_perm
    null_tot = [0] * args.n_perm
    pieces = sorted(set(P) & set(R))
    for piece in pieces:
        predc = ds._canonical(ds.collapse_wrong(P[piece], epsilon=args.eps,
                                                mode=args.collapse))
        refc = ds._canonical(ds.collapse_wrong(R[piece], epsilon=args.eps,
                                               mode=args.collapse))
        pe = [e for e in predc if e.etype in ds.ERROR_TYPES]
        re = [e for e in refc if e.etype in ds.ERROR_TYPES]
        if not pe or not re:
            continue
        o, tt = off_and_total(pe, re, args.tau)
        obs_off += o
        obs_tot += tt
        # span for circular shift: full error time-extent of the piece
        onsets = [e.onset_s for e in pe + re]
        span_lo = min(onsets)
        span = max(onsets) - span_lo
        if span <= 0:
            for k in range(args.n_perm):
                null_off[k] += o
                null_tot[k] += tt
            continue
        # deterministic offset grid: k-th permutation shifts by a distinct,
        # piece-dependent fraction of the span (no RNG -> resume-safe)
        base = (_stable_hash(piece) % 997) / 997.0
        for k in range(args.n_perm):
            frac = (base + (k + 1) / (args.n_perm + 1)) % 1.0
            delta = 0.05 * span + frac * 0.9 * span   # avoid ~0 and ~span
            shifted = shift_onsets(pe, delta, span_lo, span)
            shifted.sort(key=lambda e: e.onset_s)
            no, nt = off_and_total(shifted, re, args.tau)
            null_off[k] += no
            null_tot[k] += nt

    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        mean = sum(xs) / n
        return {"mean": mean, "min": xs[0], "max": xs[-1],
                "p50": xs[n // 2], "p95": xs[int(0.95 * n)]}

    off_s = stats(null_off)
    tot_s = stats(null_tot)
    # p-value: fraction of permutations whose off-diag >= observed
    p_off = sum(1 for x in null_off if x >= obs_off) / len(null_off)
    p_tot = sum(1 for x in null_tot if x >= obs_tot) / len(null_tot)
    result = {
        "n_pieces": len(pieces), "n_perm": args.n_perm, "tau": args.tau,
        "observed_off_diagonal": obs_off, "observed_matched_total": obs_tot,
        "null_off_diagonal": off_s, "null_matched_total": tot_s,
        "enrichment_off": obs_off / off_s["mean"] if off_s["mean"] else None,
        "enrichment_total": obs_tot / tot_s["mean"] if tot_s["mean"] else None,
        "p_off_ge_observed": p_off, "p_total_ge_observed": p_tot,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
