#!/usr/bin/env python3
"""paired_analysis.py — per-piece PAIRED comparison of two systems.

The headline result currently compares two micro-pooled scalars with
independent bootstrap CIs. That is weaker than the data supports: both systems
are evaluated on the SAME 177 pieces, so the comparison is naturally paired.
Paired analysis removes between-piece variance (which is large and is common to
both systems) and is the statistically correct treatment.

Reports, for HM and for Loc-F:
  * per-piece differences, with mean / median / IQR;
  * a two-sided Wilcoxon signed-rank test (distribution-free; the per-piece
    differences are not assumed normal);
  * a paired bootstrap CI on the mean difference (resampling PIECES, matching
    the resampling unit used in the main scorer);
  * the sign split (how many pieces favour each system) — the plainest possible
    statement of consistency, and robust to outliers;
  * effect size (Cliff's delta, non-parametric).

Honesty notes:
  * Pieces where either system has undefined HM are excluded and counted.
  * Micro-pooled HM (the headline) and the mean of per-piece HM are DIFFERENT
    estimands; both are reported so the paper never conflates them.

Consumes the per-piece table from nonidentifiability_empirical.py's JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from typing import Dict, List, Sequence, Tuple

import numpy as np


def wilcoxon_signed_rank(d: Sequence[float]) -> Tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank test. Returns (W, p_normal_approx, n).

    Zero differences are dropped (Wilcoxon convention). Uses the normal
    approximation with continuity correction and tie correction, which is
    appropriate here (n is large, ~177).
    """
    nz = [x for x in d if x != 0.0]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0, 0
    mag = sorted((abs(x), i) for i, x in enumerate(nz))
    ranks = [0.0] * n
    i = 0
    tie_groups: List[int] = []
    while i < n:
        j = i
        while j + 1 < n and mag[j + 1][0] == mag[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[mag[k][1]] = avg
        tie_groups.append(j - i + 1)
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, nz) if x > 0)
    w_minus = sum(r for r, x in zip(ranks, nz) if x < 0)
    W = min(w_plus, w_minus)
    mu = n * (n + 1) / 4.0
    tie_term = sum(t ** 3 - t for t in tie_groups)
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0)
    if sigma == 0:
        return W, 1.0, n
    z = (W - mu + 0.5) / sigma
    p = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
    return W, min(1.0, p), n


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1]."""
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    n = len(a) * len(b)
    return (gt - lt) / n if n else 0.0


def paired_bootstrap(diffs: Sequence[float], n_boot: int, seed: int
                     ) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = arr[rng.integers(0, n, size=n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True,
                    help="JSON from nonidentifiability_empirical.py")
    ap.add_argument("--system-a", required=True)
    ap.add_argument("--system-b", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as fh:
        tables: Dict[str, List[dict]] = json.load(fh)["per_piece"]
    A = {r["piece"]: r for r in tables[args.system_a]}
    B = {r["piece"]: r for r in tables[args.system_b]}
    common = sorted(set(A) & set(B))

    excluded = [p for p in common
                if A[p]["hm"] is None or B[p]["hm"] is None]
    used = [p for p in common if p not in set(excluded)]

    out: Dict[str, object] = dict(
        system_a=args.system_a, system_b=args.system_b,
        n_common_pieces=len(common), n_excluded_undefined_hm=len(excluded),
        n_used=len(used), n_boot=args.n_boot, seed=args.seed,
    )

    print(f"paired analysis: {args.system_a} vs {args.system_b}")
    print(f"  common pieces {len(common)}, used {len(used)}, "
          f"excluded (undefined HM) {len(excluded)}\n")

    for metric in ("hm", "loc_f1"):
        a = [A[p][metric] for p in used]
        b = [B[p][metric] for p in used]
        d = [x - y for x, y in zip(a, b)]          # a minus b
        W, p, n_nz = wilcoxon_signed_rank(d)
        lo, hi = paired_bootstrap(d, args.n_boot, args.seed)
        n_pos = sum(1 for x in d if x > 0)
        n_neg = sum(1 for x in d if x < 0)
        delta = cliffs_delta(a, b)
        q = statistics.quantiles(d, n=4) if len(d) >= 4 else [float("nan")] * 3

        res = dict(
            mean_a=statistics.mean(a), mean_b=statistics.mean(b),
            median_a=statistics.median(a), median_b=statistics.median(b),
            mean_diff=statistics.mean(d), median_diff=statistics.median(d),
            iqr_diff=[q[0], q[2]],
            paired_bootstrap_ci95=[lo, hi],
            wilcoxon_W=W, wilcoxon_p=p, wilcoxon_n_nonzero=n_nz,
            n_pieces_favouring_a=n_pos, n_pieces_favouring_b=n_neg,
            cliffs_delta=delta,
        )
        out[metric] = res

        print(f"  [{metric}]")
        print(f"    per-piece mean   : {args.system_a}={res['mean_a']:.4f}  "
              f"{args.system_b}={res['mean_b']:.4f}")
        print(f"    per-piece median : {args.system_a}={res['median_a']:.4f}  "
              f"{args.system_b}={res['median_b']:.4f}")
        print(f"    mean difference  : {res['mean_diff']:+.4f}  "
              f"(paired bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}])")
        print(f"    median difference: {res['median_diff']:+.4f}   "
              f"IQR [{q[0]:+.4f}, {q[2]:+.4f}]")
        print(f"    Wilcoxon signed-rank: W={W:.1f}, n={n_nz}, "
              f"p={'<1e-12' if p < 1e-12 else f'{p:.3g}'}")
        print(f"    pieces favouring {args.system_a}: {n_pos} / "
              f"{args.system_b}: {n_neg}")
        print(f"    Cliff's delta    : {delta:+.4f}")
        print()

    dest = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.inp)), "paired_analysis.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"  wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
