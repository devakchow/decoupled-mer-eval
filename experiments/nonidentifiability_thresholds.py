#!/usr/bin/env python3
"""nonidentifiability_thresholds.py — honest characterisation of how often the
Proposition-1 ambiguity is observable in real predictions.

The single-threshold search reports "N matches at (eps_shipped, min_hm_gap)",
which invites cherry-picking. This sweeps both thresholds and reports the whole
surface, so the reader sees the trade-off rather than one flattering cut. It
also reports the *converse* statistic, which is the one that actually matters
for the paper's argument:

    Among pairs the shipped protocol cannot separate (shipped L-inf <= eps),
    what is the DISTRIBUTION of |HM gap|?

If the shipped statistic were an adequate summary, near-identical shipped
vectors would imply near-identical decoupled behaviour, and that distribution
would concentrate near zero. Its spread is the empirical content of the claim.

Consumes the per-piece table already computed by
nonidentifiability_empirical.py (its JSON output), so no rescoring is needed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
from typing import Dict, List, Sequence, Tuple

SHIPPED_TRACKS = ("missed", "extra", "correct")


def shipped_vec(row: dict) -> Tuple[float, ...]:
    return tuple(row[f"ship_{k}_f1"] for k in SHIPPED_TRACKS)


def linf(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True,
                    help="JSON produced by nonidentifiability_empirical.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as fh:
        data = json.load(fh)
    tables: Dict[str, List[dict]] = data["per_piece"]

    obs = []
    for sysname, rows in tables.items():
        for r in rows:
            if r.get("hm") is None:
                continue
            o = dict(r)
            o["system"] = sysname
            obs.append(o)

    pairs = []
    for i, j in itertools.combinations(range(len(obs)), 2):
        a, b = obs[i], obs[j]
        if a["piece"] == b["piece"] and a["system"] == b["system"]:
            continue
        pairs.append((linf(shipped_vec(a), shipped_vec(b)),
                      abs(a["hm"] - b["hm"]),
                      a["piece"] == b["piece"]))

    eps_grid = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    gap_grid = [0.05, 0.10, 0.15, 0.20, 0.30]

    print(f"observations={len(obs)}  pairs examined={len(pairs)}\n")
    print("counts of pairs the shipped protocol cannot separate, "
          "by |HM gap| threshold")
    header = "  eps_ship |" + "".join(f"  gap>={g:<5.2f}" for g in gap_grid) \
             + "   n_within_eps"
    print(header)
    print("  " + "-" * (len(header) - 2))
    surface = []
    for eps in eps_grid:
        within = [p for p in pairs if p[0] <= eps]
        cells = []
        for g in gap_grid:
            n = sum(1 for p in within if p[1] >= g)
            cells.append(n)
            surface.append(dict(eps_shipped=eps, min_hm_gap=g, n_matches=n,
                                n_within_eps=len(within)))
        print(f"  {eps:<8.3f} |" + "".join(f"  {c:<10d}" for c in cells)
              + f"   {len(within)}")

    print("\ndistribution of |HM gap| among shipped-indistinguishable pairs")
    print("  eps_ship |   n |  median |    p90 |    max")
    print("  " + "-" * 48)
    dist = []
    for eps in eps_grid:
        gaps = [p[1] for p in pairs if p[0] <= eps]
        if not gaps:
            print(f"  {eps:<8.3f} |   0 |      - |      - |      -")
            continue
        gaps_sorted = sorted(gaps)
        p90 = gaps_sorted[int(0.90 * (len(gaps_sorted) - 1))]
        row = dict(eps_shipped=eps, n=len(gaps),
                   median=statistics.median(gaps), p90=p90, max=max(gaps))
        dist.append(row)
        print(f"  {eps:<8.3f} | {len(gaps):>3} |  {row['median']:.4f} | "
              f"{p90:.4f} | {max(gaps):.4f}")

    same = [p for p in pairs if p[2]]
    print(f"\nsame-piece cross-system pairs: {len(same)}")
    if same:
        sg = sorted(p[1] for p in same)
        print(f"  |HM gap| median={statistics.median(sg):.4f}  "
              f"max={max(sg):.4f}")
        sl = sorted(p[0] for p in same)
        print(f"  shipped L-inf median={statistics.median(sl):.4f}  "
              f"min={min(sl):.4f}")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.inp)),
                                   "nonidentifiability_thresholds.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(n_observations=len(obs), n_pairs=len(pairs),
                       surface=surface, gap_distribution=dist), fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
