"""Per-piece (macro) replication of the published missed/extra F1 aggregation.

The letter's replication pools TP/FP/FN over events before forming F1; the
published Polytune/LadderSym evaluators instead compute per-piece F1 and
average over pieces. This script recomputes the *published-style* per-piece
(macro) missed/extra F1 from the per-piece coupled matrices in
results/gilbreth/*_guard.json, so the letter's reconciliation sentence is
derived from artifacts rather than typed by hand.

Per piece: TP_extra = matrix[0][0] (rows = predicted class, order
["Extra", "Removed", "Correct"]; cols = reference class, same order),
TP_missed = matrix[1][1]; FP_k = pred_counts[k] - TP_k;
FN_k = gt_counts[k] - TP_k; F1 = 2TP / (2TP + FP + FN), 0 when the
denominator is 0 (mir_eval convention). Macro-average over all n=177 pieces.

Cross-checks (fail-loud): pooled per-piece TPs must equal the shipped_50ms
artifact's pooled TPs (Polytune: extra 27650, missed 7088), and n_pieces=177.

Run:   python replication_aggregation.py
Writes results/gilbreth/replication_macro.json
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GIL = os.path.join(HERE, "results", "gilbreth")

STEMS = {
    "polytune": "A_polytune_maestro",
    "laddersym_unprompted": "B_laddersym_maestro_unprompted",
    "laddersym_prompted": "B_laddersym_maestro_prompted",
}

# guard matrix orientation (asserted against config below)
ROW_ORDER = ["Extra", "Removed", "Correct"]
COL_ORDER = ["extra_notes_midi", "removed_notes_midi", "correct_notes_midi"]


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _f1(tp: int, fp: int, fn: int) -> float:
    den = 2 * tp + fp + fn
    return 2 * tp / den if den else 0.0


def macro(stem: str) -> dict:
    g = _load(os.path.join(GIL, f"{stem}_guard.json"))
    assert g["config"]["row_order"] == ROW_ORDER, g["config"]["row_order"]
    assert g["config"]["col_order"] == COL_ORDER, g["config"]["col_order"]
    assert g["n_pieces"] == 177, g["n_pieces"]

    f1_missed, f1_extra = [], []
    tp_m_pool = tp_e_pool = 0
    for piece, rec in g["per_piece"].items():
        m = rec["matrix"]
        tp_e, tp_m = m[0][0], m[1][1]
        fp_e = rec["pred_counts"]["Extra"] - tp_e
        fn_e = rec["gt_counts"]["extra_notes_midi"] - tp_e
        fp_m = rec["pred_counts"]["Removed"] - tp_m
        fn_m = rec["gt_counts"]["removed_notes_midi"] - tp_m
        assert min(fp_e, fn_e, fp_m, fn_m) >= 0, piece
        f1_extra.append(_f1(tp_e, fp_e, fn_e))
        f1_missed.append(_f1(tp_m, fp_m, fn_m))
        tp_e_pool += tp_e
        tp_m_pool += tp_m

    # cross-check pooled TPs against the shipped_50ms artifact
    sh = _load(os.path.join(GIL, f"{stem}_shipped.json"))["shipped_50ms"]
    assert tp_m_pool == sh["missed"]["tp"], (stem, tp_m_pool, sh["missed"]["tp"])
    assert tp_e_pool == sh["extra"]["tp"], (stem, tp_e_pool, sh["extra"]["tp"])

    n = len(f1_missed)
    return {
        "n_pieces": n,
        "pooled_tp_missed": tp_m_pool,
        "pooled_tp_extra": tp_e_pool,
        "pooled_f1_missed": sh["missed"]["f1"],
        "pooled_f1_extra": sh["extra"]["f1"],
        "macro_f1_missed": sum(f1_missed) / n,
        "macro_f1_extra": sum(f1_extra) / n,
    }


def main() -> None:
    out = {
        "description": "Published-style per-piece (macro) missed/extra F1, "
                       "recomputed from per-piece coupled matrices in "
                       "*_guard.json; pooled TPs cross-checked against "
                       "*_shipped.json shipped_50ms.",
        "onset_tolerance_s": 0.05,
        "systems": {},
    }
    for name, stem in STEMS.items():
        r = macro(stem)
        out["systems"][name] = r
        print(f"{name:24s} macro missed/extra F1 = "
              f"{r['macro_f1_missed']:.3f}/{r['macro_f1_extra']:.3f}  "
              f"(pooled {r['pooled_f1_missed']:.3f}/{r['pooled_f1_extra']:.3f}, "
              f"pooled TP m/e = {r['pooled_tp_missed']}/{r['pooled_tp_extra']})")
    dest = os.path.join(GIL, "replication_macro.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()
