# Note on paired-analysis artifacts

`paired_analysis.json` is STALE with respect to the letter: it records a
DIFFERENT (earlier) comparison and is NOT the source of any number printed in
the letter or the supplementary material. Do not quote it in the paper.

The letter's Table II (prompting ablation) is derived exclusively from
`paired_prompted_vs_unprompted.json` (via `figs/make_tables.py`).

The file is kept under its original name because
`experiments/verify_shipped.py` (check 6c) reads `paired_analysis.json` by
name to re-verify the two per-piece counts documented in the executive
summary (`hm.n_pieces_favouring_a = 170`, `loc_f1.n_pieces_favouring_b =
173`); renaming it would fail the verification gate. Those counts belong to
the earlier comparison only.
