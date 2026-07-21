# decoupled-mer-eval

Implementation, evaluation harness, and result artifacts for

> D. C. Charapalle, "A Decoupled Localization–Classification Measure for Music
> Error Detection," submitted to *IEEE Signal Processing Letters*, 2026.

The letter proves that the per-class scores shipped by music error-detection
systems cannot identify their localized-but-misclassified mass (**HM**), bounds
the range a shipped report leaves open, and introduces a decoupled measure —
localization `F(τ)` under a shared class-agnostic onset match, then class
confusion on the matched events — with a backward-compatible mode that
reproduces the shipped scoring.

## Layout

```
experiments/
  decoupled_scorer.py            the measure (v1.1.0): exact max-cardinality,
                                 min-total-distance onset matching (padded
                                 rectangular assignment), symmetric wrong-note
                                 collapse, HM(τ), Loc-F(τ), shipped mode
  test_decoupled_scorer.py       unit tests
  test_mass_conservation_real.py falsifiability tests for the invariant
  paired_analysis.py             Wilcoxon / Cliff's δ / paired bootstrap
  nonidentifiability_*.py        constructed witness + empirical search
  bridge_checks.py               39 numerical checks of the analytic results
  bridge_predictions.py          track-position → class-name bridge for
                                 multi-track prediction MIDIs
  verify_shipped.py              mechanical gate: re-derives every number in
                                 the letter from the artifacts (LF/syntax/
                                 imports/tests/bridge/tables/figures)
  figs/                          figure + table generators (no hand-typed
                                 numbers; tables are byte-diffed by the gate)
  results/gilbreth/              scored artifacts cited in the letter
  results/gilbreth_v110/         v1.1.0 re-score parity artifacts
```

## Reproducing the letter's numbers

Every printed number traces to an artifact in `experiments/results/`:

```bash
pip install -r requirements.txt
cd experiments
python -m pytest test_decoupled_scorer.py test_mass_conservation_real.py
python bridge_checks.py            # analytic-bridge checks
python figs/verify_two_scenario.py # Fig. 1's exact claim
python verify_shipped.py           # full gate (skips cluster parity without --cluster)
cd figs && python make_tables.py && python fig1_two_scenario.py \
    && python fig2_measured.py && python figS_witness.py
```

Re-running inference end-to-end additionally requires the systems' released
checkpoints and the error-annotated MAESTRO split released with Polytune
(HuggingFace `ben2002chou`); see the letter's Sec. V for the protocol
(τ grid {50,75,100,150,200,500} ms, ε = 50 ms, seed 20260718).

## Artifact provenance

Result JSONs are published byte-exact as produced on the evaluation cluster
(NVIDIA A100 inference; CPU scoring). `config` blocks record the run-time
paths and a scorer-version string; a historical `note` field in early
artifacts predates the final pipeline and is retained rather than edited.

## License

MIT (see `LICENSE`).
