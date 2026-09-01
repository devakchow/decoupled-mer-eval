# decoupled-mer-eval

Implementation, evaluation harness, and result artifacts for

> D. C. Charapalle, "A Decoupled Localization–Classification Measure for Music
> Error Detection," 2026. Prepared for submission to *IEEE Signal Processing
> Letters* (citation will be updated on submission/publication).

The letter proves that the per-class scores published by music error-detection
systems cannot identify their localized-but-misclassified mass (**HM**),
exhibits a range a published report leaves open, and introduces a decoupled
measure —
localization `F(τ)` under a shared class-agnostic onset match, then class
confusion on the matched events — with a backward-compatible mode that
reproduces the published scoring.

## Layout

```
experiments/
  decoupled_scorer.py            the measure (v1.1.0): exact max-cardinality,
                                 min-total-distance onset matching (padded
                                 rectangular assignment), symmetric wrong-note
                                 collapse, HM(τ), Loc-F(τ), shipped mode
  test_decoupled_scorer.py       unit tests
  test_mass_conservation_real.py falsifiability tests for the invariant
  synthetic_oracle.py            model-free recovery experiments: planted HM*
                                 recovered exactly while per-class F1 is blind
                                 to it (oracle_report.json holds the outputs)
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
  evidence/                      CocoChorales ground-truth join + census
                                 (provenance for the 85.6%/43.2% selection-bias
                                 figures in the supplementary material)
  results/gilbreth/              scored artifacts cited in the letter, incl.
                                 the collapse-free arm (*_nocollapse.json:
                                 --collapse none, the two-class functional
                                 Proposition 1 bounds)
  results/cluster/               adjudication of merged wrong-events against
                                 the score (genuine/ambiguous/unfounded), the
                                 per-piece bootstrap for those columns, the
                                 anchor-window sweep, and the matcher tie-break
                                 negation test; each carries a _provenance block
  results/gilbreth_v110/         output of the independent v1.1.0 re-score job
                                 (cluster job 11341983), retained verbatim; the
                                 scorer is deterministic and timestamp-free, so
                                 byte-identity with results/gilbreth/ is the
                                 expected outcome of the parity check
  results/gilbreth_ei/           MAESTRO-EI campaign: all three configurations
                                 scored under the 4-variant x 6-tolerance grid
                                 (with per-piece bootstrap), dominance guard,
                                 track census, and manifest adjudication
                                 (collapse_validation_ei_{A,Bu,Bp}.json)
  setup/                         MAESTRO-EI toolchain: inject_maestro_ei.py
                                 (generator), validate_maestro_ei.py (label-
                                 stream validation), collapse_validation_ei.py
                                 (manifest adjudication), gen_gt_meta.py
```

## MAESTRO-EI

MAESTRO-EI re-injects the 177 MAESTRO-E test scores at the authentic beginner
error mix (46.58 / 31.65 / 21.76 % substitution / insertion / omission;
185,346 injections) with a per-injection manifest whose substitutions carry an
explicit pairing edge (removed -> inserted), plus 4,320 flagged decoy
deletion-insertion pairs placed >= 2 s apart. Correct notes are copied
verbatim. The generator is deterministic (seed 20260831; per-piece RNG from the
SHA-1 of the piece name), so the corpus is reproducible from the MAESTRO-E
scores alone:

```bash
cd experiments/setup
python inject_maestro_ei.py --help      # writes label/, manifest/, corpus_summary.json
python validate_maestro_ei.py           # decoys, planted flips, adjudication ceiling
python collapse_validation_ei.py        # manifest-genuine rates per confusion cell
```

The label MIDIs and manifests as used in the letter are published as a release
asset (`maestro-ei-labels-manifests.tgz`, sha256
`e3fec707032c0ebf93faea24a092cf5e4bf285e6f883f2071d95d8fcf336f85c`) at
<https://github.com/devakchow/decoupled-mer-eval/releases/tag/maestro-ei-v1>.
Audio is not distributed: render the mistake and score performances with the
release corpus's own recipe (FluidSynth, the release soundfont, 16 kHz mono
PCM-24) as in `render_ei.py`; `results/cluster/maestro_ei_summary.json` and
`maestro_ei_validate.json` hold the corpus census and label-stream validation.

## Reproducing the letter's numbers

Every printed number traces to an artifact in `experiments/results/`:

```bash
python -m pip install -r requirements.txt
cd experiments
python -m pytest test_decoupled_scorer.py test_mass_conservation_real.py
python bridge_checks.py            # analytic-bridge checks
python figs/verify_two_scenario.py # Fig. 1's exact claim
python verify_shipped.py           # full gate (skips cluster parity without --cluster)
cd figs && python make_tables.py && python fig1_two_scenario.py \
    && python fig2_measured.py && python figS_witness.py
```

Note: regenerated figure PDFs are not byte-stable across matplotlib builds
(compare content, not hashes), and a CRLF-converting checkout can show
phantom diffs in the regenerated `.tex` tables.

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
