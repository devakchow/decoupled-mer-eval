#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_decoupled_scorer.py -- pytest suite for the production decoupled MER
scorer (Phase 2, IEEE SPL).

Regression anchors:
  * experiments/oracle_report.json -- the verified stdlib oracle's numbers
    (shipped mean-error-F1 = 0.6667 with HM = 1/3 vs 0; planted HM* = 0.16667
    recovered across 75-500 ms; the double-count witness).
  * experiments/synthetic_oracle.py -- imported to (a) rebuild the injected
    corpus for the Experiment-C sweep and (b) expose the GREEDY matcher the
    production module replaces, so the chord stress test can show greedy is
    strictly weaker than the exact max-cardinality matcher.

Run:  pytest -q test_decoupled_scorer.py
"""

import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

import decoupled_scorer as ds
import synthetic_oracle as oracle

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "oracle_report.json"), "r", encoding="utf-8") as _fh:
    REPORT = json.load(_fh)


def conv(oracle_events):
    """Oracle Event (onset, pitch, etype, piece) -> production Event."""
    return [ds.Event(e.onset, e.pitch, e.etype, e.piece) for e in oracle_events]


# --------------------------------------------------------------------------- #
# 1. Regression vs oracle_report.json                                         #
# --------------------------------------------------------------------------- #

class TestOracleRegressionA:
    """Experiment A: the double-count witness (1 FP + 1 FN in two tracks vs
    exactly ONE localized event)."""

    def test_a1_missed_as_extra(self):
        ref = [ds.Event(1.0, 61, "missed")]
        pred = [ds.Event(1.0, 61, "extra")]
        ship = ds.shipped_scores(pred, ref, tau=0.05)
        dec = ds.decoupled_scores(pred, ref, tau=0.05)
        exp = REPORT["experiment_A"]["A1_missed_as_extra"]

        # shipped: ONE physical event charged as FN in the missed track AND
        # FP in the extra track (F1 = 0 in both) -- the double count.
        for track in ("missed", "extra", "correct"):
            for k in ("tp", "fp", "fn"):
                assert ship[track][k] == exp["ship"][track][k]
            assert ship[track]["f1"] == pytest.approx(exp["ship"][track]["f1"])
        assert ship["missed"]["fn"] == 1 and ship["extra"]["fp"] == 1
        assert ship["missed"]["f1"] == 0.0 and ship["extra"]["f1"] == 0.0

        # decoupled: exactly ONE localized-but-misclassified event.
        assert dec.n_localized == exp["dec"]["n_localized"] == 1
        assert dec.confusion_sparse() == exp["dec"]["confusion"] == {"missed->extra": 1}
        assert dec.hm == exp["dec"]["hm"] == 1.0
        assert dec.spurious == 0 and dec.miss_margin == 0
        assert dec.false_alarm_on_correct == 0
        assert dec.mass_conserved()

    def test_a2_wrong_as_extra(self):
        ref = [ds.Event(1.0, 61, "missed"), ds.Event(1.0, 60, "extra")]
        pred = [ds.Event(1.0, 60, "extra")]
        ship = ds.shipped_scores(pred, ref, tau=0.05)
        dec = ds.decoupled_scores(pred, ref, tau=0.05)
        exp = REPORT["experiment_A"]["A2_wrong_as_extra"]

        assert ship["extra"]["tp"] == 1 and ship["missed"]["fn"] == 1
        assert ship["_mean_error_f1"] == pytest.approx(exp["ship"]["_mean_error_f1"])
        assert dec.n_localized == 1
        assert dec.confusion_sparse() == exp["dec"]["confusion"] == {"wrong->extra": 1}
        assert dec.hm == 1.0
        assert dec.mass_conserved()


def _build_experiment_b():
    """Rebuild the oracle's Experiment-B systems (README_oracle.md Table B)."""
    def at(k):
        return 2.0 * k

    ref1, pred1, ref2, pred2 = [], [], [], []
    k = 1
    for _ in range(4):
        ref1.append(ds.Event(at(k), 60, "missed")); pred1.append(ds.Event(at(k), 60, "missed"))
        ref2.append(ds.Event(at(k), 60, "missed")); pred2.append(ds.Event(at(k), 60, "missed")); k += 1
    for _ in range(4):
        ref1.append(ds.Event(at(k), 60, "extra")); pred1.append(ds.Event(at(k), 60, "extra"))
        ref2.append(ds.Event(at(k), 60, "extra")); pred2.append(ds.Event(at(k), 60, "extra")); k += 1
    for _ in range(2):  # sys1: localized mistype; sys2: pure miss
        ref1.append(ds.Event(at(k), 60, "missed")); pred1.append(ds.Event(at(k), 60, "extra"))
        ref2.append(ds.Event(at(k), 60, "missed")); k += 1
    for _ in range(2):
        ref1.append(ds.Event(at(k), 60, "extra")); pred1.append(ds.Event(at(k), 60, "missed"))
        ref2.append(ds.Event(at(k), 60, "extra")); k += 1
    for _ in range(2):  # sys2: spurious predictions
        pred2.append(ds.Event(at(k), 60, "missed")); k += 1
    for _ in range(2):
        pred2.append(ds.Event(at(k), 60, "extra")); k += 1
    return (ref1, pred1), (ref2, pred2)


class TestOracleRegressionB:
    """Experiment B: identical shipped mean-error-F1 = 0.6667, HM = 1/3 vs 0
    (the non-identifiability witness)."""

    def test_identical_shipped_f1_different_hm(self):
        (ref1, pred1), (ref2, pred2) = _build_experiment_b()
        s1 = ds.shipped_scores(pred1, ref1, tau=0.05)
        s2 = ds.shipped_scores(pred2, ref2, tau=0.05)
        d1 = ds.decoupled_scores(pred1, ref1, tau=0.05)
        d2 = ds.decoupled_scores(pred2, ref2, tau=0.05)
        b = REPORT["experiment_B"]

        # shipped per-track F1 vectors identical, = 0.6667
        for s, exp in ((s1, b["system1"]["ship"]), (s2, b["system2"]["ship"])):
            for track in ("missed", "extra"):
                assert s[track]["tp"] == exp[track]["tp"] == 4
                assert s[track]["fp"] == exp[track]["fp"] == 2
                assert s[track]["fn"] == exp[track]["fn"] == 2
                assert s[track]["f1"] == pytest.approx(2 / 3)
            assert s["_mean_error_f1"] == pytest.approx(exp["_mean_error_f1"])
        assert round(s1["_mean_error_f1"], 4) == round(s2["_mean_error_f1"], 4) == 0.6667

        # decoupled separates them: HM = 1/3 vs 0
        assert d1.hm == pytest.approx(1 / 3)
        assert d2.hm == 0.0
        assert d1.n_localized == b["system1"]["dec"]["n_localized"] == 12
        assert d2.n_localized == b["system2"]["dec"]["n_localized"] == 8
        assert d1.confusion_sparse() == b["system1"]["dec"]["confusion"]
        assert d2.confusion_sparse() == b["system2"]["dec"]["confusion"]
        assert d2.spurious == 4 and d2.miss_margin == 4
        assert d1.localization["f1"] == pytest.approx(
            b["system1"]["dec"]["localization"]["f1"])
        assert d2.localization["f1"] == pytest.approx(
            b["system2"]["dec"]["localization"]["f1"])
        assert d1.mass_conserved() and d2.mass_conserved()


@pytest.fixture(scope="module")
def corpus():
    """The oracle's Experiment-C injected corpus (jitter = 60 ms), converted
    to production events, plus the planted HM*."""
    refs, preds, tot, mis = oracle._build_injected_corpus(jitter=0.060)
    return conv(refs), conv(preds), mis / tot


class TestOracleRegressionC:
    """Experiment C: planted HM* = 0.16667 recovered across 75-500 ms; full
    sweep and confusion identical to oracle_report.json."""

    def test_planted_hm(self, corpus):
        _, _, planted = corpus
        assert planted == pytest.approx(REPORT["experiment_C"]["planted_hm"])
        assert planted == pytest.approx(1 / 6)

    def test_sweep_matches_oracle_report(self, corpus):
        refs, preds, planted = corpus
        for row in REPORT["experiment_C"]["sweep"]:
            tau = row["tau_ms"] / 1000.0
            dec = ds.decoupled_scores(preds, refs, tau=tau)
            ship = ds.shipped_scores(preds, refs, tau=tau)
            assert round(dec.localization["f1"], 4) == row["localization_f1"], row
            assert dec.n_localized == row["n_localized"], row
            if row["hm"] is None:
                assert dec.hm is None, row
            else:
                assert round(dec.hm, 4) == row["hm"], row
            assert round(ship["_mean_error_f1"], 4) == row["shipped_mean_error_f1"], row
            assert dec.mass_conserved() is True
        # HM* recovered at every tau in 75-500 ms; undefined (None) at 50 ms.
        for tau_ms in (75, 100, 150, 200, 500):
            dec = ds.decoupled_scores(preds, refs, tau=tau_ms / 1000.0)
            assert dec.hm == pytest.approx(planted, abs=1e-12)
        assert ds.decoupled_scores(preds, refs, tau=0.050).hm is None

    def test_confusion_at_500ms(self, corpus):
        refs, preds, _ = corpus
        dec = ds.decoupled_scores(preds, refs, tau=0.500)
        assert dec.confusion_sparse() == REPORT["experiment_C"]["confusion_500ms"]
        assert dec.off_diagonal() == 10 and dec.n_localized == 60

    def test_mean_loc_f1_over_tau_grid_summary(self, corpus):
        refs, preds, _ = corpus
        sw = ds.sweep(preds, refs)
        assert list(sw["taus_s"]) == list(ds.TAU_GRID_S)
        # loc F1 = [0, 1, 1, 1, 1, 1] -> mean 5/6.
        # Key renamed in v1.1.0: the average is over the TAU GRID, not over
        # pieces; the old name `mean_loc_f1` read as a macro-per-piece mean.
        assert sw["mean_loc_f1_over_tau_grid"] == pytest.approx(5 / 6)
        assert "mean_loc_f1" not in sw, "old ambiguous key must not resurface"


# --------------------------------------------------------------------------- #
# 2. Chord/dense stress: true max-cardinality vs the oracle's greedy pass.    #
#    THIS TEST IS THE POINT OF THE REBUILD.                                   #
# --------------------------------------------------------------------------- #

class TestMaxCardinalityVsGreedy:

    def test_crafted_case_greedy_loses(self):
        """Crafted onsets where greedy ascending-distance matching yields
        FEWER matches than the optimum: preds {0.45, 0.52}, refs {0.50, 0.90},
        tau = 0.40. Greedy takes (0.52, 0.50) first (d=0.02), stranding 0.45
        whose only remaining partner 0.90 is out of range (d=0.45 > 0.40).
        Optimal pairs (0.45, 0.50) + (0.52, 0.90) -> cardinality 2."""
        o_pred = [oracle.Event(0.52, 60, "extra"), oracle.Event(0.45, 60, "extra")]
        o_ref = [oracle.Event(0.50, 60, "missed"), oracle.Event(0.90, 60, "missed")]
        greedy = oracle.match_onsets(o_pred, o_ref, tau=0.40)
        assert len(greedy) == 1  # the greedy oracle pass demonstrably fails

        pred = conv(o_pred)
        ref = conv(o_ref)
        exact = ds.match_events(pred, ref, tau=0.40)
        assert len(exact) == 2   # the production matcher attains the maximum
        assert sorted(exact) == [(0, 1), (1, 0)]

        # and the full scorer sees 2 localized events, not 1
        dec = ds.decoupled_scores(pred, ref, tau=0.40, epsilon=0.0)
        assert dec.n_localized == 2
        assert dec.localization["f1"] == 1.0
        assert dec.mass_conserved()

    def test_chord_cluster_full_cardinality(self):
        """A 5-note chord on each side within tau must match completely."""
        pred = [ds.Event(1.000 + 0.001 * i, 60 + i, "extra") for i in range(5)]
        ref = [ds.Event(1.002 + 0.001 * i, 60 + i, "missed") for i in range(5)]
        m = ds.match_events(pred, ref, tau=0.05)
        assert len(m) == 5

    def test_random_dense_never_below_greedy(self):
        """On 300 random dense clusters the exact matcher's cardinality is
        never below greedy's, and strictly above at least once (greedy is
        provably suboptimal on dense data)."""
        rng = random.Random(20260718)
        strictly_better = 0
        for _ in range(300):
            n_r, n_p = rng.randint(1, 12), rng.randint(1, 12)
            o_ref = [oracle.Event(round(rng.uniform(0, 0.4), 3),
                                  rng.randint(58, 66), "missed") for _ in range(n_r)]
            o_pred = [oracle.Event(round(rng.uniform(0, 0.4), 3),
                                   rng.randint(58, 66), "extra") for _ in range(n_p)]
            tau = rng.choice([0.05, 0.075, 0.1])
            greedy_n = len(oracle.match_onsets(o_pred, o_ref, tau))
            exact_n = len(ds.match_events(conv(o_pred), conv(o_ref), tau))
            assert exact_n >= greedy_n
            assert exact_n <= min(n_r, n_p)
            if exact_n > greedy_n:
                strictly_better += 1
        assert strictly_better >= 1, \
            "expected at least one dense cluster where greedy is suboptimal"

    def test_matcher_min_total_distance_tiebreak(self):
        """Among max-cardinality matchings, total onset distance is minimal:
        preds {1.00, 1.10} vs refs {1.01, 1.11}: parallel pairing costs
        0.01+0.01=0.02; crossed costs 0.11+0.09=0.20. Must pick parallel."""
        pred = [ds.Event(1.00, 60, "extra"), ds.Event(1.10, 60, "extra")]
        ref = [ds.Event(1.01, 60, "missed"), ds.Event(1.11, 60, "missed")]
        m = ds.match_events(pred, ref, tau=0.2)
        assert sorted(m) == [(0, 0), (1, 1)]

    def test_determinism_under_input_permutation(self):
        """Scorer output is a canonical function of the event multiset."""
        rng = random.Random(7)
        ref = [ds.Event(round(rng.uniform(0, 1), 3), rng.randint(58, 66),
                        rng.choice(ds.SHIP_TRACKS)) for _ in range(20)]
        pred = [ds.Event(round(rng.uniform(0, 1), 3), rng.randint(58, 66),
                         rng.choice(ds.SHIP_TRACKS)) for _ in range(20)]
        base = ds.decoupled_scores(pred, ref, tau=0.1)
        for trial in range(5):
            rp, rr = list(pred), list(ref)
            rng.shuffle(rp)
            rng.shuffle(rr)
            got = ds.decoupled_scores(rp, rr, tau=0.1)
            assert got.confusion == base.confusion
            assert got.localization == base.localization
            assert (got.spurious, got.miss_margin, got.false_alarm_on_correct) \
                == (base.spurious, base.miss_margin, base.false_alarm_on_correct)
            assert got.hm == base.hm


# --------------------------------------------------------------------------- #
# 3. Property tests: mass conservation + tau-monotonicity, >= 500 corpora.    #
# --------------------------------------------------------------------------- #

def _random_corpus(rng):
    """Random corpus mixing well-separated onsets, DENSE clusters, and two
    pieces (to exercise within-piece isolation)."""
    events = []
    for piece in ("pA", "pB"):
        n_wide = rng.randint(0, 10)
        n_dense = rng.randint(0, 12)
        for _ in range(n_wide):
            events.append(ds.Event(round(rng.uniform(0, 10), 3),
                                   rng.randint(48, 84),
                                   rng.choice(ds.SHIP_TRACKS), piece))
        center = rng.uniform(0, 10)
        for _ in range(n_dense):
            events.append(ds.Event(round(center + rng.uniform(-0.15, 0.15), 3),
                                   rng.randint(58, 66),
                                   rng.choice(ds.SHIP_TRACKS), piece))
    return events


class TestProperties:

    @pytest.mark.parametrize("collapse", ["strict", "pitch_aware"])
    def test_mass_conservation_and_tau_monotonicity(self, collapse):
        n_corpora = 250  # x2 collapse modes = 500 random corpora total
        rng = random.Random(20260718 if collapse == "strict" else 42)
        for trial in range(n_corpora):
            ref = _random_corpus(rng)
            pred = _random_corpus(rng)
            prev_tp = -1
            n_pop = None
            for tau in ds.TAU_GRID_S:
                r = ds.decoupled_scores(pred, ref, tau=tau, collapse=collapse)
                # mass conservation at every tau (also enforced internally)
                assert r.mass_conserved(), (collapse, trial, tau)
                # tau-monotonicity of the max-cardinality matched count
                assert r.localization["tp"] >= prev_tp, (collapse, trial, tau)
                prev_tp = r.localization["tp"]
                # constant event population across the sweep (eps independent
                # of tau)
                if n_pop is None:
                    n_pop = (r.n_pred_err, r.n_ref_err)
                assert (r.n_pred_err, r.n_ref_err) == n_pop, (collapse, trial, tau)
                # HM is None exactly when nothing is localized -- never 0/0=0
                assert (r.hm is None) == (r.n_localized == 0)

    def test_hm_none_not_zero_when_nothing_localized(self):
        ref = [ds.Event(0.0, 60, "missed")]
        pred = [ds.Event(9.0, 60, "extra")]
        r = ds.decoupled_scores(pred, ref, tau=0.05)
        assert r.n_localized == 0
        assert r.hm is None
        assert r.spurious == 1 and r.miss_margin == 1
        assert r.mass_conserved()

    def test_false_alarm_on_correct_second_pass(self):
        """An unmatched predicted error that onset-matches a reference
        CORRECT note lands in the false-alarm margin, not spurious."""
        ref = [ds.Event(1.0, 60, "correct")]
        pred = [ds.Event(1.01, 60, "extra")]
        r = ds.decoupled_scores(pred, ref, tau=0.05)
        assert r.false_alarm_on_correct == 1
        assert r.spurious == 0
        assert r.n_localized == 0 and r.hm is None
        assert r.mass_conserved()


# --------------------------------------------------------------------------- #
# 4. Wrong-note collapse rules (strict PRIMARY / pitch-aware SENSITIVITY).    #
# --------------------------------------------------------------------------- #

class TestCollapse:

    def test_strict_collapses_any_pitch_pair(self):
        events = [ds.Event(1.0, 60, "missed"), ds.Event(1.0, 60, "extra")]
        out = ds.collapse_wrong(events, epsilon=0.05, mode="strict")
        assert [e.etype for e in out] == ["wrong"]
        assert out[0].pitch_midi == 60 and out[0].onset_s == 1.0

    def test_pitch_aware_requires_pitch_difference(self):
        same = [ds.Event(1.0, 60, "missed"), ds.Event(1.0, 60, "extra")]
        out = ds.collapse_wrong(same, epsilon=0.05, mode="pitch_aware")
        assert sorted(e.etype for e in out) == ["extra", "missed"]  # NOT collapsed
        diff = [ds.Event(1.0, 61, "missed"), ds.Event(1.0, 60, "extra")]
        out2 = ds.collapse_wrong(diff, epsilon=0.05, mode="pitch_aware")
        assert [e.etype for e in out2] == ["wrong"]
        assert out2[0].pitch_midi == 60  # wrong carries the played (extra) pitch

    def test_epsilon_independent_of_tau_and_eps_zero(self):
        pair = [ds.Event(1.00, 61, "missed"), ds.Event(1.04, 60, "extra")]
        assert [e.etype for e in ds.collapse_wrong(pair, epsilon=0.05)] == ["wrong"]
        # sensitivity eps = 0: 40 ms apart -> no collapse
        out = ds.collapse_wrong(pair, epsilon=0.0)
        assert sorted(e.etype for e in out) == ["extra", "missed"]

    def test_collapse_none_passes_events_through(self):
        pair = [ds.Event(1.00, 60, "missed"), ds.Event(1.00, 64, "extra")]
        out = ds.collapse_wrong(pair, epsilon=0.05, mode="none")
        assert [e.etype for e in out] == ["missed", "extra"]
        assert len(out) == 2

    def test_score_consistency_filter_keeps_only_score_named_deletions(self):
        ref = [ds.Event(1.00, 60, "correct"), ds.Event(2.00, 62, "missed"),
               ds.Event(3.00, 64, "extra")]
        pred = [
            ds.Event(1.04, 60, "missed"),   # names a played score note: keep
            ds.Event(2.05, 62, "missed"),   # names an omitted score note: keep
            ds.Event(2.06, 62, "missed"),   # 60 ms from it: drop
            ds.Event(1.00, 61, "missed"),   # pitch absent from score: drop
            ds.Event(3.00, 64, "missed"),   # names only a reference extra: drop
            ds.Event(3.00, 64, "extra"),    # non-missed claims pass through
            ds.Event(9.00, 99, "correct"),
        ]
        kept, dropped = ds.score_consistency_filter(pred, ref, anchor=0.05)
        assert dropped == 3
        assert [(e.onset_s, e.etype) for e in kept] == [
            (1.04, "missed"), (2.05, "missed"), (3.00, "extra"), (9.00, "correct")]

    def test_chord_resolution_strict_min_onset_distance(self):
        """Two missed + two extra in a chord: strict pairing must be the
        min-total-onset-distance max-cardinality assignment."""
        events = [
            ds.Event(1.000, 60, "missed"), ds.Event(1.030, 64, "missed"),
            ds.Event(1.001, 70, "extra"), ds.Event(1.029, 71, "extra"),
        ]
        out = ds.collapse_wrong(events, epsilon=0.05, mode="strict")
        assert sorted(e.etype for e in out) == ["wrong", "wrong"]
        # wrong events carry the extra onsets/pitches
        assert sorted((e.onset_s, e.pitch_midi) for e in out) \
            == [(1.001, 70), (1.029, 71)]

    def test_chord_resolution_pitch_aware_min_semitone_distance(self):
        """Pitch-aware pairing minimizes total semitone distance: missed
        {60, 64} + extra {61}: pairs (60,61) [1 semitone] not (64,61)
        [3 semitones]; the leftover missed 64 survives."""
        events = [
            ds.Event(1.0, 60, "missed"), ds.Event(1.0, 64, "missed"),
            ds.Event(1.0, 61, "extra"),
        ]
        out = ds.collapse_wrong(events, epsilon=0.05, mode="pitch_aware")
        etypes = sorted(e.etype for e in out)
        assert etypes == ["missed", "wrong"]
        leftover = [e for e in out if e.etype == "missed"][0]
        assert leftover.pitch_midi == 64
        # cross-check: strict on the same input also leaves exactly one wrong
        out_s = ds.collapse_wrong(events, epsilon=0.05, mode="strict")
        assert sorted(e.etype for e in out_s) == ["missed", "wrong"]

    def test_collapse_is_within_piece(self):
        events = [ds.Event(1.0, 60, "missed", "pA"),
                  ds.Event(1.0, 61, "extra", "pB")]
        out = ds.collapse_wrong(events, epsilon=0.05)
        assert sorted(e.etype for e in out) == ["extra", "missed"]


# --------------------------------------------------------------------------- #
# 5. Backward-compat: shipped mode == direct mir_eval, to float tolerance.    #
# --------------------------------------------------------------------------- #

class TestShippedModeMirEval:

    def test_shipped_equals_mireval_on_random_note_sets(self):
        """60 random single-track corpora (incl. dense/pitch-colliding, and
        the 3-decimal onset grid that exercises mir_eval's N_DECIMALS = 4
        boundary rounding): shipped-mode P/R/F1 must equal a direct
        mir_eval.transcription.precision_recall_f1_overlap call with
        onset_tolerance=0.05, offset_ratio=None."""
        import mir_eval
        rng = random.Random(20260718)
        for trial in range(60):
            n_r, n_p = rng.randint(1, 15), rng.randint(1, 15)
            dense = trial % 2 == 0
            span = 0.5 if dense else 5.0

            def mk(n, etype):
                return [ds.Event(round(rng.uniform(0, span), 3),
                                 rng.randint(60, 64), etype) for _ in range(n)]

            ref = mk(n_r, "extra")
            pred = mk(n_p, "extra")
            ours = ds.shipped_scores(pred, ref, tau=0.05)["extra"]

            def hz(m):
                return 440.0 * (2.0 ** ((m - 69) / 12.0))

            ri = np.array([[e.onset_s, e.onset_s + 0.1] for e in ref])
            rp = np.array([hz(e.pitch_midi) for e in ref])
            pi = np.array([[e.onset_s, e.onset_s + 0.1] for e in pred])
            pp = np.array([hz(e.pitch_midi) for e in pred])
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ri, rp, pi, pp, onset_tolerance=0.05, offset_ratio=None)
            assert ours["precision"] == pytest.approx(p, abs=1e-12), trial
            assert ours["recall"] == pytest.approx(r, abs=1e-12), trial
            assert ours["f1"] == pytest.approx(f, abs=1e-12), trial

    def test_module_crosscheck_helper_agrees(self):
        rng = random.Random(1)
        ref = [ds.Event(round(rng.uniform(0, 1), 3), rng.randint(60, 63), "extra")
               for _ in range(10)]
        pred = [ds.Event(round(rng.uniform(0, 1), 3), rng.randint(60, 63), "extra")
                for _ in range(12)]
        ours = ds.shipped_scores(pred, ref, tau=0.05)["extra"]
        p, r, f = ds.mir_eval_track_prf(pred, ref, tau=0.05)
        assert ours["f1"] == pytest.approx(f, abs=1e-12)

    def test_shipped_mode_no_collapse(self):
        """Shipped mode must NOT collapse: a co-located missed+extra ref pair
        stays two separate track entries."""
        ref = [ds.Event(1.0, 61, "missed"), ds.Event(1.0, 60, "extra")]
        pred = [ds.Event(1.0, 61, "missed"), ds.Event(1.0, 60, "extra")]
        s = ds.shipped_scores(pred, ref, tau=0.05)
        assert s["missed"]["tp"] == 1 and s["extra"]["tp"] == 1
        assert s["_mean_error_f1"] == 1.0


# --------------------------------------------------------------------------- #
# 6. Bootstrap CI helper.                                                     #
# --------------------------------------------------------------------------- #

class TestBootstrap:

    def _corpus(self, hm_by_piece):
        """One localized event pair per slot; per piece, `mis` of `tot` slots
        are mistyped -> per-piece HM = mis/tot."""
        ref, pred = [], []
        for piece, (mis, tot) in hm_by_piece.items():
            for k in range(tot):
                onset = 2.0 * (k + 1)
                ref.append(ds.Event(onset, 60, "missed", piece))
                ptype = "extra" if k < mis else "missed"
                pred.append(ds.Event(onset, 60, ptype, piece))
        return pred, ref

    def test_point_estimates_match_pooled(self):
        pred, ref = self._corpus({"p0": (1, 4), "p1": (0, 4), "p2": (3, 4)})
        pooled = ds.decoupled_scores(pred, ref, tau=0.05)
        b = ds.bootstrap_piece_ci(pred, ref, tau=0.05, n_boot=200, seed=1)
        assert b["point_hm"] == pytest.approx(pooled.hm)
        assert b["point_loc_f1"] == pytest.approx(pooled.localization["f1"])
        assert b["n_pieces"] == 3

    def test_degenerate_ci_for_identical_pieces(self):
        pred, ref = self._corpus({"p%d" % i: (1, 4) for i in range(4)})
        b = ds.bootstrap_piece_ci(pred, ref, tau=0.05, n_boot=300, seed=3)
        lo, hi = b["hm_ci95"]
        assert lo == pytest.approx(0.25) and hi == pytest.approx(0.25)
        assert b["point_hm"] == pytest.approx(0.25)
        assert b["n_hm_undefined"] == 0

    def test_ci_brackets_point_and_is_deterministic(self):
        pred, ref = self._corpus({"p0": (2, 4), "p1": (0, 4), "p2": (1, 4),
                                  "p3": (4, 4), "p4": (0, 4)})
        b1 = ds.bootstrap_piece_ci(pred, ref, tau=0.05, n_boot=500, seed=99)
        b2 = ds.bootstrap_piece_ci(pred, ref, tau=0.05, n_boot=500, seed=99)
        assert b1 == b2  # deterministic under a fixed seed
        lo, hi = b1["hm_ci95"]
        assert lo <= b1["point_hm"] <= hi
        lo_f, hi_f = b1["loc_f1_ci95"]
        assert lo_f <= b1["point_loc_f1"] <= hi_f


# --------------------------------------------------------------------------- #
# 7. Real-data loader (tiny generated MIDI; Phase-3 schema flag).             #
# --------------------------------------------------------------------------- #

def _write_midi(path, notes_by_track):
    """notes_by_track: {track_name: [(start_s, pitch), ...]}."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    for tname, notes in notes_by_track.items():
        inst = pretty_midi.Instrument(program=0, name=tname)
        for start, pitch in notes:
            inst.notes.append(pretty_midi.Note(
                velocity=100, pitch=pitch, start=start, end=start + 0.2))
        pm.instruments.append(inst)
    pm.write(str(path))


@pytest.fixture()
def midi_fixture(tmp_path):
    """Tiny synthetic prediction dir + ground-truth metadata JSON following
    the documented (Phase-3-reconfirm) schema."""
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()

    # predicted multi-track MIDI for piece "id0"
    _write_midi(pred_dir / "id0.mid", {
        "Extra Notes": [(1.0, 61)],
        "Missing Notes": [(2.0, 62)],
        "Correct Notes": [(0.0, 60), (3.0, 65)],
    })
    # ground-truth reference MIDIs, keyed by the documented metadata keys
    _write_midi(gt_dir / "id0_extra.mid", {"t": [(1.0, 61)]})
    _write_midi(gt_dir / "id0_removed.mid", {"t": [(2.0, 62)]})
    _write_midi(gt_dir / "id0_correct.mid", {"t": [(0.0, 60), (3.0, 65)]})
    meta = {
        "id0": {
            "extra_notes_midi": "id0_extra.mid",
            "removed_notes_midi": "id0_removed.mid",
            "correct_notes_midi": "id0_correct.mid",
        }
    }
    meta_path = gt_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return pred_dir, meta_path


class TestLoader:

    def test_load_pred_events(self, midi_fixture):
        pred_dir, _ = midi_fixture
        events = ds.load_pred_events(str(pred_dir))
        by_type = {}
        for e in events:
            by_type.setdefault(e.etype, []).append(e)
        assert set(by_type) == {"extra", "missed", "correct"}
        assert len(by_type["extra"]) == 1 and len(by_type["missed"]) == 1
        assert len(by_type["correct"]) == 2
        assert all(e.piece == "id0" for e in events)
        # "Missing Notes" track mapped to missed; MIDI tick quantization only
        m = by_type["missed"][0]
        assert m.pitch_midi == 62
        assert m.onset_s == pytest.approx(2.0, abs=0.01)

    def test_load_ref_events_removed_maps_to_missed(self, midi_fixture):
        _, meta_path = midi_fixture
        events = ds.load_ref_events(str(meta_path))
        etypes = sorted(e.etype for e in events)
        assert etypes == ["correct", "correct", "extra", "missed"]
        # 'wrong' must NEVER be read from disk -- it is derived by collapse
        assert "wrong" not in etypes
        missed = [e for e in events if e.etype == "missed"][0]
        assert missed.pitch_midi == 62 and missed.piece == "id0"

    def test_loaded_events_score_end_to_end(self, midi_fixture):
        pred_dir, meta_path = midi_fixture
        pred = ds.load_pred_events(str(pred_dir))
        ref = ds.load_ref_events(str(meta_path))
        # pred errors match ref errors perfectly (same tracks written)
        r = ds.decoupled_scores(pred, ref, tau=0.05)
        assert r.n_localized == 2 and r.hm == 0.0
        assert r.localization["f1"] == 1.0
        assert r.mass_conserved()

    def test_unknown_track_name_raises(self, tmp_path):
        pred_dir = tmp_path / "pred"
        pred_dir.mkdir()
        _write_midi(pred_dir / "idX.mid", {"Bogus Track": [(1.0, 60)]})
        with pytest.raises(ValueError, match="unrecognized track name"):
            ds.load_pred_events(str(pred_dir))

    def test_empty_pred_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            ds.load_pred_events(str(empty))


# --------------------------------------------------------------------------- #
# 8. CLI contract + import hygiene.                                           #
# --------------------------------------------------------------------------- #

SCORER = os.path.join(HERE, "decoupled_scorer.py")


class TestCLI:

    def test_cli_end_to_end(self, midi_fixture, tmp_path):
        pred_dir, meta_path = midi_fixture
        out = tmp_path / "results.json"
        proc = subprocess.run(
            [sys.executable, SCORER,
             "--pred_dir", str(pred_dir), "--gt_meta", str(meta_path),
             "--tau", "50", "75", "100", "150", "200", "500",
             "--eps", "0.05", "--collapse", "strict",
             "--shipped", "--bootstrap", "50",
             "--out", str(out)],
            capture_output=True, text=True, cwd=HERE, timeout=300)
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["config"]["collapse"] == "strict"
        assert payload["config"]["epsilon_s"] == pytest.approx(0.05)
        assert payload["config"]["tau_s"] == pytest.approx(list(ds.TAU_GRID_S))
        per_tau = payload["decoupled"]["per_tau"]
        assert [r["tau_ms"] for r in per_tau] == [50, 75, 100, 150, 200, 500]
        assert all(r["mass_conserved"] for r in per_tau)
        assert per_tau[0]["localization"]["f1"] == 1.0
        assert "shipped_50ms" in payload
        assert payload["shipped_50ms"]["_mean_error_f1"] == pytest.approx(1.0)
        assert "bootstrap" in payload and "50" in payload["bootstrap"]

    def test_self_check_cli(self):
        proc = subprocess.run([sys.executable, SCORER, "--self-check"],
                              capture_output=True, text=True, cwd=HERE,
                              timeout=300)
        assert proc.returncode == 0, proc.stderr
        assert "SELF-CHECK PASSED" in proc.stdout

    def test_import_has_no_side_effects(self, tmp_path):
        """Importing the module must print nothing and write nothing."""
        proc = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); import decoupled_scorer"
             % HERE],
            capture_output=True, text=True, cwd=str(tmp_path), timeout=120)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == ""
        assert list(tmp_path.iterdir()) == []
