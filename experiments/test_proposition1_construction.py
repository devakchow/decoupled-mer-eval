"""Executable verification of the letter's Proposition 1 and its two corollaries.

The proofs are counting arguments over a planted instance. This module plants
that instance literally, runs the real coupled matcher (shipped_scores) and
the real decoupled scorer over it, and asserts what the proposition claims:

  * Prop. 1(b): for every integer X' in [0, X] the construction reproduces the
    published per-class (TP, FP, FN) exactly under the coupled match while
    the onset-only match localizes T + X' pairs, X' of them off-diagonal, so
    HM_0 = X'/(T + X'). Exhaustive over all published-count tuples with
    entries in {0, 1, 2} (T >= 1), plus seeded random tuples with entries up
    to 40, at both ends of the tolerance sweep.
  * Supplement "Conservativeness", post-collapse: planting the TP_k pairs as
    co-located missed/extra pairs on both sides merges min_k TP_k of them
    into wrong->wrong pairs and gives HM = X'/(max_k TP_k + X'), while the
    published counts are unchanged.
  * Inner bound: a collapse-free instance with X = 0 whose onset-only match
    prefers the nearer cross-class events, reaching HM_0 = 1.

Run: pytest experiments/test_proposition1_construction.py -q
"""
from __future__ import annotations

import itertools
import random
from fractions import Fraction

import pytest

import decoupled_scorer as ds

SPACING = 2.0        # grid spacing (s), far beyond every tau and epsilon
TAUS = (0.050, 0.500)
PITCH_A, PITCH_B = 60, 64


def plant(tp_m, fp_m, fn_m, tp_e, fp_e, fn_e, x_prime, colocate_tp=False):
    """The supplement's construction. Returns (pred, ref, T, X)."""
    a = min(fp_m, fn_e)
    b = min(fp_e, fn_m)
    X = a + b
    assert 0 <= x_prime <= X
    x_a = min(x_prime, a)
    x_b = x_prime - x_a
    assert x_b <= b
    pred, ref = [], []
    slot = [0]

    def t():
        slot[0] += 1
        return slot[0] * SPACING

    if colocate_tp:
        # TP_m and TP_e pairs share slots pairwise: each side then holds a
        # co-located (missed, extra) pair, which the collapse merges
        n_shared = min(tp_m, tp_e)
        for _ in range(n_shared):
            at = t()
            pred += [ds.Event(at, PITCH_A, "missed"), ds.Event(at, PITCH_B, "extra")]
            ref += [ds.Event(at, PITCH_A, "missed"), ds.Event(at, PITCH_B, "extra")]
        for _ in range(tp_m - n_shared):
            at = t(); pred.append(ds.Event(at, PITCH_A, "missed")); ref.append(ds.Event(at, PITCH_A, "missed"))
        for _ in range(tp_e - n_shared):
            at = t(); pred.append(ds.Event(at, PITCH_B, "extra")); ref.append(ds.Event(at, PITCH_B, "extra"))
    else:
        for _ in range(tp_m):
            at = t(); pred.append(ds.Event(at, PITCH_A, "missed")); ref.append(ds.Event(at, PITCH_A, "missed"))
        for _ in range(tp_e):
            at = t(); pred.append(ds.Event(at, PITCH_B, "extra")); ref.append(ds.Event(at, PITCH_B, "extra"))
    # (ii) cross-class slots: (predicted missed, reference extra) x_a times,
    #      (predicted extra, reference missed) x_b times
    for _ in range(x_a):
        at = t(); pred.append(ds.Event(at, PITCH_A, "missed")); ref.append(ds.Event(at, PITCH_B, "extra"))
    for _ in range(x_b):
        at = t(); pred.append(ds.Event(at, PITCH_B, "extra")); ref.append(ds.Event(at, PITCH_A, "missed"))
    # (iii) residual one-sided slots
    for _ in range(fp_m - x_a):
        pred.append(ds.Event(t(), PITCH_A, "missed"))
    for _ in range(fp_e - x_b):
        pred.append(ds.Event(t(), PITCH_B, "extra"))
    for _ in range(fn_m - x_b):
        ref.append(ds.Event(t(), PITCH_A, "missed"))
    for _ in range(fn_e - x_a):
        ref.append(ds.Event(t(), PITCH_B, "extra"))
    return pred, ref, tp_m + tp_e, X


def _published(pred, ref, tau):
    s = ds.shipped_scores(pred, ref, tau=tau)
    return tuple(int(s[k][f]) for k in ("missed", "extra") for f in ("tp", "fp", "fn"))


def _check_prop1b(counts, tau):
    tp_m, fp_m, fn_m, tp_e, fp_e, fn_e = counts
    T = tp_m + tp_e
    X = min(fp_m, fn_e) + min(fp_e, fn_m)
    for x_prime in range(X + 1):
        pred, ref, T_, X_ = plant(*counts, x_prime=x_prime)
        assert (T_, X_) == (T, X)
        # published counts reproduced exactly for every X'
        assert _published(pred, ref, tau) == (tp_m, fp_m, fn_m, tp_e, fp_e, fn_e), (counts, x_prime, tau)
        # (a): the published P/R/F1 are functions of (TP, FP, FN) alone
        s = ds.shipped_scores(pred, ref, tau=tau)
        for k in ("missed", "extra"):
            assert (s[k]["precision"], s[k]["recall"], s[k]["f1"]) == ds.prf(s[k]["tp"], s[k]["fp"], s[k]["fn"])
        # the instance is collapse-free: the strict collapse merges nothing
        for side in (pred, ref):
            assert len(ds.collapse_wrong(side, epsilon=ds.DEFAULT_EPSILON_S)) == len(side)
        # the onset-only match localizes T + X' pairs, X' off-diagonal
        r = ds.decoupled_scores(pred, ref, tau=tau, collapse="none")
        assert r.mass_conserved()
        assert r.n_localized == T + x_prime, (counts, x_prime, tau)
        assert r.off_diagonal() == x_prime, (counts, x_prime, tau)
        if T + x_prime > 0:
            assert Fraction(r.off_diagonal(), r.n_localized) == Fraction(x_prime, T + x_prime)
            assert abs(r.hm - x_prime / (T + x_prime)) < 1e-12
        # event totals do not depend on X'
        assert (r.n_pred_err, r.n_ref_err) == (tp_m + fp_m + tp_e + fp_e, tp_m + fn_m + tp_e + fn_e)


@pytest.mark.parametrize("tau", TAUS)
def test_prop1b_exhaustive_small_counts(tau):
    n = 0
    for counts in itertools.product(range(3), repeat=6):
        if counts[0] + counts[3] < 1:   # T >= 1 as in the proposition
            continue
        _check_prop1b(counts, tau)
        n += 1
    assert n == 3 ** 6 - 3 ** 4  # every tuple with TP_m + TP_e >= 1


@pytest.mark.parametrize("tau", TAUS)
def test_prop1b_random_large_counts(tau):
    rng = random.Random(20260906)
    for _ in range(60):
        counts = tuple(rng.randint(0, 40) for _ in range(6))
        if counts[0] + counts[3] < 1:
            continue
        _check_prop1b(counts, tau)


def test_prop1b_convex_hull_is_the_printed_interval():
    """The achievable set {X'/(T+X')} is increasing in X', so its hull is
    [0, X/(T+X)]; checked on the letter's own Polytune counts."""
    tp_m, fp_m, fn_m, tp_e, fp_e, fn_e = 7088, 24039, 15999, 27650, 18829, 6888
    T = tp_m + tp_e
    X = min(fp_m, fn_e) + min(fp_e, fn_m)
    assert (T, X) == (34738, 22887)
    vals = [Fraction(x, T + x) for x in range(X + 1)]
    assert vals == sorted(vals) and vals[0] == 0
    assert round(float(vals[-1]), 3) == 0.397


@pytest.mark.parametrize("counts", [(2, 1, 1, 3, 2, 1), (5, 3, 4, 2, 6, 2), (1, 0, 0, 4, 2, 2), (4, 2, 2, 4, 1, 1)])
def test_post_collapse_bound_construction(counts):
    """Supplement, Conservativeness: TP pairs planted as co-located pairs on
    both sides merge min_k TP_k of them into wrong->wrong, leaving
    max_k TP_k diagonal pairs beside the X' cross-class slots."""
    tp_m, fp_m, fn_m, tp_e, fp_e, fn_e = counts
    X = min(fp_m, fn_e) + min(fp_e, fn_m)
    for x_prime in range(X + 1):
        pred, ref, T, _ = plant(*counts, x_prime=x_prime, colocate_tp=True)
        # published (uncollapsed, coupled) counts unchanged by the co-location
        assert _published(pred, ref, 0.050) == counts
        r = ds.decoupled_scores(pred, ref, tau=0.050, collapse="strict")
        assert r.mass_conserved()
        assert r.confusion["wrong"]["wrong"] == min(tp_m, tp_e)
        assert r.n_localized == max(tp_m, tp_e) + x_prime
        assert r.off_diagonal() == x_prime
        assert Fraction(r.off_diagonal(), r.n_localized) == Fraction(x_prime, max(tp_m, tp_e) + x_prime)


def test_post_collapse_bound_polytune_value():
    tp_m, fp_m, fn_m, tp_e, fp_e, fn_e = 7088, 24039, 15999, 27650, 18829, 6888
    X = min(fp_m, fn_e) + min(fp_e, fn_m)
    assert round(X / (max(tp_m, tp_e) + X), 3) == 0.453


def test_inner_bound_counterexample_collapse_free():
    """X = 0 (both coupled matches perfect) yet HM_0 = 1: the onset-only
    matcher prefers the nearer cross-class events. Same-side events are 0.2 s
    apart, so the instance is collapse-free at epsilon = 50 ms; tau = 500 ms
    is the sweep's upper end."""
    pred = [ds.Event(0.0, PITCH_A, "missed"), ds.Event(0.2, PITCH_B, "extra")]
    ref = [ds.Event(0.1, PITCH_B, "extra"), ds.Event(0.3, PITCH_A, "missed")]
    tau = 0.500
    assert _published(pred, ref, tau) == (1, 0, 0, 1, 0, 0)        # X = 0
    for side in (pred, ref):
        assert len(ds.collapse_wrong(side, epsilon=ds.DEFAULT_EPSILON_S)) == 2
    r = ds.decoupled_scores(pred, ref, tau=tau, collapse="strict")
    assert r.mass_conserved() and r.n_localized == 2
    assert r.off_diagonal() == 2 and r.hm == 1.0                     # HM_0 = 1 > X/(T+X) = 0


def test_inner_bound_counterexample_collapse_none_at_50ms():
    """The same mechanism at tau = 50 ms under the collapse-free scorer
    (collapse='none'), which is the HM_0 of Proposition 1."""
    pred = [ds.Event(0.00, PITCH_A, "missed"), ds.Event(0.03, PITCH_B, "extra")]
    ref = [ds.Event(0.01, PITCH_B, "extra"), ds.Event(0.04, PITCH_A, "missed")]
    assert _published(pred, ref, 0.050) == (1, 0, 0, 1, 0, 0)
    r = ds.decoupled_scores(pred, ref, tau=0.050, collapse="none")
    assert r.n_localized == 2 and r.hm == 1.0
