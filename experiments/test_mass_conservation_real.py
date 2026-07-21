#!/usr/bin/env python3
"""Regression test proving mass_conserved() is no longer a tautology (v1.1.0).

Verifier finding R1 (2026-07-20): through v1.0.0 `spurious` was computed as
`len(unmatched_pred) - fa_on_correct`, so the prediction-side identity reduced
to `tp + (n - tp) == n` and held for every possible input — the guard at the
end of decoupled_scores() was dead code.

These tests assert the invariant is now falsifiable: perturbing any single
partition count breaks it. If someone reverts `spurious` to a subtraction,
test_invariant_is_falsifiable fails.
"""
import copy

import pytest

import decoupled_scorer as ds


def _mk(t, etype, pitch=60):
    return ds.Event(t, pitch, etype)


@pytest.fixture
def scored():
    """A corpus exercising all three prediction-side classes at once."""
    ref = [
        _mk(1.00, "extra"),      # -> matched (diagonal)
        _mk(2.00, "missed"),     # -> matched off-diagonal
        _mk(3.00, "wrong"),      # -> unmatched by pred (miss_margin)
        _mk(5.00, "correct"),    # -> target for a false alarm
    ]
    pred = [
        _mk(1.01, "extra"),      # matched, diagonal
        _mk(2.01, "wrong"),      # matched, off-diagonal
        _mk(5.01, "extra"),      # false alarm on a correct note
        _mk(9.00, "extra"),      # spurious (nothing nearby)
    ]
    return ds.decoupled_scores(pred, ref, tau=0.05)


def test_all_three_prediction_classes_populated(scored):
    """Guard against a corpus that trivially satisfies the invariant."""
    assert scored.confusion_total() > 0
    assert scored.false_alarm_on_correct > 0
    assert scored.spurious > 0
    assert scored.miss_margin > 0
    assert scored.mass_conserved()


@pytest.mark.parametrize(
    "field,delta",
    [
        ("false_alarm_on_correct", +1),
        ("false_alarm_on_correct", -1),
        ("spurious", +1),
        ("spurious", -1),
        ("miss_margin", +1),
        ("miss_margin", -1),
        ("n_pred_err", +1),
        ("n_ref_err", +1),
    ],
)
def test_invariant_is_falsifiable(scored, field, delta):
    """Perturbing any single partition count must break mass conservation.

    A tautological implementation passes regardless of these perturbations,
    so this is the test that would have caught R1.
    """
    broken = copy.deepcopy(scored)
    setattr(broken, field, getattr(broken, field) + delta)
    assert not broken.mass_conserved(), (
        "mass_conserved() returned True after perturbing %s by %+d — the "
        "invariant is tautological again (see verifier finding R1)" % (field, delta)
    )


def test_double_counted_prediction_is_caught(scored):
    """The realistic failure mode: a prediction counted as matched AND as a
    false alarm. Simulated by moving one unit from spurious into FA without
    changing the total."""
    broken = copy.deepcopy(scored)
    broken.false_alarm_on_correct += 1  # counted twice
    assert not broken.mass_conserved()


def test_sweep_key_renamed():
    ref = [_mk(1.0, "extra")]
    pred = [_mk(1.0, "extra")]
    sw = ds.sweep(pred, ref)
    assert "mean_loc_f1_over_tau_grid" in sw
    assert "mean_loc_f1" not in sw
