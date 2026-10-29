import numpy as np
import pytest

from coach.exercises import make_exercise
from coach.filters import OneEuroFilter
from coach.geometry import angle_deg, lean_deg
from synth import knee_frame, ramp, run, shoulder_frame


def test_angle():
    assert angle_deg([1, 0], [0, 0], [0, 1]) == pytest.approx(90)
    assert angle_deg([1, 0], [0, 0], [-1, 0]) == pytest.approx(180)
    assert np.isnan(angle_deg([0, 0], [0, 0], [1, 1]))


def test_lean_sign():
    assert lean_deg((0, 0), (0, 1)) == pytest.approx(0)
    assert lean_deg((0.1, 0), (0, 1)) > 0


def test_one_euro_reduces_noise():
    rng = np.random.default_rng(0)
    f, out = OneEuroFilter(), []
    raw = 100 + rng.normal(0, 3, 200)
    for i, x in enumerate(raw):
        out.append(f(x, i / 30))
    assert np.std(out[30:]) < np.std(raw[30:]) / 2


def _codes(events):
    return [e.code for e in events]


def test_knee_good_reps_and_hold():
    ex = make_exercise("knee_extension")
    sig = [(0, 90), (1, 90), (2, 170), (5, 170), (6, 90), (7, 90),
           (8, 170), (11, 170), (12, 90), (13, 90)]
    ev = run(ex, *ramp(sig), knee_frame)
    assert ex.reps == 2 and ex.good_reps == 2
    assert _codes(ev) == ["good_rep", "good_rep"]


def test_knee_short_hold_and_incomplete():
    ex = make_exercise("knee_extension")
    sig = [(0, 90), (1, 90), (1.5, 170), (2.2, 170), (3, 90), (4, 90),
           (5, 140), (6, 90), (7, 90)]
    ev = run(ex, *ramp(sig), knee_frame)
    assert _codes(ev) == ["hold_longer", "extend_more"]
    assert ex.reps == 2 and ex.good_reps == 0


def test_knee_jitter_is_not_a_rep():
    ex = make_exercise("knee_extension")
    t, v = ramp([(0, 100), (1, 100)])
    v = v + 12 * np.sin(np.arange(len(v)) * 2)
    run(ex, t, v, knee_frame)
    assert ex.reps == 0


def test_knee_withholds_advice_when_unreliable():
    ex = make_exercise("knee_extension")
    for i in range(10):
        ex.update(i / 30, knee_frame(90))
    u = ex.update(0.4, knee_frame(170, vis=0.2))
    assert not u.reliable and u.events == [] and "unreliable" in u.message
    assert not ex.update(0.5, None).reliable


def test_tracking_jump_rejected():
    ex = make_exercise("knee_extension")
    for i in range(10):
        ex.update(i / 30, knee_frame(90))
    assert ex.update(0.4, knee_frame(170)).reason == "tracking jump"


def test_abort_gap_resets_phase():
    ex = make_exercise("knee_extension")
    t, v = ramp([(0, 90), (1, 90), (1.5, 150)])
    run(ex, t, v, knee_frame)
    assert ex.phase == "extending"
    for i in range(45):
        ex.update(1.5 + i / 30, None)
    assert ex.phase == "rest" and ex.reps == 0


def test_sit_to_stand_tempo():
    ex = make_exercise("sit_to_stand")
    sig = [(0, 95), (1, 95), (3, 170), (4, 170), (6, 95), (7, 95),   # good
           (8, 95), (8.3, 170), (9, 170), (11, 95), (12, 95),         # too fast up
           (13, 95), (14.5, 170), (15, 170), (15.4, 95), (16, 95)]    # too fast down
    ev = run(ex, *ramp(sig), knee_frame)
    assert _codes(ev) == ["good_rep", "ascent_too_fast", "descent_too_fast"]
    assert ex.reps == 3 and ex.good_reps == 1


def test_sit_to_stand_incomplete():
    ex = make_exercise("sit_to_stand")
    sig = [(0, 95), (1, 95), (2.5, 140), (4, 95), (5, 95)]
    ev = run(ex, *ramp(sig), knee_frame)
    assert _codes(ev) == ["stand_fully"] and ex.reps == 0


def test_shoulder_raise_ok_and_low():
    ex = make_exercise("shoulder_raise")
    sig = [(0, 5), (1, 5), (3, 90), (4, 90), (6, 5), (7, 5),
           (8, 50), (9, 5), (10, 5)]
    ev = run(ex, *ramp(sig), shoulder_frame)
    assert _codes(ev) == ["good_rep", "raise_higher"] and ex.reps == 2


def test_shoulder_trunk_lean_flagged_once():
    ex = make_exercise("shoulder_raise")
    t, v = ramp([(0, 5), (1, 5), (3, 90), (4, 90), (6, 5), (7, 5)])
    ev = []
    for ti, vi in zip(t, v):
        ev += ex.update(ti, shoulder_frame(vi, lean_deg=15 if 1.5 < ti < 5 else 0)).events
    assert _codes(ev).count("trunk_lean") == 1
    assert "good_rep" not in _codes(ev)
