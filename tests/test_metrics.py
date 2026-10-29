import pytest

from coach.evaluation.metrics import angle_errors, feedback_counts, feedback_delays, precision_recall


def test_angle_errors_and_missed():
    rows = [{"t": 1.0, "reliable": True, "angle": 100.0},
            {"t": 2.0, "reliable": False, "angle": 150.0}]
    r = angle_errors(rows, [(1.02, 98.0), (2.0, 150.0), (5.0, 90.0)])
    assert r["errors"] == [2.0] and r["missed"] == 2


def test_feedback_pr():
    ev = [{"attempt": 1, "code": "extend_more"}, {"attempt": 2, "code": "hold_longer"},
          {"attempt": 3, "code": "good_rep"}]
    pt = {"1": ["extend_more"], "2": ["extend_more"], "3": []}
    c = feedback_counts(ev, pt)
    assert (c["tp"], c["fp"], c["fn"]) == (1, 1, 1)
    assert precision_recall(1, 1, 1) == (0.5, 0.5)


def test_delays():
    assert feedback_delays([{"attempt": 1, "code": "x", "t": 2.3}], {"1:x": 2.0}) == [pytest.approx(0.3)]
