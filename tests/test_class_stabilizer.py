"""v3.4.2 per-track class stabilisation: label flicker suppressed, feature groups never changed, real changes accepted."""
from __future__ import annotations

import pytest

from src.detection.detector import Detection
from src.tracking.tracker import ClassStabilizer, class_group

TRUCK, BUS, CAR, PERSON = 7, 5, 2, 0


def run(seq, st=None, tid=1, start=0):
    st = st or ClassStabilizer()
    out = []
    for i, (c, conf) in enumerate(seq):
        d = Detection(start + i, c, "", conf, (0, 0, 1, 1), tid)
        st([d], start + i)
        out.append(d.cls)
    return out, st


def test_flicker_is_suppressed():
    seq = [(TRUCK, 0.8)] * 10 + [(BUS, 0.6), (TRUCK, 0.7), (BUS, 0.5), (CAR, 0.4), (TRUCK, 0.8)] * 4
    out, _ = run(seq)
    assert out == [TRUCK] * len(seq)


def test_persistent_change_is_accepted_after_about_one_second():
    out, _ = run([(CAR, 0.6)] * 30 + [(TRUCK, 0.6)] * 30)
    switch = out.index(TRUCK)
    assert 30 + 8 <= switch <= 30 + 13  # ~10 steps (1.0 s at 10 Hz) with decay 0.9, margin 2


def test_feature_groups_are_never_changed():
    seq = [(CAR, 0.9)] * 20 + [(PERSON, 0.3)] + [(CAR, 0.9)] * 5 + [(BUS, 0.4)]
    out, _ = run(seq)
    assert [class_group(o) for o in out] == [class_group(c) for c, _ in seq]  # cross-group flip passed through
    assert out[-1] == CAR  # within-group flicker smoothed


def test_tracks_are_independent_reset_and_forget():
    st = ClassStabilizer()
    run([(TRUCK, 0.9)] * 10, st, tid=1)
    out, _ = run([(BUS, 0.9)] * 3, st, tid=2, start=10)
    assert out == [BUS] * 3
    run([(TRUCK, 0.9)], st, tid=3, start=200)  # tracks 1 and 2 unseen for > 50 steps -> forgotten
    assert set(st.state) == {3}
    st.reset()
    assert st.state == {}
    with pytest.raises(ValueError):
        ClassStabilizer(mode="all")  # cross-group smoothing would change model features
