"""Real stock clips (Pexels, 10 fps frames) in tests/clips/."""
from pathlib import Path

import cv2
import pytest

from app.challenge import continuity, find_events
from app.pipeline import Landmarks

CLIPS = Path(__file__).resolve().parent / "clips"


@pytest.fixture(scope="module")
def trace():
    lm = Landmarks()

    def run(name):
        paths = sorted((CLIPS / name).glob("*.jpg"))
        return [lm.analyze(cv2.imread(str(p))) for p in paths], [i / 10 for i in range(len(paths))]
    yield run
    lm.close()


def test_head_shake(trace):
    # 8627673: turns to his right first, then left; hands cross the face; camera zooms
    frames, times = trace("8627673")
    names = [e[0] for e in find_events(frames, times)]
    assert names[:2] == ["turn_right", "turn_left"]
    assert continuity(frames, times) == []


def test_blinks(trace):
    # 6620642: front-facing, blinks at frames 14-16 and 36-38
    frames, times = trace("blink")
    assert [e[0] for e in find_events(frames, times)].count("blink") >= 2


def test_frontal_has_no_turns(trace):
    # 6968225: front-facing, eye glances only
    frames, times = trace("6968225")
    assert not [e for e in find_events(frames, times) if e[0].startswith("turn")]
