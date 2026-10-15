from pathlib import Path

import cv2
import pytest

from app.pipeline import Passive

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(scope="module")
def passive():
    return Passive()


@pytest.mark.parametrize("name, is_real", [("image_T1", True), ("image_F1", False), ("image_F2", False)])
def test_upstream_samples(passive, name, is_real):
    img = cv2.imread(str(SAMPLES / f"{name}.jpg"))
    box = passive.face_box(img)
    assert box is not None
    assert (passive.score(img, box)["real"] >= 0.5) == is_real
