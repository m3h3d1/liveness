from app.challenge import check, find_events
from app.decision import decide
from app.pipeline import Frame

FPS = 10


def clip(*parts):
    """parts: (frames, yaw, blink) segments -> frames and times at 10 fps."""
    frames = []
    for n, yaw, blink in parts:
        frames += [Frame(1, cx=0.5, cy=0.5, size=0.4, yaw=yaw, pitch=0.0, blink=blink) for _ in range(n)]
    return frames, [i / FPS for i in range(len(frames))]


STILL = (6, 0, 0.1)
LEFT = (5, 30, 0.1)
RIGHT = (5, -30, 0.1)
BLINK = (3, 0, 0.8)


def test_turn_left_then_blink_passes():
    frames, times = clip(STILL, LEFT, STILL, BLINK, STILL)
    assert check(["turn_left", "blink"], frames, times) == []


def test_direct_swing_left_to_right():
    # head passes the center between two frames, never within RETURN_DEG
    frames, times = clip(STILL, (4, 30, 0.1), (4, -40, 0.1), STILL)
    assert check(["turn_left", "turn_right"], frames, times) == []


def test_turn_still_held_at_end_counts():
    frames, times = clip(STILL, BLINK, STILL, LEFT)
    assert check(["blink", "turn_left"], frames, times) == []


def test_wrong_order():
    frames, times = clip(STILL, BLINK, STILL, LEFT, STILL)
    assert check(["turn_left", "blink"], frames, times) == ["CHALLENGE_ORDER"]


def test_wrong_side_is_missing():
    frames, times = clip(STILL, RIGHT, STILL, BLINK, STILL)
    assert check(["turn_left", "blink"], frames, times) == ["CHALLENGE_MISSING"]


def test_no_movement():
    frames, times = clip((40, 0, 0.1))
    assert check(["turn_right", "blink"], frames, times) == ["CHALLENGE_MISSING"]


def test_too_slow():
    frames, times = clip(STILL, LEFT, (160, 0, 0.1), BLINK, STILL)
    assert check(["turn_left", "blink"], frames, times) == ["TOO_SLOW"]


def test_weak_blink_with_glasses_counts():
    frames, times = clip(STILL, (2, 0, 0.38), STILL)   # rise 0.28 over open-eye 0.1
    assert [e[0] for e in find_events(frames, times)] == ["blink"]


def test_eye_noise_is_not_a_blink():
    frames, times = clip(STILL, (1, 0, 0.22), STILL)   # rise 0.12
    assert find_events(frames, times) == []


def test_looking_down_is_not_a_blink():
    frames, times = clip(STILL, (17, 0, 0.7), STILL)
    assert find_events(frames, times) == []


def test_blink_during_turn_is_not_counted():
    frames, times = clip(STILL, (1, 0, 0.8), (1, 15, 0.8), (1, 30, 0.8), STILL)
    assert [e[0] for e in find_events(frames, times)] == ["turn_left"]


def test_face_jump():
    frames, times = clip(STILL, LEFT, STILL, BLINK, STILL)
    frames[12] = Frame(1, cx=0.9, cy=0.5, size=0.4, yaw=0, pitch=0, blink=0.1)
    assert "FACE_CHANGED" in check(["turn_left", "blink"], frames, times)


def test_face_missing_too_long():
    frames, times = clip(STILL, LEFT, STILL, BLINK, STILL)
    for i in range(7, 15):
        frames[i] = Frame(0)
    assert "NO_FACE" in check(["turn_left", "blink"], frames, times)


def test_short_face_loss_during_turn_is_ok():
    frames, times = clip(STILL, LEFT, STILL, BLINK, STILL)
    frames[8] = frames[9] = Frame(0)
    assert check(["turn_left", "blink"], frames, times) == []


def test_second_face_needs_several_frames():
    frames, times = clip(STILL, LEFT, STILL, BLINK, STILL)
    frames[3] = Frame(2, cx=0.5, cy=0.5, size=0.4, yaw=0, pitch=0, blink=0.1)
    assert check(["turn_left", "blink"], frames, times) == []
    for i in (4, 5):
        frames[i] = Frame(2, cx=0.5, cy=0.5, size=0.4, yaw=0, pitch=0, blink=0.1)
    assert "MULTIPLE_FACES" in check(["turn_left", "blink"], frames, times)


def test_decide():
    assert decide([], 0.9) == ("PASS", [])
    assert decide([], 0.5) == ("UNCERTAIN", ["PASSIVE_BORDERLINE"])
    assert decide([], 0.1) == ("FAIL", ["PASSIVE_LOW"])
    assert decide(["CHALLENGE_ORDER"], 0.9) == ("FAIL", ["CHALLENGE_ORDER"])
