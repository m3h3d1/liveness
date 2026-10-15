"""Challenge rules on a frame sequence: turns, blinks, order, timing, continuity.

Yaw sign: positive = the subject turned to their own left (checked on a stock
clip; confirm with the live page).
"""
from statistics import median

ACTIONS = ("turn_left", "turn_right", "blink")

BASELINE_FRAMES = 5    # frames used for the starting head angle and open-eye level
TURN_DEG = 20.0        # turn must move this far from the starting yaw
RETURN_DEG = 12.0      # ...and come back within this (or swing to the other side)
BLINK_RISE = 0.20      # blink score above the open-eye level (glasses give ~0.26-0.30)
BLINK_MAX_S = 0.6      # a real blink is short; looking down lasts longer
STILL_DEG = 10.0       # head steady during a blink (yaw and pitch range)
MAX_GAP_S = 0.6        # face may be lost this long (fast turns)
MIN_EXTRA_FACE_FRAMES = 3  # frames with a 2nd face before it counts
MAX_MOVE = 0.15        # face center move between frames (fraction of frame)
MAX_RESIZE = 0.20      # face size change between frames
MAX_TOTAL_S = 15.0


def baseline(frames):
    seen = [f for f in frames if f.faces][:BASELINE_FRAMES]
    return (median(f.yaw for f in seen), median(f.pitch for f in seen), median(f.blink for f in seen))


def find_turns(frames, times, base_yaw):
    """A turn ends when the head comes back near the start, swings to the other side, or the clip ends."""
    events, start, side = [], None, None
    for f, t in zip(frames, times):
        if not f.faces:
            continue
        d = f.yaw - base_yaw
        if side and (abs(d) <= RETURN_DEG or (d > 0) != (side == "turn_left")):
            events.append((side, start, t))
            side = None
        if side is None and abs(d) >= TURN_DEG:
            side, start = ("turn_left" if d > 0 else "turn_right"), t
    if side:
        events.append((side, start, times[-1]))
    return events


def find_blinks(frames, times, base_blink):
    events, run = [], []

    def spread(values):
        values = list(values)
        return max(values) - min(values)

    def close():
        if run and max(f.blink for f, _ in run) >= base_blink + BLINK_RISE:
            still = spread(f.yaw for f, _ in run) < STILL_DEG and spread(f.pitch for f, _ in run) < STILL_DEG
            if still and run[-1][1] - run[0][1] <= BLINK_MAX_S:
                events.append(("blink", run[0][1], run[-1][1]))

    for f, t in zip(frames, times):
        if f.faces and f.blink >= base_blink + BLINK_RISE / 2:
            run.append((f, t))
        else:
            close()
            run = []
    close()
    return events


def find_events(frames, times):
    base = baseline(frames)
    return sorted(find_turns(frames, times, base[0]) + find_blinks(frames, times, base[2]), key=lambda e: e[1])


def continuity(frames, times):
    """Reason codes for a missing, extra or jumping face; empty if fine."""
    reasons, lost_at, prev = set(), None, None
    if sum(f.faces > 1 for f in frames) >= MIN_EXTRA_FACE_FRAMES:
        reasons.add("MULTIPLE_FACES")
    for f, t in zip(frames, times):
        if not f.faces:
            lost_at = t if lost_at is None else lost_at
            if t - lost_at > MAX_GAP_S:
                reasons.add("NO_FACE")
            continue
        # After a short loss the face may have moved more; compare per elapsed frame.
        lost_at = None
        if prev and (abs(f.cx - prev.cx) > MAX_MOVE or abs(f.cy - prev.cy) > MAX_MOVE
                     or abs(f.size - prev.size) > MAX_RESIZE * prev.size):
            reasons.add("FACE_CHANGED")
        prev = f
    return sorted(reasons)


def check(actions, frames, times):
    """Reason codes for the challenge; empty list means it passed."""
    if not any(f.faces for f in frames):
        return ["NO_FACE"]
    reasons = continuity(frames, times)
    events = find_events(frames, times)
    names = [e[0] for e in events]
    i, matched = 0, []
    for e in events:
        if i < len(actions) and e[0] == actions[i]:
            matched.append(e)
            i += 1
    if i < len(actions):
        reasons.append("CHALLENGE_MISSING" if any(a not in names for a in actions) else "CHALLENGE_ORDER")
    elif matched[-1][2] - times[0] > MAX_TOTAL_S:
        reasons.append("TOO_SLOW")
    return reasons


def summary(frames, times):
    """Measured values, so a failed check can be explained."""
    seen = [f for f in frames if f.faces]
    out = {"frames": len(frames), "face_frames": len(seen)}
    if not seen:
        return out
    base_yaw, _, base_blink = baseline(frames)
    out.update(
        turn_left_deg=round(max(0.0, max(f.yaw - base_yaw for f in seen)), 1),
        turn_right_deg=round(max(0.0, max(base_yaw - f.yaw for f in seen)), 1),
        blink_rise=round(max(0.0, max(f.blink for f in seen) - base_blink), 2),
        seen=[e[0] for e in find_events(frames, times)],
    )
    return out

