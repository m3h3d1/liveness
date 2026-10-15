"""Profile photo candidates from a passed session: frontal, eyes open, sharpest, not near-duplicates."""
import base64

import cv2

from app import challenge

COUNT = 3
MAX_TILT_DEG = 6.0      # yaw and pitch close to the starting pose
MAX_BLINK = 0.1         # eyes open: blink score close to the open-eye level
MIN_APART = 5           # frames between picks, so the photos differ
OUT_SIDE = 512
PREVIEW_SIDE = 160


def crop(img, box):
    """Square crop around the face with room for hair and shoulders."""
    x, y, w, h = box
    height, width = img.shape[:2]
    side = min(max(w, h) * 1.9, width, height)
    cx, cy = x + w / 2, y + h * 0.45
    x1 = int(min(max(cx - side / 2, 0), width - side))
    y1 = int(min(max(cy - side / 2, 0), height - side))
    out = img[y1:y1 + int(side), x1:x1 + int(side)]
    if out.shape[0] > OUT_SIDE:
        out = cv2.resize(out, (OUT_SIDE, OUT_SIDE), interpolation=cv2.INTER_AREA)
    return out


def candidates(images, frames, passive):
    base_yaw, base_pitch, base_blink = challenge.baseline(frames)
    scored = []
    for i, f in enumerate(frames):
        if (f.faces != 1 or abs(f.yaw - base_yaw) > MAX_TILT_DEG or abs(f.pitch - base_pitch) > MAX_TILT_DEG
                or f.blink > base_blink + MAX_BLINK):
            continue
        box = passive.face_box(images[i])
        if box is not None:
            scored.append((passive.quality(images[i], box)[1], i, box))
    picks = []
    for _, i, box in sorted(scored, key=lambda s: s[0], reverse=True):
        if all(abs(i - j) >= MIN_APART for j, _ in picks):
            picks.append((i, box))
        if len(picks) == COUNT:
            break
    return [crop(images[i], box) for i, box in sorted(picks)]


def jpeg(img, side=None):
    if side:
        img = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def preview(img):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg(img, PREVIEW_SIDE)).decode()
