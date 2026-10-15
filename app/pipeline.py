"""Per-frame analysis: face box, passive anti-spoof score, landmarks."""
import math

import cv2
import mediapipe as mp
import numpy as np

from app import models

REAL = 1  # upstream test.py: class 1 is a real face


def crop_box(src_w, src_h, box, scale):
    # Same expansion as upstream src/generate_patches.py CropImage._get_new_box
    x, y, box_w, box_h = box
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    cx, cy = x + box_w / 2, y + box_h / 2
    x1, y1, x2, y2 = cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > src_w - 1:
        x1 -= x2 - src_w + 1
        x2 = src_w - 1
    if y2 > src_h - 1:
        y1 -= y2 - src_h + 1
        y2 = src_h - 1
    return int(x1), int(y1), int(x2), int(y2)


class Passive:
    """YuNet face box + MiniFASNet V2 and V1SE, averaged."""

    def __init__(self):
        self.detector = models.yunet()
        self.nets = {name: (models.onnx(path), scale) for name, (path, scale) in models.PASSIVE.items()}

    def face_box(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[-1])[:4]

    @staticmethod
    def quality(img, box):
        """Brightness (0-255) and sharpness (Laplacian variance) of the face box."""
        x, y, w, h = (int(v) for v in box)
        face = cv2.cvtColor(img[max(y, 0):y + h, max(x, 0):x + w], cv2.COLOR_BGR2GRAY)
        if face.size == 0:
            return 0.0, 0.0
        return float(face.mean()), float(cv2.Laplacian(face, cv2.CV_64F).var())

    def score(self, img, box):
        h, w = img.shape[:2]
        scores = {}
        for name, (net, scale) in self.nets.items():
            x1, y1, x2, y2 = crop_box(w, h, box, scale)
            face = cv2.resize(img[y1:y2 + 1, x1:x2 + 1], (80, 80))
            x = face.astype(np.float32).transpose(2, 0, 1)[None]  # raw 0-255 BGR, NCHW
            logits = net.run(None, {net.get_inputs()[0].name: x})[0][0]
            p = np.exp(logits - logits.max())
            scores[name] = float(p[REAL] / p.sum())
        scores["real"] = sum(scores.values()) / len(scores)
        return scores


class Frame:
    """What the Face Landmarker sees in one frame. Positions are 0-1 of the frame."""

    def __init__(self, faces, cx=None, cy=None, size=None, yaw=None, pitch=None, blink=None):
        self.faces = faces
        self.cx, self.cy, self.size = cx, cy, size
        self.yaw, self.pitch = yaw, pitch
        self.blink = blink  # mean of eyeBlinkLeft and eyeBlinkRight


class Landmarks:
    """Face Landmarker on single frames: face position, head angle, blink score."""

    def __init__(self):
        self.landmarker = models.landmarker()

    def close(self):
        self.landmarker.close()

    def analyze(self, img):
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        r = self.landmarker.detect(rgb)
        if not r.face_landmarks:
            return Frame(faces=0)
        pts = r.face_landmarks[0]
        xs, ys = [p.x for p in pts], [p.y for p in pts]
        m = np.array(r.facial_transformation_matrixes[0])
        yaw = math.degrees(math.atan2(-m[2, 0], math.hypot(m[2, 1], m[2, 2])))
        pitch = math.degrees(math.atan2(m[2, 1], m[2, 2]))
        shapes = {c.category_name: c.score for c in r.face_blendshapes[0]}
        return Frame(
            faces=len(r.face_landmarks),
            cx=(min(xs) + max(xs)) / 2, cy=(min(ys) + max(ys)) / 2,
            size=max(max(xs) - min(xs), max(ys) - min(ys)),
            yaw=yaw, pitch=pitch,
            blink=(shapes["eyeBlinkLeft"] + shapes["eyeBlinkRight"]) / 2,
        )
