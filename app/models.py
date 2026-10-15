"""Model files and loaders."""
from pathlib import Path

import cv2
import onnxruntime as ort
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODELS = Path(__file__).resolve().parent.parent / "models"
LANDMARKER = MODELS / "face_landmarker.task"
YUNET = MODELS / "face_detection_yunet_2026may.onnx"
# name: (file, crop scale around the face box)
PASSIVE = {
    "v2": (MODELS / "minifasnet_v2.onnx", 2.7),
    "v1se": (MODELS / "minifasnet_v1se.onnx", 4.0),
}


def onnx(path):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def yunet():
    return cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.8)


def landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(LANDMARKER)),
        num_faces=2,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)
