import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from .paths import resource

ANCHORS = [33, 263, 133, 362, 1, 61, 291, 152, 10,
           159, 386, 145, 374, 70, 300, 105, 334, 199, 234, 454, 127, 356]
NOSE_TIP = 1
DETECT_SIDE = 1600
TILE_MIN = 640


def _starts(length, tile, step):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)
    return starts


def _same_face(a, b):
    span = float(np.linalg.norm(a[ANCHORS[1]] - a[ANCHORS[0]]))
    return float(np.linalg.norm(a[NOSE_TIP] - b[NOSE_TIP])) < 0.5 * span


class FaceFinder:
    def __init__(self):
        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=resource("models/face_landmarker.task")),
            num_faces=12,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb):
        h, w = rgb.shape[:2]
        tile = max(TILE_MIN, min(h, w) // 2)
        step = tile // 2
        starts_y = _starts(h, tile, step)
        starts_x = _starts(w, tile, step)
        found = []
        for y in starts_y:
            for x in starts_x:
                crop = np.ascontiguousarray(rgb[y:y + tile, x:x + tile])
                for face in self._run(crop):
                    face = face + np.array([x, y], np.float64)
                    if not any(_same_face(face, kept) for kept in found):
                        found.append(face)
        return found

    def _run(self, rgb):
        h, w = rgb.shape[:2]
        scale = min(1.0, DETECT_SIDE / max(h, w))
        small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        sh, sw = small.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
        result = self._landmarker.detect(image)
        return [
            np.array([[p.x * sw, p.y * sh] for p in face], np.float64) / scale
            for face in result.face_landmarks
        ]

    def close(self):
        self._landmarker.close()
