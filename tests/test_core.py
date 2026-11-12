import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portrait import compose as C
from portrait.face import FaceFinder, ANCHORS, NOSE_TIP

FACE_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


@pytest.fixture(scope="session")
def finder():
    return FaceFinder()


@pytest.fixture(scope="session")
def portrait_rgb():
    img = cv2.imread(str(Path(__file__).resolve().parent / "lena.jpg"))
    if img is None:
        img = cv2.imread(cv2.samples.findFile("lena.jpg"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def scene(portrait_rgb, scale, offset, canvas=(900, 1200)):
    """Ставит портрет в холст с заданным масштабом и смещением."""
    h, w = portrait_rgb.shape[:2]
    resized = cv2.resize(portrait_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    out = np.full((canvas[0], canvas[1], 3), 90, np.uint8)
    hh, ww = resized.shape[:2]
    x, y = offset
    out[y:y + hh, x:x + ww] = resized
    return out


def test_face_detected(finder, portrait_rgb):
    faces = finder.detect(portrait_rgb)
    assert len(faces) == 1
    assert faces[0].shape == (478, 2)


def test_alignment_recovers_known_transform(finder, portrait_rgb):
    base = scene(portrait_rgb, 1.0, (300, 150))
    donor = scene(portrait_rgb, 0.8, (50, 40))

    base_faces = finder.detect(base)
    donor_faces = finder.detect(donor)
    assert len(base_faces) == 1 and len(donor_faces) == 1

    matrix = C.estimate_alignment(donor_faces[0], base_faces[0])
    assert matrix is not None

    moved = C.__dict__ and cv2.transform(np.array([donor_faces[0][ANCHORS]], np.float64), matrix)[0]
    target = base_faces[0][ANCHORS]
    error = np.linalg.norm(moved - target, axis=1)
    assert error.mean() < 3.0, f"средняя ошибка выравнивания {error.mean():.2f}px"


def test_layer_moves_selection_to_face(finder, portrait_rgb):
    base = scene(portrait_rgb, 1.0, (300, 150))
    donor = scene(portrait_rgb, 0.8, (50, 40))
    base_faces = finder.detect(base)
    donor_faces = finder.detect(donor)

    face = donor_faces[0]
    center = face[:, :].mean(axis=0)
    span = float(np.linalg.norm(face[ANCHORS[1]] - face[ANCHORS[0]]))
    half = span * 1.4
    x0, y0 = int(center[0] - half), int(center[1] - half)
    x1, y1 = int(center[0] + half), int(center[1] + half)
    selection_box_x0, selection_box_y0 = x0, y0
    selection = np.zeros(donor.shape[:2], np.uint8)
    selection[y0:y1, x0:x1] = 255
    layer = C.build_layer(donor, selection, donor_faces, base_faces, base_point=base_faces[0][NOSE_TIP])
    assert layer.aligned == 1

    result, mode = layer.apply(base)
    assert result.shape == base.shape
    assert mode in ("poisson", "alpha")

    crop_origin = np.array([max(0, selection_box_x0 - C.CROP_MARGIN), max(0, selection_box_y0 - C.CROP_MARGIN)], np.float64)
    matrix = layer.pieces[0][1]
    moved = cv2.transform(np.array([face[ANCHORS] - crop_origin], np.float64), matrix)[0]
    error = np.linalg.norm(moved - base_faces[0][ANCHORS], axis=1)
    assert error.mean() < 3.0, f"лицо не совместилось: {error.mean():.2f}px"


def test_seam_is_smooth(finder, portrait_rgb):
    base = scene(portrait_rgb, 1.0, (300, 150))
    donor = scene(portrait_rgb, 0.8, (50, 40))
    base_faces, donor_faces = finder.detect(base), finder.detect(donor)
    selection = np.zeros(donor.shape[:2], np.uint8)
    face = donor_faces[0]
    c = face.mean(axis=0)
    span = float(np.linalg.norm(face[ANCHORS[1]] - face[ANCHORS[0]]))
    cv2.ellipse(selection, (int(c[0]), int(c[1])), (int(span), int(span * 1.3)), 0, 0, 360, 255, -1)

    layer = C.build_layer(donor, selection, donor_faces, base_faces, feather=14)
    result, _ = layer.apply(base)

    ys, xs = np.nonzero(selection)
    gradient = cv2.Laplacian(cv2.cvtColor(result, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    seam_zone = np.zeros_like(selection)
    edges = cv2.Canny(selection, 50, 150)
    seam_zone = cv2.dilate(edges, np.ones((9, 9), np.uint8))
    band = float(np.abs(gradient[seam_zone > 0]).mean())
    inner = float(np.abs(gradient[selection > 0]).mean())
    assert band < inner * 2.0 + 5, "на границе виден шов"


def test_refine_and_soft_alpha(portrait_rgb):
    mask = np.zeros(portrait_rgb.shape[:2], np.uint8)
    mask[100:300, 100:300] = 255
    refined = C.refine_edge(portrait_rgb, mask)
    assert refined.shape == mask.shape
    assert refined.any()
    alpha = C.soft_alpha(mask, 10)
    assert alpha.max() <= 1.0 and alpha.min() >= 0.0
    assert 0.0 < alpha[100 + 1, 200] < 1.0
