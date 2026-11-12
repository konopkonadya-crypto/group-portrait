import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from portrait.face import ANCHORS, NOSE_TIP
from portrait.ui.main_window import MainWindow

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def scene(portrait_rgb, scale, offset, canvas=(900, 1200)):
    h, w = portrait_rgb.shape[:2]
    resized = cv2.resize(portrait_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    out = np.full((canvas[0], canvas[1], 3), 90, np.uint8)
    hh, ww = resized.shape[:2]
    x, y = offset
    out[y:y + hh, x:x + ww] = resized
    return out


@pytest.fixture
def photos(tmp_path):
    portrait = cv2.cvtColor(cv2.imread(str(HERE / "lena.jpg")), cv2.COLOR_BGR2RGB)
    ref = scene(portrait, 1.0, (300, 150))
    donor = scene(portrait, 0.8, (50, 40))
    paths = []
    for name, img in (("ref.jpg", ref), ("donor.jpg", donor)):
        path = tmp_path / name
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(path)
    return paths


def test_full_workflow(app, photos, tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    win = MainWindow()
    win.show()
    app.processEvents()
    try:
        pid = win.catalog.create("Тест", photos)
        win.open_project(pid)
        assert len(win.entries) == 2
        assert win.work_index == 0, "рабочее фото слева выбирается первым"
        assert win.ref_index == 1, "эталон справа — второй снимок"

        # Выделяем лицо на эталоне и переносим на рабочее
        ref = win.ref_rgb
        ref_face = win.finder.detect(ref)[0]
        center = ref_face.mean(axis=0)
        span = float(np.linalg.norm(ref_face[ANCHORS[1]] - ref_face[ANCHORS[0]]))
        mask = np.zeros(ref.shape[:2], np.uint8)
        cv2.ellipse(mask, (int(center[0]), int(center[1])), (int(span * 1.1), int(span * 1.4)), 0, 0, 360, 255, -1)
        win.ref_canvas.set_mask(mask)
        assert win.ref_canvas.has_selection()

        work_nose = win.finder.detect(win.work_rgb)[0][NOSE_TIP]
        win.ref_canvas._cursor_image = QPointF(float(work_nose[0]), float(work_nose[1]))
        win.transfer_selection()
        assert len(win.document.layers) == 1
        assert not win.ref_canvas.has_selection()

        win.undo()
        assert len(win.document.layers) == 0
        win.redo()
        assert len(win.document.layers) == 1

        win.drag_layer(10.0, 0.0)
        assert win.layer.dx == pytest.approx(10.0)

        out = tmp_path / "result.jpg"
        from portrait.imaging import save_jpeg
        save_jpeg(win.document.render(win.work_rgb, poisson=True), out)
        saved = cv2.imread(str(out))
        assert saved is not None and saved.shape == win.work_rgb.shape
        assert not np.array_equal(saved, cv2.cvtColor(win.work_rgb, cv2.COLOR_RGB2BGR))

        win.set_work(1)
        assert win.work_index == 1
        assert win.document.layers == [], "при смене рабочего снимка слои сбрасываются"
    finally:
        win.close()
