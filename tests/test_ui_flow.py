import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from portrait.face import ANCHORS, NOSE_TIP

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def scene(portrait, scale, offset, canvas=(900, 1200), tint=(1.0, 1.0, 1.0)):
    h, w = portrait.shape[:2]
    r = cv2.resize(portrait, (int(w * scale), int(h * scale)))
    r = np.clip(r.astype(np.float32) * np.array(tint), 0, 255).astype(np.uint8)
    out = np.full((canvas[0], canvas[1], 3), 90, np.uint8)
    x, y = offset
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    from portrait.ui.main_window import MainWindow
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    portrait = cv2.cvtColor(cv2.imread(str(HERE / "lena.jpg")), cv2.COLOR_BGR2RGB)
    work = scene(portrait, 1.0, (300, 150))
    ref = scene(portrait, 0.85, (420, 210), tint=(0.8, 0.85, 1.0))
    paths = []
    for name, img in (("work.jpg", work), ("ref.jpg", ref)):
        p = tmp_path / name
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(p)
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    app.processEvents()
    w.open_project(w.catalog.create("Тест", paths))
    app.processEvents()
    return w


def test_panels_swapped(win):
    """Слева рабочее, справа эталон."""
    assert win.work_index == 0, "рабочий снимок должен быть первым слева"
    assert win.ref_index == 1, "эталон должен быть вторым справа"
    assert win.work_canvas.editable is False
    assert win.ref_canvas.editable is True


def test_thumbnail_click_switches_reference(win, app):
    """Клик по миниатюре меняет эталон, рабочий остаётся."""
    work_before = win.work_index
    win.thumbs._on_click(0)
    app.processEvents()
    assert win.ref_index == 0, "клик по миниатюре не переключил эталон"
    assert win.work_index == work_before, "клик по миниатюре сбросил рабочее фото"


def test_thumbnail_right_click_switches_work(win, app):
    win.thumbs.referenceRequested.emit(1)
    app.processEvents()
    assert win.work_index == 1, "правый клик не переключил рабочее фото"


def test_lasso_then_drag_transfers_by_mouse(win, app):
    """Полный путь: лассо по лицу справа, затем перетаскивание на рабочее."""
    win.set_work(0)
    win.set_reference(1)
    app.processEvents()

    ref = win.ref_rgb
    face = win.finder.detect(ref)[0]
    c = face[NOSE_TIP]
    span = float(np.linalg.norm(face[ANCHORS[1]] - face[ANCHORS[0]]))
    canvas = win.ref_canvas
    win._set_tool("lasso")

    def to_screen(x, y):
        return canvas.image_to_screen(QPointF(x, y)).toPoint()

    pts = [to_screen(c[0] - span * 1.2, c[1] - span * 1.5),
           to_screen(c[0] + span * 1.2, c[1] - span * 1.5),
           to_screen(c[0] + span * 1.2, c[1] + span * 1.3),
           to_screen(c[0] - span * 1.2, c[1] + span * 1.3)]
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pts[0])
    for p in pts[1:]:
        QTest.mouseMove(canvas, p)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, pts[-1])
    app.processEvents()
    assert canvas.has_selection(), "лассо не создало выделение"

    before = win.work_canvas.image().copy()
    win._set_tool("move")
    start = to_screen(c[0], c[1])
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(canvas, start + QPoint(-40, -20))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(-40, -20))
    app.processEvents()

    assert len(win.document.layers) == 1, "перенос мышью не создал слой"
    after = win.work_canvas.image()
    assert not np.array_equal(before, after), "рабочий снимок не изменился"
    assert win.layer.faces == 1 and win.layer.aligned == 1, "лицо не выровнено автоматически"


def test_face_lands_on_reference_face_position(win, app):
    """Перенесённое лицо должно совпасть с лицом на рабочем снимке."""
    win.set_work(0)
    win.set_reference(1)
    app.processEvents()
    ref_face = win.finder.detect(win.ref_rgb)[0]
    work_face = win.finder.detect(win.work_rgb)[0]
    c = ref_face[NOSE_TIP]
    span = float(np.linalg.norm(ref_face[ANCHORS[1]] - ref_face[ANCHORS[0]]))
    mask = np.zeros(win.ref_rgb.shape[:2], np.uint8)
    cv2.ellipse(mask, (int(c[0]), int(c[1])), (int(span * 1.2), int(span * 1.4)), 0, 0, 360, 255, -1)
    win.ref_canvas.set_mask(mask)
    win.ref_canvas._cursor_image = QPointF(0, 0)
    win.transfer_selection()
    app.processEvents()

    ox, oy = win.layer.origins[0]
    moved = cv2.transform(np.array([ref_face[ANCHORS] - np.array([ox, oy])], np.float64), win.layer.pieces[0][1])[0]
    error = np.linalg.norm(moved - work_face[ANCHORS], axis=1).mean()
    assert error < 6.0, f"лицо село мимо: ошибка {error:.1f}px"


def test_move_layer_on_work_side(win, app):
    """«Перемещение» двигает слой и на рабочем окне."""
    win.set_work(0)
    win.set_reference(1)
    ref_face = win.finder.detect(win.ref_rgb)[0]
    c = ref_face[NOSE_TIP]
    span = float(np.linalg.norm(ref_face[ANCHORS[1]] - ref_face[ANCHORS[0]]))
    mask = np.zeros(win.ref_rgb.shape[:2], np.uint8)
    cv2.ellipse(mask, (int(c[0]), int(c[1])), (int(span), int(span * 1.2)), 0, 0, 360, 255, -1)
    win.ref_canvas.set_mask(mask)
    win.transfer_selection()
    app.processEvents()

    dx0 = win.layer.dx
    work_canvas = win.work_canvas
    win._set_tool("move")
    p = work_canvas.image_to_screen(QPointF(c[0], c[1])).toPoint()
    QTest.mousePress(work_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p)
    QTest.mouseMove(work_canvas, p + QPoint(30, 0))
    QTest.mouseRelease(work_canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p + QPoint(30, 0))
    app.processEvents()
    assert abs(win.layer.dx - dx0) > 1, "слой не сдвинулся на рабочем окне"


def test_brush_alt_ctrl_on_reference(win, app):
    win.set_reference(1)
    canvas = win.ref_canvas
    canvas.clear_mask()
    win._set_tool("brush")
    canvas.set_brush_radius(20)
    a = canvas.image_to_screen(QPointF(300, 300)).toPoint()
    b = canvas.image_to_screen(QPointF(450, 450)).toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    assert canvas.mask[300, 300] > 0
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, b)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.AltModifier, b)
    app.processEvents()
    assert canvas.mask[450, 450] > 0, "Alt не добавил"
    assert canvas.mask[300, 300] > 0, "Alt стёр"
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, a)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, a)
    app.processEvents()
    assert canvas.mask[300, 300] == 0, "Ctrl не вычел"
    assert canvas.mask[450, 450] > 0


def test_export_uses_work_image(win, tmp_path, app):
    win.set_work(0)
    win.set_reference(1)
    ref_face = win.finder.detect(win.ref_rgb)[0]
    c = ref_face[NOSE_TIP]
    span = float(np.linalg.norm(ref_face[ANCHORS[1]] - ref_face[ANCHORS[0]]))
    mask = np.zeros(win.ref_rgb.shape[:2], np.uint8)
    cv2.ellipse(mask, (int(c[0]), int(c[1])), (int(span), int(span * 1.2)), 0, 0, 360, 255, -1)
    win.ref_canvas.set_mask(mask)
    win.transfer_selection()
    app.processEvents()

    out = tmp_path / "out.jpg"
    from portrait.imaging import save_jpeg
    save_jpeg(win.document.render(win.work_rgb), out)
    saved = cv2.imread(str(out))
    assert saved.shape == cv2.cvtColor(win.work_rgb, cv2.COLOR_RGB2BGR).shape, "экспорт не рабочего размера"
