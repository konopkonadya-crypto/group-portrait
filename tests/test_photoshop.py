import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from portrait.compose import build_layer, plan_faces
from portrait.face import FaceFinder
from portrait.ui.canvas import Canvas

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(app):
    img = cv2.cvtColor(cv2.imread(str(HERE / "lena.jpg")), cv2.COLOR_BGR2RGB)
    w = Canvas(editable=True)
    w.resize(800, 800)
    w.set_image(img, keep_view=False)
    w.show()
    app.processEvents()
    return w


def press_release(widget, point, modifier=Qt.KeyboardModifier.NoModifier, button=Qt.MouseButton.LeftButton):
    QTest.mousePress(widget, button, modifier, point)
    QTest.mouseRelease(widget, button, modifier, point)


def test_brush_alt_adds_and_ctrl_subtracts(canvas, app):
    canvas.clear_mask()
    canvas.set_tool("brush")
    canvas.set_brush_radius(18)
    a = canvas.image_to_screen(QPointF(200, 200)).toPoint()
    b = canvas.image_to_screen(QPointF(300, 300)).toPoint()

    press_release(canvas, a)
    assert canvas.mask[200, 200] > 0

    press_release(canvas, b, Qt.KeyboardModifier.AltModifier)
    assert canvas.mask[300, 300] > 0, "Alt не добавил к выделению"
    assert canvas.mask[200, 200] > 0, "Alt стёр прежнее выделение"

    press_release(canvas, a, Qt.KeyboardModifier.ControlModifier)
    assert canvas.mask[200, 200] == 0, "Ctrl не вычел из выделения"
    assert canvas.mask[300, 300] > 0, "Ctrl стёр лишнее"


def test_brush_replace_clears_previous(canvas, app):
    canvas.clear_mask()
    canvas.set_tool("brush")
    canvas.set_brush_radius(18)
    press_release(canvas, canvas.image_to_screen(QPointF(180, 180)).toPoint())
    press_release(canvas, canvas.image_to_screen(QPointF(320, 320)).toPoint())
    assert canvas.mask[320, 320] > 0
    assert canvas.mask[180, 180] == 0, "новое выделение должно заменять старое"


def test_lasso_alt_and_ctrl(canvas, app):
    canvas.clear_mask()
    canvas.set_tool("lasso")

    def draw(points, modifier):
        screen = [canvas.image_to_screen(QPointF(x, y)).toPoint() for x, y in points]
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, modifier, screen[0])
        for p in screen[1:]:
            QTest.mouseMove(canvas, p)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, modifier, screen[-1])
        app.processEvents()

    draw([(150, 150), (260, 160), (265, 265), (145, 258)], Qt.KeyboardModifier.NoModifier)
    assert canvas.mask[210, 210] > 0

    draw([(300, 300), (400, 305), (405, 405), (298, 400)], Qt.KeyboardModifier.AltModifier)
    assert canvas.mask[350, 350] > 0, "Alt-лассо не добавило область"
    assert canvas.mask[210, 210] > 0, "Alt-лассо стёрло прежнее"

    draw([(150, 150), (260, 160), (265, 265), (145, 258)], Qt.KeyboardModifier.ControlModifier)
    assert canvas.mask[210, 210] == 0, "Ctrl-лассо не вычло область"
    assert canvas.mask[350, 350] > 0


def test_brush_size_hotkeys(app):
    from PySide6.QtGui import QKeySequence
    from portrait.ui.main_window import MainWindow

    win = MainWindow()
    win.show()
    app.processEvents()
    start = win.ref_canvas.brush_radius
    win.ref_canvas.bump_brush(4)
    assert win.ref_canvas.brush_radius == start + 4
    win.ref_canvas.bump_brush(-8)
    assert win.ref_canvas.brush_radius == start - 4
    assert QKeySequence("Ъ").toString() != ""
    assert QKeySequence("Х").toString() != ""
    win.close()


def test_drag_selection_requests_transfer(canvas, app):
    canvas.clear_mask()
    canvas.set_tool("brush")
    canvas.set_brush_radius(20)
    press_release(canvas, canvas.image_to_screen(QPointF(250, 250)).toPoint())
    assert canvas.has_selection()

    calls = []
    canvas.transferRequested.connect(lambda: calls.append(1))
    canvas.set_tool("move")
    start = canvas.image_to_screen(QPointF(250, 250)).toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(canvas, start + __import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(30, 10))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                       start + __import__("PySide6.QtCore", fromlist=["QPoint"]).QPoint(30, 10))
    app.processEvents()
    assert calls, "перетаскивание выделения не вызвало перенос"
    assert canvas.has_selection(), "выделение пропало после перетаскивания"


def test_two_faces_go_to_two_targets():
    finder = FaceFinder()
    portrait = cv2.cvtColor(cv2.imread(str(HERE / "lena.jpg")), cv2.COLOR_BGR2RGB)
    small = cv2.resize(portrait, (300, 300))

    donor = np.full((700, 1100, 3), 90, np.uint8)
    donor[100:400, 100:400] = small
    donor[100:400, 600:900] = small
    work = np.full((700, 1100, 3), 90, np.uint8)
    work[150:450, 150:450] = small
    work[150:450, 650:950] = small

    donor_faces = finder.detect(donor)
    work_faces = finder.detect(work)
    assert len(donor_faces) == 2 and len(work_faces) == 2

    mask = np.zeros(donor.shape[:2], np.uint8)
    mask[100:400, 100:400] = 255
    mask[100:400, 600:900] = 255

    chosen, matches = plan_faces(mask, donor_faces, work_faces, np.array([0, 0]))
    assert len(chosen) == 2
    assert len(set(matches.values())) == 2, "оба лица ушли на одно место"

    layer = build_layer(donor, mask, donor_faces, work_faces, base_point=(300, 300))
    assert layer.faces == 2
    assert layer.aligned == 2, f"автовыравнивание сработало только для {layer.aligned} из 2"


def test_single_face_unchanged():
    finder = FaceFinder()
    portrait = cv2.cvtColor(cv2.imread(str(HERE / "lena.jpg")), cv2.COLOR_BGR2RGB)
    donor = np.full((700, 800, 3), 90, np.uint8)
    donor[150:550, 150:550] = cv2.resize(portrait, (400, 400))
    work = np.full((700, 800, 3), 90, np.uint8)
    work[120:520, 150:550] = cv2.resize(portrait, (400, 400))
    dfaces, wfaces = finder.detect(donor), finder.detect(work)
    mask = np.zeros(donor.shape[:2], np.uint8)
    mask[150:550, 150:550] = 255
    layer = build_layer(donor, mask, dfaces, wfaces, base_point=(350, 350))
    assert layer.faces == 1 and layer.aligned == 1
