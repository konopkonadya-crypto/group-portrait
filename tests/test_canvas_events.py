import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from portrait.ui.canvas import Canvas

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(app):
    img = cv2.cvtColor(cv2.imread(str(HERE / "lena.jpg")), cv2.COLOR_BGR2RGB)
    widget = Canvas(editable=True)
    widget.resize(800, 800)
    widget.set_image(img, keep_view=False)
    widget.show()
    app.processEvents()
    return widget


def test_image_fits_window(canvas):
    assert canvas.zoom() > 0
    rect = canvas.image_to_screen(canvas.screen_to_image(__import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(100, 200)))
    assert abs(rect.x() - 100) < 1e-6 and abs(rect.y() - 200) < 1e-6


def test_brush_paints_where_clicked(canvas, app):
    canvas.set_tool("brush")
    canvas.set_brush_radius(20)
    start = canvas.image_to_screen(__import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(200, 200))
    end = canvas.image_to_screen(__import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(300, 260))

    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start.toPoint())
    QTest.mouseMove(canvas, end.toPoint())
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end.toPoint())
    app.processEvents()

    assert canvas.has_selection()
    painted = canvas.mask > 0
    assert painted[200, 200], "начало мазка не закрашено"
    assert painted[260, 300], "конец мазка не закрашен"
    assert not painted[20, 20], "закрашено вне мазка"
    xs = np.nonzero(painted)[1]
    assert xs.min() > 150 and xs.max() < 350


def test_brush_leaves_undo_snapshot(canvas, app):
    canvas.clear_mask()
    canvas.set_tool("brush")
    p = canvas.image_to_screen(__import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(250, 250)).toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p)
    app.processEvents()
    assert canvas.has_selection()
    assert canvas.undo_stroke()
    assert not canvas.has_selection()


def test_lasso_fills_polygon(canvas, app):
    canvas.clear_mask()
    canvas.set_tool("lasso")
    points = [(200, 200), (320, 210), (330, 330), (190, 320)]
    screen = [canvas.image_to_screen(__import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(x, y)).toPoint()
              for x, y in points]
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, screen[0])
    for point in screen[1:]:
        QTest.mouseMove(canvas, point)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, screen[-1])
    app.processEvents()

    assert canvas.has_selection()
    painted = canvas.mask > 0
    assert painted[260, 260], "центр лассо не закрашен"
    assert not painted[100, 100], "закрашено вне лассо"


def test_move_tool_requests_transfer(canvas, app):
    """Перенос запускается перетаскиванием выделения, как в Photoshop."""
    from PySide6.QtCore import QPoint, QPointF

    canvas.clear_mask()
    canvas.set_tool("brush")
    canvas.set_brush_radius(20)
    start = canvas.image_to_screen(QPointF(250, 250)).toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    app.processEvents()
    assert canvas.has_selection()

    calls = []
    canvas.transferRequested.connect(lambda: calls.append(1))
    canvas.set_tool("move")
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(canvas, start + QPoint(25, 15))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(25, 15))
    app.processEvents()
    assert calls == [1]


def test_reference_canvas_drags_layer(app):
    img = cv2.cvtColor(cv2.imread(str(HERE / "lena.jpg")), cv2.COLOR_BGR2RGB)
    ref = Canvas(editable=False)
    ref.resize(800, 800)
    ref.set_image(img, keep_view=False)
    ref.layer_drag = True
    ref.set_tool("move")
    ref.show()
    app.processEvents()

    moves = []
    ref.layerDragged.connect(lambda dx, dy: moves.append((dx, dy)))
    start = ref.image_to_screen(__import__("PySide6.QtCore", fromlist=["QPointF"]).QPointF(300, 300)).toPoint()
    QTest.mousePress(ref, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(ref, start + QPoint(40, 0))
    QTest.mouseRelease(ref, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start + QPoint(40, 0))
    app.processEvents()

    assert moves, "эталон не передал смещение слоя"
    assert any(dx > 0 for dx, _ in moves)


def test_zoom_keeps_point_under_cursor(canvas):
    from PySide6.QtCore import QPointF

    anchor = QPointF(400, 400)
    before = canvas.screen_to_image(anchor)
    canvas.set_zoom(canvas.zoom() * 1.5, anchor)
    after = canvas.screen_to_image(anchor)
    assert abs(before.x() - after.x()) < 0.01 and abs(before.y() - after.y()) < 0.01


def test_right_button_erases(canvas, app):
    from PySide6.QtCore import QPointF

    canvas.clear_mask()
    canvas.set_tool("brush")
    canvas.set_brush_radius(20)
    point = canvas.image_to_screen(QPointF(250, 250)).toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    assert canvas.mask[250, 250] > 0

    QTest.mousePress(canvas, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, point)
    QTest.mouseRelease(canvas, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, point)
    app.processEvents()
    assert canvas.mask[250, 250] == 0
