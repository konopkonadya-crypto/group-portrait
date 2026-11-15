"""Всё окно программы и холст в одном файле.

Так меньше точек отказа при обновлении: заменять нужно только этот файл
и portrait/compose.py.
"""

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..catalog import Catalog
from ..compose import build_layer, refine_edge
from ..document import Document
from ..face import FaceFinder
from ..imaging import load_rgb, save_jpeg
from ..paths import data_dir
from .thumbs import ThumbStrip

VERSION = "3.1"
MIN_ZOOM, MAX_ZOOM = 0.02, 16.0
BRUSH_MIN, BRUSH_MAX = 3, 400
INK = QColor(28, 30, 34)
ACCENT = QColor(45, 110, 200)

TOOL_HINTS = {
    "move": ("Перемещение", "Тянет выделение справа или готовый слой слева."),
    "brush": ("Кисть", "Левая рисует, правая стирает. Alt — добавить, Ctrl — вычесть."),
    "eraser": ("Ластик", "Стирает выделение справа или часть слоя слева."),
    "lasso": ("Лассо", "Обведите область, затем тяните её на рабочее фото."),
    "zoom": ("Лупа", "Клик приближает, Alt+клик отдаляет."),
    "view": ("Просмотр", "Двигает вид, не трогая снимок."),
}

HELP_HTML = """
<h2>Как работать</h2>
<ol>
<li><b>Рабочее фото слева</b> — клик по миниатюре. К нему применяются правки.</li>
<li><b>Эталон справа</b> — клик по миниатюре. С него берём лица.</li>
<li><b>Выделение на эталоне.</b> Кисть (B) или лассо (L). Alt — добавить, Ctrl — вычесть.</li>
<li><b>Перенос.</b> После обводки лассо инструмент сам перейдёт в «Перемещение».
Тяните выделение на рабочее фото.</li>
<li><b>Проверка.</b> Слой появится в списке справа. Нажмите «Подтвердить перенос».</li>
<li><b>Правка.</b> Ползунки, перетаскивание слоя слева, ластик (E) для чистки края.</li>
<li><b>Экспорт.</b> Ctrl+E — рабочее фото в JPEG 95.</li>
</ol>
<h3>Клавиши</h3>
<table cellpadding="4">
<tr><td><b>V</b></td><td>Перемещение</td></tr>
<tr><td><b>B</b></td><td>Кисть</td></tr>
<tr><td><b>E</b></td><td>Ластик</td></tr>
<tr><td><b>L</b></td><td>Лассо</td></tr>
<tr><td><b>Z</b></td><td>Лупа</td></tr>
<tr><td><b>H</b></td><td>Просмотр</td></tr>
<tr><td><b>Ъ / Х</b></td><td>Размер кисти</td></tr>
<tr><td><b>Ctrl+O / Ctrl+E</b></td><td>Импорт / экспорт</td></tr>
<tr><td><b>Ctrl+Z / Ctrl+Y</b></td><td>Отменить / повторить</td></tr>
<tr><td><b>Ctrl+0 / Ctrl+1</b></td><td>Вписать / 100%</td></tr>
<tr><td><b>Delete</b></td><td>Снять выделение</td></tr>
<tr><td><b>F1</b></td><td>Справка</td></tr>
</table>
<h3>Несколько лиц</h3>
<p>Обведите двух человек одним лассо — программа разделит выделение между лицами
и посадит каждое на своё место рабочего снимка.</p>
"""


def to_qimage(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def rgba_to_qimage(rgba):
    rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    return QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


def draw_icon(kind, size=26):
    """Чёрные иконки: панель светлая, поэтому светлые значки не видны."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    s = size / 32.0

    def sc(v):
        return v * s

    if kind == "move":
        p.setPen(QPen(INK, sc(2.2), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(sc(16), sc(6)), QPointF(sc(16), sc(26)))
        p.drawLine(QPointF(sc(6), sc(16)), QPointF(sc(26), sc(16)))
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            tip = QPointF(sc(16 + 10 * dx), sc(16 + 10 * dy))
            base = QPointF(sc(16 + 5 * dx), sc(16 + 5 * dy))
            p.drawLine(tip, QPointF(base.x() - sc(3) * dy, base.y() + sc(3) * dx))
            p.drawLine(tip, QPointF(base.x() + sc(3) * dy, base.y() - sc(3) * dx))
    elif kind == "brush":
        p.setPen(QPen(INK, sc(2.8), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(sc(8), sc(25)), QPointF(sc(19), sc(13)))
        p.setPen(QPen(ACCENT, sc(2.8), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(sc(20), sc(12)), QPointF(sc(24), sc(7)))
        p.setBrush(QBrush(INK))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(sc(26), sc(5)), sc(1.8), sc(1.8))
    elif kind == "eraser":
        p.save()
        p.translate(sc(16), sc(15))
        p.rotate(-35)
        p.setPen(QPen(INK, sc(2.1)))
        p.setBrush(QBrush(QColor(250, 250, 250)))
        p.drawRoundedRect(QRectF(sc(-9), sc(-5.5), sc(18), sc(11)), sc(2), sc(2))
        p.setBrush(QBrush(QColor(150, 155, 165)))
        p.drawRect(QRectF(sc(-9), sc(-5.5), sc(7), sc(11)))
        p.restore()
        p.setPen(QPen(INK, sc(1.8), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(sc(6), sc(27)), QPointF(sc(26), sc(27)))
    elif kind == "lasso":
        p.setPen(QPen(INK, sc(2.1)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(QPointF(sc(16), sc(6)))
        path.cubicTo(sc(26), sc(6), sc(28), sc(15), sc(20), sc(19))
        path.cubicTo(sc(13), sc(22), sc(6), sc(20), sc(6), sc(14))
        path.cubicTo(sc(6), sc(9), sc(12), sc(6), sc(17), sc(7))
        p.drawPath(path)
        p.setPen(QPen(ACCENT, sc(2.1), Qt.PenStyle.DashLine))
        p.drawLine(QPointF(sc(19), sc(19)), QPointF(sc(16), sc(26)))
    elif kind == "zoom":
        p.setPen(QPen(INK, sc(2.6)))
        p.setBrush(QBrush(QColor(255, 255, 255, 140)))
        p.drawEllipse(QPointF(sc(13.5), sc(13.5)), sc(7.0), sc(7.0))
        p.setPen(QPen(INK, sc(3.2), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(sc(19), sc(19)), QPointF(sc(26), sc(26)))
        p.setPen(QPen(ACCENT, sc(1.8)))
        p.drawLine(QPointF(sc(10), sc(13.5)), QPointF(sc(17), sc(13.5)))
        p.drawLine(QPointF(sc(13.5), sc(10)), QPointF(sc(13.5), sc(17)))
    elif kind == "view":
        p.setPen(QPen(INK, sc(2.1)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(QPointF(sc(4), sc(16)))
        path.cubicTo(sc(11), sc(7), sc(21), sc(7), sc(28), sc(16))
        path.cubicTo(sc(21), sc(25), sc(11), sc(25), sc(4), sc(16))
        p.drawPath(path)
        p.setBrush(QBrush(ACCENT))
        p.setPen(QPen(INK, sc(1.3)))
        p.drawEllipse(QPointF(sc(16), sc(16)), sc(3.4), sc(3.4))
    p.end()
    return QIcon(pix)


def make_cursor(kind):
    """Курсор в виде инструмента."""
    size = 32
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    ink, white = QColor(25, 27, 30), QColor(255, 255, 255)

    def outline(width, draw):
        for color, w in ((white, width + 2.4), (ink, width)):
            p.setPen(QPen(color, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            draw()

    hotspot = (4, 4)
    if kind == "move":
        def arrows():
            p.drawLine(QPointF(16, 3), QPointF(16, 29))
            p.drawLine(QPointF(3, 16), QPointF(29, 16))
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                tip = QPointF(16 + 11 * dx, 16 + 11 * dy)
                base = QPointF(16 + 6 * dx, 16 + 6 * dy)
                p.drawLine(tip, QPointF(base.x() - 3.4 * dy, base.y() + 3.4 * dx))
                p.drawLine(tip, QPointF(base.x() + 3.4 * dy, base.y() - 3.4 * dx))
        p.setBrush(Qt.BrushStyle.NoBrush)
        outline(1.9, arrows)
        hotspot = (16, 16)
    elif kind == "brush":
        def brush():
            p.drawLine(QPointF(6, 26), QPointF(23, 9))
            p.drawLine(QPointF(23, 9), QPointF(26, 6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        outline(2.8, brush)
        hotspot = (6, 26)
    elif kind == "eraser":
        def eraser():
            p.drawRoundedRect(QRectF(8, 15, 19, 10), 2, 2)
        p.setBrush(QBrush(QColor(245, 245, 245)))
        outline(1.9, eraser)
        p.setBrush(Qt.BrushStyle.NoBrush)
        hotspot = (10, 20)
    elif kind == "lasso":
        def lasso():
            path = QPainterPath(QPointF(16, 5))
            path.cubicTo(28, 5, 29, 16, 19, 20)
            path.cubicTo(10, 24, 3, 21, 5, 14)
            path.cubicTo(6, 8, 13, 5, 18, 7)
            p.drawPath(path)
            p.drawLine(QPointF(18, 20), QPointF(15, 28))
        p.setBrush(Qt.BrushStyle.NoBrush)
        outline(1.8, lasso)
        hotspot = (16, 5)
    elif kind == "zoom":
        def lens():
            p.drawEllipse(QPointF(13, 13), 8, 8)
            p.drawLine(QPointF(19, 19), QPointF(28, 28))
        p.setBrush(QBrush(QColor(255, 255, 255, 100)))
        outline(2.0, lens)
        hotspot = (13, 13)
    else:
        hotspot = (0, 0)
    p.end()
    return QCursor(pix, hotspot[0], hotspot[1])


class Canvas(QWidget):
    selectionChanged = Signal()
    transferRequested = Signal()
    layerDragStarted = Signal()
    layerDragged = Signal(float, float)
    layerErased = Signal(float, float, float)
    zoomChanged = Signal(float)
    strokeFinished = Signal()
    brushChanged = Signal(int)

    def __init__(self, editable=False, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 160)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAutoFillBackground(False)

        self._image = None
        self._qimage = None
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._auto_fit = True

        self.editable = editable
        self.layer_drag = False
        self._dragging_layer = False
        self._erasing_layer = False
        self.tool = "move"
        self.brush_radius = 24
        self.select_mode = "replace"
        self.mask = None
        self._mask_qimage = None
        self._strokes = []
        self._sketch = None
        self._painting = False
        self._panning = False
        self._last_screen = None
        self._cursor_image = QPointF(0, 0)
        self._drag_start = None
        self._drag_mask0 = None
        self.drag_preview = None
        self._paint_mode = "replace"
        self._apply_cursor()

    # ---------- данные ----------
    def set_image(self, rgb, keep_view=True):
        previous = (self._zoom, self._offset) if (keep_view and self._image is not None) else None
        self._image = rgb
        self._qimage = to_qimage(rgb) if rgb is not None else None
        self.mask = np.zeros(rgb.shape[:2], np.uint8) if rgb is not None else None
        self._strokes = []
        self._mask_qimage = None
        self.drag_preview = None
        if previous is not None and rgb is not None:
            self._zoom, self._offset = previous
            self._auto_fit = False
        else:
            self._auto_fit = True
            self._fit()
        self.update()

    def has_image(self):
        return self._image is not None

    def image(self):
        return self._image

    def clear_mask(self):
        if self.mask is not None:
            self.mask[:] = 0
            self._strokes = []
            self._mask_qimage = None
            self.selectionChanged.emit()
            self.update()

    def set_mask(self, mask):
        self.mask = mask
        self._mask_qimage = None
        self._strokes = []
        self.selectionChanged.emit()
        self.update()

    def has_selection(self):
        return self.mask is not None and bool(self.mask.any())

    def set_brush_radius(self, value):
        value = int(max(BRUSH_MIN, min(BRUSH_MAX, value)))
        if value == self.brush_radius:
            return
        self.brush_radius = value
        self.brushChanged.emit(value)
        self.update()

    def bump_brush(self, delta):
        self.set_brush_radius(self.brush_radius + delta)

    def set_select_mode(self, mode):
        self.select_mode = mode

    def _apply_cursor(self):
        if self.tool in ("brush", "eraser", "lasso", "zoom", "move"):
            self.setCursor(make_cursor(self.tool))
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_tool(self, tool):
        self.tool = tool
        self._apply_cursor()
        self.update()

    # ---------- вид ----------
    def zoom(self):
        return self._zoom

    def _fit(self):
        if self._image is None or self.width() <= 0 or self.height() <= 0:
            return
        h, w = self._image.shape[:2]
        if w <= 0 or h <= 0:
            return
        self._zoom = min(self.width() / w, self.height() / h) * 0.985
        self._offset = QPointF(
            (self.width() - w * self._zoom) / 2.0,
            (self.height() - h * self._zoom) / 2.0,
        )
        self.zoomChanged.emit(self._zoom)

    def fit(self):
        self._auto_fit = True
        self._fit()
        self.update()

    def set_zoom(self, value, anchor=None):
        if self._image is None:
            return
        value = max(MIN_ZOOM, min(MAX_ZOOM, value))
        anchor = anchor or QPointF(self.width() / 2, self.height() / 2)
        before = self.screen_to_image(anchor)
        self._zoom = value
        after = self.image_to_screen(before)
        self._offset += anchor - after
        self._auto_fit = False
        self.zoomChanged.emit(self._zoom)
        self.update()

    def zoom_step(self, factor, anchor=None):
        self.set_zoom(self._zoom * factor, anchor)

    def image_to_screen(self, point):
        return QPointF(point.x() * self._zoom + self._offset.x(), point.y() * self._zoom + self._offset.y())

    def screen_to_image(self, point):
        if self._zoom <= 0:
            return QPointF(0, 0)
        return QPointF((point.x() - self._offset.x()) / self._zoom, (point.y() - self._offset.y()) / self._zoom)

    def cursor_image(self):
        return self._cursor_image

    # ---------- отрисовка ----------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image is not None and self._auto_fit:
            self._fit()
            self.update()

    def showEvent(self, event):
        super().showEvent(event)
        if self._image is not None and self._auto_fit:
            self._fit()

    def _mask_layer(self):
        if self._mask_qimage is None:
            tint = self.mask > 127
            rgba = np.zeros((*self.mask.shape, 4), np.uint8)
            rgba[tint] = (70, 150, 255, 110)
            self._mask_qimage = rgba_to_qimage(rgba)
        return self._mask_qimage

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(46, 48, 53))
        if self._qimage is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Перетащите снимок сюда")
            return

        w = self._image.shape[1] * self._zoom
        h = self._image.shape[0] * self._zoom
        target = QRectF(self._offset.x(), self._offset.y(), w, h)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(target, self._qimage)

        if self.drag_preview is not None:
            rgba, pos = self.drag_preview
            painter.setOpacity(0.7)
            painter.drawImage(
                QRectF(self._offset.x() + pos[0] * self._zoom, self._offset.y() + pos[1] * self._zoom,
                       rgba.shape[1] * self._zoom, rgba.shape[0] * self._zoom),
                rgba_to_qimage(rgba),
            )
            painter.setOpacity(1.0)

        if self.mask is not None and self.mask.any():
            painter.drawImage(target, self._mask_layer())

        if self._sketch is not None and len(self._sketch) > 1:
            painter.setPen(QPen(QColor(255, 190, 40), 1.8, Qt.PenStyle.DashLine))
            path = QPainterPath(self._sketch[0])
            for point in self._sketch[1:]:
                path.lineTo(point)
            painter.drawPath(path)

        if self.editable and self.tool in ("brush", "eraser"):
            center = self.image_to_screen(self._cursor_image)
            radius = max(3.0, self.brush_radius * self._zoom)
            color = QColor(240, 90, 90, 235) if self.tool == "eraser" else QColor(255, 255, 255, 235)
            painter.setPen(QPen(color, 1.4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, radius, radius)

    # ---------- вспомогательное ----------
    def _mode_for(self, event):
        mods = event.modifiers() | QApplication.keyboardModifiers()
        if mods & Qt.KeyboardModifier.AltModifier:
            return "add"
        if mods & Qt.KeyboardModifier.ControlModifier:
            return "subtract"
        return self.select_mode

    def _make_drag_preview(self):
        if self.mask is None or self._image is None or not self.mask.any():
            self.drag_preview = None
            return
        ys, xs = np.nonzero(self.mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        sub = self.mask[y0:y1, x0:x1]
        rgba = np.zeros((*sub.shape, 4), np.uint8)
        rgba[..., :3] = self._image[y0:y1, x0:x1]
        rgba[..., 3] = sub
        self.drag_preview = (rgba, (x0, y0))
        self.update()

    def _paint_dot(self, point, mode):
        value = 0 if mode == "subtract" else 255
        cv2.circle(self.mask, (int(round(point.x())), int(round(point.y()))), self.brush_radius, value, -1)
        self._mask_qimage = None
        self.selectionChanged.emit()
        self.update()

    # ---------- мышь ----------
    def mousePressEvent(self, event):
        if self._image is None:
            return
        pos = event.position()
        self._last_screen = pos
        self._cursor_image = self.screen_to_image(pos)
        button = event.button()

        if button == Qt.MouseButton.MiddleButton:
            self._panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if button == Qt.MouseButton.RightButton and self.tool in ("brush", "eraser") and self.editable:
            self._painting = True
            self._paint_mode = "subtract"
            self._strokes.append(self.mask.copy())
            self._paint_dot(self._cursor_image, "subtract")
            return

        if button != Qt.MouseButton.LeftButton:
            return

        if self.tool == "move":
            if self.layer_drag:
                self._dragging_layer = True
                self.layerDragStarted.emit()
            elif self.editable and self.has_selection():
                self._drag_start = self._cursor_image
                self._drag_mask0 = self.mask.copy()
                self._make_drag_preview()
            else:
                self._panning = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self.tool == "view":
            self._panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self.tool == "zoom":
            factor = 0.8 if (event.modifiers() & Qt.KeyboardModifier.AltModifier) else 1.25
            self.zoom_step(factor, pos)
            return

        if not self.editable:
            return

        if self.tool == "eraser":
            if self.layer_drag:
                self._erasing_layer = True
                self.layerErased.emit(self._cursor_image.x(), self._cursor_image.y(), float(self.brush_radius))
            elif self.mask is not None:
                self._painting = True
                self._paint_mode = "subtract"
                self._strokes.append(self.mask.copy())
                self._paint_dot(self._cursor_image, "subtract")
            return

        if self.tool == "brush" and self.mask is not None:
            self._painting = True
            self._paint_mode = self._mode_for(event)
            self._strokes.append(self.mask.copy())
            if self._paint_mode == "replace":
                self.mask[:] = 0
                self._mask_qimage = None
            self._paint_dot(self._cursor_image, self._paint_mode)
            return

        if self.tool == "lasso":
            self._strokes.append(self.mask.copy())
            self._lasso_mode = self._mode_for(event)
            self._sketch = [pos]

    def mouseMoveEvent(self, event):
        if self._image is None:
            return
        pos = event.position()
        self._cursor_image = self.screen_to_image(pos)

        if self._panning and self._last_screen is not None:
            self._offset += pos - self._last_screen
            self._last_screen = pos
            self.update()
            return

        if self._dragging_layer and self._last_screen is not None:
            delta = pos - self._last_screen
            self.layerDragged.emit(delta.x() / self._zoom, delta.y() / self._zoom)
            self._last_screen = pos
            return

        if self._erasing_layer:
            self.layerErased.emit(self._cursor_image.x(), self._cursor_image.y(), float(self.brush_radius))
            return

        if self._drag_start is not None and self._drag_mask0 is not None:
            current = self._cursor_image
            dx = int(round(current.x() - self._drag_start.x()))
            dy = int(round(current.y() - self._drag_start.y()))
            shift = np.float32([[1, 0, dx], [0, 1, dy]])
            self.mask = cv2.warpAffine(self._drag_mask0, shift, (self.mask.shape[1], self.mask.shape[0]),
                                       flags=cv2.INTER_NEAREST, borderValue=0)
            self._mask_qimage = None
            if self.drag_preview is not None:
                rgba, (x0, y0) = self.drag_preview
                self.drag_preview = (rgba, (x0 + dx, y0 + dy))
                self._drag_start = current
            self.selectionChanged.emit()
            self.update()
            return

        if not self.editable:
            self.update()
            return

        if self._painting:
            start = self.screen_to_image(self._last_screen)
            value = 0 if self._paint_mode == "subtract" else 255
            cv2.line(self.mask, (int(start.x()), int(start.y())),
                     (int(self._cursor_image.x()), int(self._cursor_image.y())), (value,), self.brush_radius * 2)
            self._mask_qimage = None
            self.selectionChanged.emit()
        elif self._sketch is not None:
            self._sketch.append(pos)
        self._last_screen = pos
        self.update()

    def mouseReleaseEvent(self, event):
        if self._dragging_layer:
            self._dragging_layer = False
            self._last_screen = None
            return
        if self._erasing_layer:
            self._erasing_layer = False
            return
        if self._drag_start is not None:
            self.drag_preview = None
            self._drag_start = None
            if self._drag_mask0 is not None:
                self.mask = self._drag_mask0.copy()
                self._mask_qimage = None
            self._drag_mask0 = None
            self.selectionChanged.emit()
            self.update()
            self.transferRequested.emit()
            return
        if self._panning:
            self._panning = False
            self._apply_cursor()
        if self._painting:
            self._painting = False
            self.strokeFinished.emit()
        if self._sketch is not None and len(self._sketch) > 2:
            self._fill_lasso()
        self._sketch = None
        self._last_screen = None
        self.update()

    def _fill_lasso(self):
        points = np.array([[p.x(), p.y()] for p in self._sketch], np.float64)
        points = (points - np.array([self._offset.x(), self._offset.y()])) / self._zoom
        polygon = np.round(points).astype(np.int32)
        mode = getattr(self, "_lasso_mode", "replace")
        if mode == "add":
            cv2.fillPoly(self.mask, [polygon], 255)
        elif mode == "subtract":
            cv2.fillPoly(self.mask, [polygon], 0)
        else:
            self.mask[:] = 0
            cv2.fillPoly(self.mask, [polygon], 255)
        self._mask_qimage = None
        self.selectionChanged.emit()
        self.strokeFinished.emit()

    def wheelEvent(self, event):
        if self._image is None:
            return
        step = event.angleDelta().y() / 120.0
        if step:
            self.zoom_step(1.15 ** step, event.position())

    def undo_stroke(self):
        if self._strokes:
            previous = self._strokes.pop()
            self.mask = previous[0] if isinstance(previous, tuple) else previous
            self._mask_qimage = None
            self.selectionChanged.emit()
            self.update()
            return True
        return False


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка")
        self.resize(660, 680)
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setHtml(HELP_HTML)
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Групповые портреты {VERSION}")
        self.resize(1500, 950)

        root = data_dir()
        self.catalog = Catalog(root / "catalog.sqlite", root / "thumbs")
        self.finder = FaceFinder()
        self._faces_cache = {}

        self.project_id = None
        self.entries = []
        self.work_index = None
        self.ref_index = None
        self.work_rgb = None
        self.ref_rgb = None
        self.document = None
        self.layer = None
        self.layer_confirmed = False
        self.select_mode = "replace"

        self._build_ui()
        self._build_actions()
        self._set_tool("move")
        self.refresh_projects()

    # ---------- построение ----------
    def _build_ui(self):
        self.work_canvas = Canvas(editable=False)
        self.ref_canvas = Canvas(editable=True)
        self.ref_canvas.selectionChanged.connect(self._update_status)
        self.ref_canvas.transferRequested.connect(self.transfer_selection)
        self.ref_canvas.strokeFinished.connect(self._after_stroke)
        self.ref_canvas.brushChanged.connect(self._sync_brush)
        self.work_canvas.layerDragStarted.connect(self._begin_layer_change)
        self.work_canvas.layerDragged.connect(self.drag_layer)
        self.work_canvas.layerErased.connect(self.erase_layer)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        for title, widget in (("Рабочий снимок", self.work_canvas),
                              ("Эталон — с него берём выделение", self.ref_canvas)):
            panel = QWidget()
            box = QVBoxLayout(panel)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(0)
            header = QLabel(f"<b>{title}</b>")
            header.setStyleSheet("padding:5px 8px; color:#ddd; background:#3a3d43;")
            box.addWidget(header)
            box.addWidget(widget, 1)
            splitter.addWidget(panel)
        splitter.setSizes([750, 750])

        self.thumbs = ThumbStrip()
        self.thumbs.picked.connect(self.set_reference)
        self.thumbs.referenceRequested.connect(self.set_work)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.thumbs)
        self.setCentralWidget(central)

        self._build_side()
        self._build_library()
        self.statusBar()
        self.hint = QLabel("")
        self.statusBar().addPermanentWidget(self.hint)

    def _slider(self, form, title, low, high, value, tip):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.setToolTip(tip)
        slider.valueChanged.connect(self._on_slider)
        slider.sliderPressed.connect(self._begin_layer_change)
        form.addRow(title, slider)
        return slider

    def _build_side(self):
        panel = QWidget()
        box = QVBoxLayout(panel)

        box.addWidget(QLabel("<b>Слои</b>"))
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(140)
        self.layer_list.setToolTip("Список перенесённых фрагментов. Клик — выбрать для правки")
        self.layer_list.currentRowChanged.connect(self._select_layer)
        box.addWidget(self.layer_list)

        self.layer_info = QLabel("Слой не создан")
        self.layer_info.setWordWrap(True)
        box.addWidget(self.layer_info)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Выделение:"))
        self.mode_replace = QPushButton("Новое")
        self.mode_add = QPushButton("+ Добавить")
        self.mode_sub = QPushButton("− Вычесть")
        for btn, mode, tip in (
            (self.mode_replace, "replace", "Новое выделение заменяет прежнее"),
            (self.mode_add, "add", "Добавлять к выделению (или Alt)"),
            (self.mode_sub, "subtract", "Вычитать из выделения (или Ctrl)"),
        ):
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _=False, m=mode: self._set_select_mode(m))
            mode_row.addWidget(btn)
        box.addLayout(mode_row)
        self._set_select_mode("replace")

        form = QFormLayout()
        self.slider_feather = self._slider(form, "Растушёвка", 0, 60, 12, "Мягкость границы")
        self.slider_color = self._slider(form, "Цвет и свет", 0, 100, 85, "Подгонка под рабочий снимок")
        self.slider_angle = self._slider(form, "Поворот", -30, 30, 0, "Поворот слоя, градусы")
        self.slider_scale = self._slider(form, "Масштаб", 70, 140, 100, "Размер слоя, проценты")
        self.slider_brush = self._slider(form, "Кисть", 3, 400, 24, "Размер кисти, пиксели")
        box.addLayout(form)

        self.chk_poisson = QCheckBox("Бесшовное смешивание (Poisson)")
        self.chk_poisson.setChecked(True)
        self.chk_poisson.setToolTip("Выключите, если появляется ореол вокруг лица")
        self.chk_poisson.stateChanged.connect(lambda: self._render_work())
        box.addWidget(self.chk_poisson)

        self.btn_confirm = QPushButton("✓ Подтвердить перенос")
        self.btn_confirm.setToolTip("Закрепить слой и начать следующий")
        self.btn_confirm.clicked.connect(self.confirm_layer)
        self.btn_confirm.setEnabled(False)
        box.addWidget(self.btn_confirm)

        row = QHBoxLayout()
        self.btn_cancel_layer = QPushButton("Убрать слой")
        self.btn_cancel_layer.clicked.connect(self.cancel_layer)
        self.btn_refine = QPushButton("Уточнить границу")
        self.btn_refine.clicked.connect(self.refine_selection)
        row.addWidget(self.btn_cancel_layer)
        row.addWidget(self.btn_refine)
        box.addLayout(row)

        box.addStretch(1)
        dock = QDockWidget("Настройки", self)
        dock.setWidget(panel)
        dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_library(self):
        panel = QWidget()
        box = QVBoxLayout(panel)
        self.btn_new = QPushButton("Новая серия")
        self.btn_new.clicked.connect(self.new_project)
        self.btn_add_photos = QPushButton("Добавить снимки")
        self.btn_add_photos.clicked.connect(self.import_photos)
        box.addWidget(self.btn_new)
        box.addWidget(self.btn_add_photos)

        self.project_list = QListWidget()
        self.project_list.setIconSize(QSize(48, 48))
        self.project_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.project_list.itemDoubleClicked.connect(self._open_selected)
        box.addWidget(self.project_list, 1)

        row = QHBoxLayout()
        self.btn_open = QPushButton("Открыть")
        self.btn_open.clicked.connect(self._open_selected)
        self.btn_delete = QPushButton("Удалить")
        self.btn_delete.clicked.connect(self.delete_project)
        row.addWidget(self.btn_open)
        row.addWidget(self.btn_delete)
        box.addLayout(row)

        dock = QDockWidget("Каталог серий", self)
        dock.setWidget(panel)
        dock.setMinimumWidth(210)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_actions(self):
        bar = QToolBar("Инструменты")
        bar.setMovable(False)
        bar.setIconSize(QSize(24, 24))
        bar.setStyleSheet("QToolBar{background:#f0f0f0; spacing:3px; padding:2px;}"
                          "QToolButton{padding:3px; border-radius:4px;}"
                          "QToolButton:checked{background:#cfe0f7;}")
        self.addToolBar(bar)

        self.tool_actions = {}
        for key, title, shortcut in (
            ("move", "Перемещение", "V"),
            ("brush", "Кисть", "B"),
            ("eraser", "Ластик", "E"),
            ("lasso", "Лассо", "L"),
            ("zoom", "Лупа", "Z"),
            ("view", "Просмотр", "H"),
        ):
            action = QAction(draw_icon(key), title, self)
            action.setShortcut(QKeySequence(shortcut))
            action.setToolTip(f"{title} ({shortcut}) — {TOOL_HINTS[key][1]}")
            action.setCheckable(True)
            action.triggered.connect(lambda _=False, k=key: self._set_tool(k))
            bar.addAction(action)
            self.tool_actions[key] = action

        bar.addSeparator()
        for title, shortcut, slot in (
            ("Отменить", "Ctrl+Z", self.undo),
            ("Повторить", "Ctrl+Y", self.redo),
            ("Снять", "Delete", self.clear_selection),
            ("Вписать", "Ctrl+0", self.fit_view),
            ("100%", "Ctrl+1", self.zoom_100),
            ("Экспорт", "Ctrl+E", self.export_image),
            ("Импорт", "Ctrl+O", self.import_photos),
            ("Справка", "F1", lambda: HelpDialog(self).exec()),
        ):
            action = QAction(title, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(slot)
            bar.addAction(action)

        for key, delta in (("Ъ", 4), ("Х", -4)):
            action = QAction(f"Кисть {delta:+d}", self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(lambda _=False, d=delta: self.ref_canvas.bump_brush(d))
            self.addAction(action)

    def _set_tool(self, tool):
        self.ref_canvas.set_tool(tool)
        self.work_canvas.set_tool(tool)
        for key, action in self.tool_actions.items():
            action.setChecked(key == tool)
        title, tip = TOOL_HINTS[tool]
        self.hint.setText(f"{title}: {tip}")
        draggable = self.layer is not None and tool in ("move", "eraser")
        self.work_canvas.layer_drag = draggable
        self.ref_canvas.layer_drag = False

    def _after_stroke(self):
        if self.ref_canvas.tool == "lasso" and self.ref_canvas.has_selection():
            self._set_tool("move")
            self.statusBar().showMessage("Выделение готово. Тяните его на рабочий снимок слева", 6000)

    def _sync_brush(self, value):
        self.slider_brush.blockSignals(True)
        self.slider_brush.setValue(value)
        self.slider_brush.blockSignals(False)

    def _set_select_mode(self, mode):
        self.select_mode = mode
        for btn, value in ((self.mode_replace, "replace"), (self.mode_add, "add"), (self.mode_sub, "subtract")):
            btn.setChecked(value == mode)
        self.ref_canvas.set_select_mode(mode)

    # ---------- каталог ----------
    def refresh_projects(self):
        self.project_list.clear()
        for pid, name, created, count in self.catalog.projects():
            item = QListWidgetItem(f"{name} · {count} фото · {created}")
            item.setData(Qt.ItemDataRole.UserRole, pid)
            self.project_list.addItem(item)
            if pid == self.project_id:
                self.project_list.setCurrentItem(item)

    def _open_selected(self, *_):
        item = self.project_list.currentItem()
        if item is not None:
            self.open_project(item.data(Qt.ItemDataRole.UserRole))

    def _ask_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Выберите снимки", "", "Снимки JPEG (*.jpg *.jpeg)")
        return paths

    def new_project(self):
        paths = self._ask_images()
        if not paths:
            return
        default = Path(paths[0]).parent.name or "Серия"
        name, ok = QInputDialog.getText(self, "Новая серия", "Название серии:", text=default)
        if not ok or not name.strip():
            return
        try:
            pid = self.catalog.create(name.strip(), paths)
        except OSError as error:
            QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать снимки:\n{error}")
            return
        self.open_project(pid)

    def open_project(self, project_id):
        self.project_id = project_id
        self.entries = self.catalog.photos(project_id)
        self._faces_cache.clear()
        self.document = None
        self.layer = None
        self.work_index = None
        self.ref_index = None
        self.thumbs.set_photos(self.entries)
        self.thumbs.set_work(None)
        if self.entries:
            self.set_work(0)
            if len(self.entries) > 1:
                self.set_reference(1)
        self.refresh_projects()

    def delete_project(self):
        item = self.project_list.currentItem()
        if item is None:
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        if QMessageBox.question(self, "Удалить серию",
                                "Удалить серию из каталога? Файлы на диске останутся.") != QMessageBox.StandardButton.Yes:
            return
        self.catalog.delete(pid)
        if pid == self.project_id:
            self.project_id = None
            self.entries = []
            self.document = None
            self.layer = None
            self.thumbs.set_photos([])
            self.work_canvas.set_image(None, keep_view=False)
            self.ref_canvas.set_image(None, keep_view=False)
        self.refresh_projects()

    def import_photos(self):
        paths = self._ask_images()
        if not paths:
            return
        if self.project_id is None:
            default = Path(paths[0]).parent.name or "Серия"
            name, ok = QInputDialog.getText(self, "Новая серия", "Название серии:", text=default)
            if not ok or not name.strip():
                return
            try:
                self.project_id = self.catalog.create(name.strip(), paths)
            except OSError as error:
                QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать снимки:\n{error}")
                return
        else:
            try:
                self.catalog.add_photos(self.project_id, paths)
            except OSError as error:
                QMessageBox.critical(self, "Ошибка", f"Не удалось прочитать снимки:\n{error}")
                return
        self.open_project(self.project_id)

    # ---------- снимки ----------
    def _detect(self, rgb):
        key = id(rgb)
        if key not in self._faces_cache:
            self._faces_cache[key] = self.finder.detect(rgb)
        return self._faces_cache[key]

    def set_work(self, index):
        if not self.entries or not 0 <= index < len(self.entries):
            return
        try:
            self.work_rgb = load_rgb(self.entries[index][0])
        except OSError as error:
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть снимок:\n{error}")
            return
        self.work_index = index
        self.document = Document()
        self.layer = None
        self.layer_confirmed = False
        self.work_canvas.set_image(self.work_rgb, keep_view=self.work_canvas.has_image())
        self.thumbs.set_work(index)
        self._refresh_layer_ui()
        self._update_status()

    def set_reference(self, index):
        if not self.entries or not 0 <= index < len(self.entries):
            return
        try:
            self.ref_rgb = load_rgb(self.entries[index][0])
        except OSError as error:
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть снимок:\n{error}")
            return
        self.ref_index = index
        self.ref_canvas.set_image(self.ref_rgb, keep_view=self.ref_canvas.has_image())
        self.thumbs.set_active(index)
        if index == self.work_index:
            self.statusBar().showMessage("Этот снимок сейчас рабочий: правки применятся к нему же", 4000)
        self._update_status()

    def clear_selection(self):
        self.ref_canvas.clear_mask()
        self._update_status()

    def refine_selection(self):
        if self.ref_canvas.has_selection():
            self.ref_canvas.set_mask(refine_edge(self.ref_rgb, self.ref_canvas.mask))
            self._update_status()

    # ---------- слой ----------
    def transfer_selection(self):
        if self.document is None or not self.ref_canvas.has_selection():
            self.statusBar().showMessage("Выделите лицо на эталоне справа", 4000)
            return
        drop = self._drop_point()
        layer = build_layer(
            self.ref_rgb,
            self.ref_canvas.mask,
            self._detect(self.ref_rgb),
            self._detect(self.work_rgb),
            base_point=drop,
            feather=self.slider_feather.value(),
        )
        if layer is None:
            return
        layer.color = self.slider_color.value() / 100.0
        if layer.faces > 1:
            layer.name = f"Лица ({layer.faces})"
        elif layer.faces == 1:
            layer.name = f"Лицо {len(self.document.layers) + 1}"
        else:
            layer.name = f"Фрагмент {len(self.document.layers) + 1}"
        self.document.add(layer)
        self.layer = layer
        self.layer_confirmed = False
        self.ref_canvas.clear_mask()
        self._render_work()
        self._refresh_layer_ui()
        if layer.faces and layer.aligned < layer.faces:
            self.statusBar().showMessage(
                f"Лиц: {layer.faces}, выровнено: {layer.aligned}. Поправьте перетаскиванием "
                f"и нажмите «Подтвердить перенос»", 8000)
        else:
            self.statusBar().showMessage("Слой создан. Проверьте и нажмите «Подтвердить перенос»", 6000)

    def _drop_point(self):
        pos = QCursor.pos()
        local = self.work_canvas.mapFromGlobal(pos)
        if self.work_canvas.rect().contains(local):
            point = self.work_canvas.screen_to_image(local)
            return (point.x(), point.y())
        return None

    def confirm_layer(self):
        if self.layer is None:
            return
        self.layer_confirmed = True
        self.statusBar().showMessage(f"«{self.layer.name}» закреплён. Выделите следующее лицо", 6000)
        self.ref_canvas.clear_mask()
        self._refresh_layer_ui()

    def _select_layer(self, row):
        if self.document is None or row < 0:
            return
        ordered = list(reversed(self.document.layers))
        if row >= len(ordered):
            return
        self.layer = ordered[row]
        self.layer_confirmed = True
        self._sync_sliders()
        self._refresh_layer_ui()

    def _sync_sliders(self):
        if self.layer is None:
            return
        for slider, value in ((self.slider_feather, self.layer.feather),
                              (self.slider_color, int(self.layer.color * 100)),
                              (self.slider_angle, int(self.layer.angle)),
                              (self.slider_scale, int(self.layer.scale * 100))):
            slider.blockSignals(True)
            slider.setValue(int(value))
            slider.blockSignals(False)

    def _begin_layer_change(self):
        if self.document is not None and self.layer is not None:
            self.document.begin_change()

    def drag_layer(self, dx, dy):
        if self.layer is None:
            return
        self.layer.dx += dx
        self.layer.dy += dy
        self._render_work(preview=True)

    def erase_layer(self, x, y, radius):
        if self.layer is None:
            return
        self.layer.erase(x, y, radius)
        self._render_work(preview=True)

    def _render_work(self, preview=False):
        if self.document is None:
            return
        poisson = self.chk_poisson.isChecked()
        self.work_canvas.set_image(
            self.document.render(self.work_rgb, poisson=poisson and not preview), keep_view=True
        )

    def _on_slider(self):
        for canvas in (self.ref_canvas, self.work_canvas):
            canvas.set_brush_radius(self.slider_brush.value())
        if self.layer is None:
            return
        self.layer.feather = self.slider_feather.value()
        self.layer.color = self.slider_color.value() / 100.0
        self.layer.angle = float(self.slider_angle.value())
        self.layer.scale = self.slider_scale.value() / 100.0
        self._render_work(preview=True)

    def _refresh_layer_ui(self):
        layers = list(reversed(self.document.layers)) if self.document else []
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for layer in layers:
            self.layer_list.addItem(layer.name)
            if layer is self.layer:
                self.layer_list.setCurrentRow(self.layer_list.count() - 1)
        self.layer_list.blockSignals(False)

        if self.layer is None:
            self.layer_info.setText("Слой не создан. Выделите область справа и перетащите на рабочий снимок.")
        else:
            self.layer_info.setText(
                f"Активный: {self.layer.name}. Лиц: {self.layer.faces}, выровнено: {self.layer.aligned}."
            )
        self.btn_confirm.setEnabled(self.layer is not None and not self.layer_confirmed)
        self.work_canvas.layer_drag = self.layer is not None and self.work_canvas.tool in ("move", "eraser")

    def cancel_layer(self):
        if self.layer is None:
            return
        self.document.remove(self.layer)
        self.layer = self.document.layers[-1] if self.document.layers else None
        self._render_work()
        self._refresh_layer_ui()

    def export_image(self):
        if self.document is None:
            return
        name = "result.jpg"
        if self.work_index is not None:
            name = f"{Path(self.entries[self.work_index][0]).stem}_портрет.jpg"
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт рабочего снимка", name, "JPEG (*.jpg)")
        if not path:
            return
        try:
            save_jpeg(self.document.render(self.work_rgb, poisson=self.chk_poisson.isChecked()), path)
        except OSError as error:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{error}")
            return
        self.statusBar().showMessage(f"Экспортировано: {path}", 5000)

    # ---------- прочее ----------
    def undo(self):
        if self.ref_canvas.undo_stroke():
            self._update_status()
            return
        if self.document is not None and self.document.undo():
            self.layer = self.document.layers[-1] if self.document.layers else None
            self._render_work()
            self._refresh_layer_ui()

    def redo(self):
        if self.document is not None and self.document.redo():
            self.layer = self.document.layers[-1] if self.document.layers else None
            self._render_work()
            self._refresh_layer_ui()

    def fit_view(self):
        for canvas in (self.work_canvas, self.ref_canvas):
            canvas.fit()

    def zoom_100(self):
        for canvas in (self.work_canvas, self.ref_canvas):
            canvas.set_zoom(1.0)

    def _update_status(self):
        if self.ref_canvas.has_selection():
            self.statusBar().showMessage(
                "Выделение есть. Тяните его инструментом «Перемещение» на рабочее фото."
            )
        elif self.document is not None:
            self.statusBar().showMessage(f"Слоёв на рабочем: {len(self.document.layers)}")

    def closeEvent(self, event):
        self.finder.close()
        super().closeEvent(event)
