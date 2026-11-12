import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget

MIN_ZOOM, MAX_ZOOM = 0.02, 16.0
BRUSH_MIN, BRUSH_MAX = 3, 400


def to_qimage(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def rgba_to_qimage(rgba):
    rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    return QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


def draw_icon(kind):
    """Простые иконки в стиле Photoshop, рисуются кодом."""
    size = 32
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    ink = QColor(230, 230, 230)

    if kind == "move":
        p.setPen(QPen(ink, 1.8))
        p.drawLine(16, 5, 16, 27)
        p.drawLine(5, 16, 27, 16)
        for x, y, dx, dy in ((16, 5, 0, -1), (16, 27, 0, 1), (5, 16, -1, 0), (27, 16, 1, 0)):
            p.drawLine(x, y, x - 3 * dx + 3 * dy, y - 3 * dy + 3 * dx)
            p.drawLine(x, y, x - 3 * dx - 3 * dy, y - 3 * dy - 3 * dx)
    elif kind == "brush":
        p.setPen(QPen(ink, 2))
        p.drawLine(7, 25, 22, 10)
        p.setBrush(QBrush(ink))
        p.drawEllipse(QPointF(23.5, 8.5), 3.2, 3.2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(6, 26, 10, 22)
    elif kind == "lasso":
        p.setPen(QPen(ink, 1.8))
        path = QPainterPath(QPointF(16, 5))
        path.cubicTo(26, 5, 28, 14, 20, 19)
        path.cubicTo(13, 23, 6, 21, 6, 15)
        path.cubicTo(6, 9, 12, 6, 17, 7)
        p.drawPath(path)
        p.drawLine(20, 19, 17, 26)
    elif kind == "zoom":
        p.setPen(QPen(ink, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(13.5, 13.5), 7.5, 7.5)
        p.drawLine(19, 19, 26, 26)
    elif kind == "view":
        p.setPen(QPen(ink, 1.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(QPointF(4, 16))
        path.cubicTo(11, 7, 21, 7, 28, 16)
        path.cubicTo(21, 25, 11, 25, 4, 16)
        p.drawPath(path)
        p.drawEllipse(QPointF(16, 16), 3.4, 3.4)
    p.end()
    return QIcon(pix)


class Canvas(QWidget):
    selectionChanged = Signal()
    transferRequested = Signal()
    layerDragStarted = Signal()
    layerDragged = Signal(float, float)
    zoomChanged = Signal(float)
    strokeFinished = Signal()
    brushChanged = Signal(int)

    def __init__(self, editable=False, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 200)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        self._image = None
        self._qimage = None
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._fit_on_next = True

        self.editable = editable
        self.layer_drag = False
        self._dragging_layer = False
        self.tool = "move"
        self.brush_radius = 24
        self.mask = None
        self._mask_qimage = None
        self._strokes = []
        self._sketch = None
        self._painting = False
        self._erasing = False
        self._panning = False
        self._last_screen = None
        self._overlay = None
        self._overlay_qimage = None
        self._overlay_pos = (0, 0)
        self._cursor_image = QPointF(0, 0)
        self._drag_start = None
        self._drag_mask0 = None
        self.drag_preview = None
        self._paint_mode = "replace"

    # ---------- данные ----------
    def set_image(self, rgb, keep_view=True):
        self._image = rgb
        self._qimage = to_qimage(rgb) if rgb is not None else None
        self.mask = np.zeros(rgb.shape[:2], np.uint8) if rgb is not None else None
        self._strokes = []
        self._mask_qimage = None
        self._overlay = None
        self._overlay_qimage = None
        if not keep_view:
            self._fit_on_next = True
            if self.width() > 0:
                self.fit()
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

    def set_overlay(self, rgba, pos):
        self._overlay = rgba
        self._overlay_qimage = rgba_to_qimage(rgba) if rgba is not None else None
        self._overlay_pos = pos
        self.update()

    def clear_overlay(self):
        self.set_overlay(None, (0, 0))

    def set_brush_radius(self, value):
        value = int(max(BRUSH_MIN, min(BRUSH_MAX, value)))
        if value == self.brush_radius:
            return
        self.brush_radius = value
        self.brushChanged.emit(value)
        self.update()

    def bump_brush(self, delta):
        self.set_brush_radius(self.brush_radius + delta)

    def set_tool(self, tool):
        self.tool = tool
        cursors = {
            "brush": Qt.CursorShape.CrossCursor,
            "lasso": Qt.CursorShape.CrossCursor,
            "zoom": Qt.CursorShape.PointingHandCursor,
            "move": Qt.CursorShape.SizeAllCursor,
            "view": Qt.CursorShape.ArrowCursor,
        }
        self.setCursor(cursors.get(tool, Qt.CursorShape.ArrowCursor))
        self.update()

    # ---------- вид ----------
    def zoom(self):
        return self._zoom

    def fit(self):
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        if w <= 0 or h <= 0 or self.width() <= 0:
            return
        self._zoom = min(self.width() / w, self.height() / h) * 0.98
        self._center()
        self._fit_on_next = False
        self.zoomChanged.emit(self._zoom)
        self.update()

    def _center(self):
        if self._image is None:
            return
        h, w = self._image.shape[:2]
        self._offset = QPointF((self.width() - w * self._zoom) / 2, (self.height() - h * self._zoom) / 2)

    def set_zoom(self, value, anchor=None):
        if self._image is None:
            return
        value = max(MIN_ZOOM, min(MAX_ZOOM, value))
        anchor = anchor or QPointF(self.width() / 2, self.height() / 2)
        before = self.screen_to_image(anchor)
        self._zoom = value
        after = self.image_to_screen(before)
        self._offset += anchor - after
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
        if self._fit_on_next:
            self.fit()
        super().resizeEvent(event)

    def _mask_layer(self):
        if self._mask_qimage is None:
            tint = self.mask > 0
            rgba = np.zeros((*self.mask.shape, 4), np.uint8)
            rgba[tint] = (255, 60, 60, 110)
            self._mask_qimage = rgba_to_qimage(rgba)
        return self._mask_qimage

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(38, 40, 44))
        if self._qimage is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Перетащите снимок сюда")
            return

        target = QRectF(self._offset.x(), self._offset.y(),
                        self._image.shape[1] * self._zoom, self._image.shape[0] * self._zoom)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 1.0)
        painter.drawImage(target, self._qimage)

        if self._overlay_qimage is not None:
            painter.drawImage(
                QRectF(self._offset.x() + self._overlay_pos[0] * self._zoom,
                       self._offset.y() + self._overlay_pos[1] * self._zoom,
                       self._overlay.shape[1] * self._zoom,
                       self._overlay.shape[0] * self._zoom),
                self._overlay_qimage,
            )

        if self.drag_preview is not None:
            rgba, pos = self.drag_preview
            painter.setOpacity(0.65)
            painter.drawImage(
                QRectF(self._offset.x() + pos[0] * self._zoom, self._offset.y() + pos[1] * self._zoom,
                       rgba.shape[1] * self._zoom, rgba.shape[0] * self._zoom),
                rgba_to_qimage(rgba),
            )
            painter.setOpacity(1.0)

        if self.mask is not None and self.mask.any():
            painter.drawImage(target, self._mask_layer())

        if self._sketch is not None and len(self._sketch) > 1:
            painter.setPen(QPen(QColor(255, 210, 90), 1.5, Qt.PenStyle.DashLine))
            path = QPainterPath(self._sketch[0])
            for point in self._sketch[1:]:
                path.lineTo(point)
            painter.drawPath(path)

        if self.editable and self.tool == "brush":
            center = self.image_to_screen(self._cursor_image)
            radius = self.brush_radius * self._zoom
            painter.setPen(QPen(QColor(255, 255, 255, 210), 1))
            painter.setBrush(QBrush(QColor(255, 255, 255, 35)))
            painter.drawEllipse(center, max(3.0, radius), max(3.0, radius))

    # ---------- мышь ----------
    @staticmethod
    def paint_mode(modifiers):
        if modifiers & Qt.KeyboardModifier.AltModifier:
            return "add"
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            return "subtract"
        return "replace"

    def mousePressEvent(self, event):
        if self._image is None:
            return
        pos = event.position()
        self._last_screen = pos
        button = event.button()

        if button == Qt.MouseButton.MiddleButton:
            self._panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if button != Qt.MouseButton.LeftButton and not (
            button == Qt.MouseButton.RightButton and self.tool == "brush" and self.editable
        ):
            return

        if self.tool == "move" and button == Qt.MouseButton.LeftButton:
            if self.layer_drag:
                self._dragging_layer = True
                self.layerDragStarted.emit()
            elif self.editable and self.has_selection():
                self._drag_start = self.screen_to_image(pos)
                self._drag_mask0 = self.mask.copy()
                self._make_drag_preview()
            elif self.editable:
                self._panning = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            else:
                self._panning = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self.tool == "view":
            self._panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if not self.editable:
            if self.tool == "zoom":
                factor = 0.8 if event.modifiers() & Qt.KeyboardModifier.AltModifier else 1.25
                self.zoom_step(factor, pos)
            return

        if self.tool == "zoom":
            factor = 0.8 if event.modifiers() & Qt.KeyboardModifier.AltModifier else 1.25
            self.zoom_step(factor, pos)
        elif self.tool == "brush" and self.mask is not None:
            self._painting = True
            self._erasing = button == Qt.MouseButton.RightButton
            self._paint_mode = "subtract" if self._erasing else self.paint_mode(event.modifiers())
            self._strokes.append((self.mask.copy(), self._paint_mode))
            if self._paint_mode == "replace":
                self.mask[:] = 0
                self._mask_qimage = None
            self._paint_dot(self.screen_to_image(pos), self._paint_mode)
        elif self.tool == "lasso":
            self._strokes.append((self.mask.copy(), self.paint_mode(event.modifiers())))
            self._sketch = [pos]

    def _make_drag_preview(self):
        if self.mask is None or self._image is None or not self.mask.any():
            self.drag_preview = None
            return
        ys, xs = np.nonzero(self.mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        patch = self._image[y0:y1, x0:x1]
        sub = self.mask[y0:y1, x0:x1]
        rgba = np.zeros((*sub.shape, 4), np.uint8)
        rgba[..., :3] = patch
        rgba[..., 3] = sub
        self.drag_preview = (rgba, (x0, y0))
        self.update()

    @staticmethod
    def _paint_value(mode):
        return 0 if mode == "subtract" else 255

    def _paint_dot(self, point, mode="replace"):
        value = 0 if mode == "subtract" else 255
        x, y = int(round(point.x())), int(round(point.y()))
        cv2.circle(self.mask, (x, y), self.brush_radius, value, -1)
        self._mask_qimage = None
        self.selectionChanged.emit()
        self.update()

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
        if self._drag_start is not None and self._drag_mask0 is not None:
            current = self.screen_to_image(pos)
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
            value = self._paint_value(self._paint_mode)
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
        if self._drag_start is not None:
            self.drag_preview = None
            self._drag_start = None
            self._drag_mask0 = None
            self.transferRequested.emit()
            return
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.tool == "move" else Qt.CursorShape.ArrowCursor)
        if self._painting:
            self._painting = False
            self._erasing = False
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
        _, mode = self._strokes[-1] if self._strokes else (None, "replace")
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
        if step == 0:
            return
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
