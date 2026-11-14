import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

MIN_ZOOM, MAX_ZOOM = 0.02, 16.0
BRUSH_MIN, BRUSH_MAX = 3, 400

INK = QColor(45, 48, 54)
ACCENT = QColor(40, 110, 200)


def to_qimage(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def rgba_to_qimage(rgba):
    rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    return QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


def draw_icon(kind, size=26):
    """Иконки инструментов. Тёмные, чтобы читались на светлой панели."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    s = size / 32.0

    def sc(v):
        return v * s

    if kind == "move":
        p.setPen(QPen(INK, sc(2.0), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(sc(16), sc(6)), QPointF(sc(16), sc(26)))
        p.drawLine(QPointF(sc(6), sc(16)), QPointF(sc(26), sc(16)))
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            tip = QPointF(sc(16 + 10 * dx), sc(16 + 10 * dy))
            base = QPointF(sc(16 + 5 * dx), sc(16 + 5 * dy))
            left = QPointF(base.x() - sc(3) * dy, base.y() + sc(3) * dx)
            right = QPointF(base.x() + sc(3) * dy, base.y() - sc(3) * dx)
            p.drawLine(tip, left)
            p.drawLine(tip, right)
    elif kind == "brush":
        p.setPen(QPen(INK, sc(2.6), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(sc(8), sc(24)), QPointF(sc(20), sc(12)))
        p.setPen(QPen(ACCENT, sc(2.6), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(sc(20.5), sc(11.5)), QPointF(sc(23.5), sc(8.5)))
        p.setBrush(QBrush(INK))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(sc(24), sc(8)), sc(1.6), sc(1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(INK, sc(1.6), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(sc(6), sc(26)), QPointF(sc(9), sc(23)))
    elif kind == "eraser":
        p.save()
        p.translate(sc(16), sc(16))
        p.rotate(-35)
        p.setPen(QPen(INK, sc(2.0)))
        p.setBrush(QBrush(QColor(240, 240, 240)))
        p.drawRoundedRect(QRectF(sc(-9), sc(-5), sc(18), sc(10)), sc(2), sc(2))
        p.setBrush(QBrush(QColor(200, 205, 212)))
        p.drawRect(QRectF(sc(-9), sc(-5), sc(7), sc(10)))
        p.restore()
    elif kind == "lasso":
        p.setPen(QPen(INK, sc(1.9)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(QPointF(sc(16), sc(6)))
        path.cubicTo(sc(26), sc(6), sc(28), sc(15), sc(20), sc(19))
        path.cubicTo(sc(13), sc(22), sc(6), sc(20), sc(6), sc(14))
        path.cubicTo(sc(6), sc(9), sc(12), sc(6), sc(17), sc(7))
        p.drawPath(path)
        p.setPen(QPen(ACCENT, sc(2.0), Qt.PenStyle.DashLine))
        p.drawLine(QPointF(sc(19), sc(19)), QPointF(sc(16), sc(26)))
    elif kind == "zoom":
        p.setPen(QPen(INK, sc(2.4)))
        p.setBrush(QBrush(QColor(255, 255, 255, 120)))
        p.drawEllipse(QPointF(sc(13.5), sc(13.5)), sc(7.0), sc(7.0))
        p.setPen(QPen(INK, sc(3.0), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(sc(19), sc(19)), QPointF(sc(26), sc(26)))
        p.setPen(QPen(ACCENT, sc(1.6)))
        p.drawLine(QPointF(sc(10), sc(13.5)), QPointF(sc(17), sc(13.5)))
        p.drawLine(QPointF(sc(13.5), sc(10)), QPointF(sc(13.5), sc(17)))
    elif kind == "view":
        p.setPen(QPen(INK, sc(1.9)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(QPointF(sc(4), sc(16)))
        path.cubicTo(sc(11), sc(7), sc(21), sc(7), sc(28), sc(16))
        path.cubicTo(sc(21), sc(25), sc(11), sc(25), sc(4), sc(16))
        p.drawPath(path)
        p.setBrush(QBrush(ACCENT))
        p.setPen(QPen(INK, sc(1.2)))
        p.drawEllipse(QPointF(sc(16), sc(16)), sc(3.2), sc(3.2))
    p.end()
    return QIcon(pix)


def make_cursor(kind, radius=0):
    """Курсор в виде инструмента: сразу видно, что включено."""
    size = 32
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    ink = QColor(30, 32, 36)
    white = QColor(255, 255, 255)

    def outline(pen_width, draw):
        for color, width in ((white, pen_width + 2.2), (ink, pen_width)):
            p.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            draw()

    hotspot = QPointF(4, 4)

    if kind == "move":
        def arrows():
            p.drawLine(QPointF(16, 3), QPointF(16, 29))
            p.drawLine(QPointF(3, 16), QPointF(29, 16))
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                tip = QPointF(16 + 11 * dx, 16 + 11 * dy)
                base = QPointF(16 + 6 * dx, 16 + 6 * dy)
                p.drawLine(tip, QPointF(base.x() - 3.2 * dy, base.y() + 3.2 * dx))
                p.drawLine(tip, QPointF(base.x() + 3.2 * dy, base.y() - 3.2 * dx))
        p.setBrush(Qt.BrushStyle.NoBrush)
        outline(1.8, arrows)
        hotspot = QPointF(16, 16)
    elif kind == "brush":
        def brush():
            p.drawLine(QPointF(6, 26), QPointF(21, 11))
            p.drawLine(QPointF(21, 11), QPointF(24, 8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        outline(2.6, brush)
        hotspot = QPointF(6, 26)
    elif kind == "eraser":
        def eraser():
            p.drawRoundedRect(QRectF(8, 14, 20, 11), 2, 2)
        p.setBrush(QBrush(QColor(245, 245, 245)))
        outline(1.8, eraser)
        p.setBrush(Qt.BrushStyle.NoBrush)
        hotspot = QPointF(10, 20)
    elif kind == "lasso":
        def lasso():
            path = QPainterPath(QPointF(16, 5))
            path.cubicTo(28, 5, 29, 16, 19, 20)
            path.cubicTo(10, 24, 3, 21, 5, 14)
            path.cubicTo(6, 8, 13, 5, 18, 7)
            p.drawPath(path)
            p.drawLine(QPointF(18, 20), QPointF(15, 28))
        p.setBrush(Qt.BrushStyle.NoBrush)
        outline(1.7, lasso)
        hotspot = QPointF(16, 5)
    elif kind == "zoom":
        def lens():
            p.drawEllipse(QPointF(13, 13), 8, 8)
            p.drawLine(QPointF(19, 19), QPointF(28, 28))
        p.setBrush(QBrush(QColor(255, 255, 255, 90)))
        outline(1.9, lens)
        hotspot = QPointF(13, 13)
    elif kind == "cross":
        def cross():
            p.drawLine(QPointF(16, 4), QPointF(16, 12))
            p.drawLine(QPointF(16, 20), QPointF(16, 28))
            p.drawLine(QPointF(4, 16), QPointF(12, 16))
            p.drawLine(QPointF(20, 16), QPointF(28, 16))
        p.setBrush(Qt.BrushStyle.NoBrush)
        outline(1.4, cross)
        hotspot = QPointF(16, 16)
    else:
        hotspot = QPointF(0, 0)

    p.end()
    return QCursor(pix, int(hotspot.x()), int(hotspot.y()))


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
        self._erasing_layer = False
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
        self.select_mode = "replace"
        self._apply_cursor()

    # ---------- данные ----------
    def set_image(self, rgb, keep_view=True):
        """Смена снимка. По умолчанию масштаб и позиция сохраняются."""
        previous = None
        if keep_view and self._image is not None:
            previous = (self._zoom, self._offset)
        self._image = rgb
        self._qimage = to_qimage(rgb) if rgb is not None else None
        self.mask = np.zeros(rgb.shape[:2], np.uint8) if rgb is not None else None
        self._strokes = []
        self._mask_qimage = None
        self._overlay = None
        self._overlay_qimage = None
        if previous is not None and rgb is not None:
            zoom, offset = previous
            self._zoom = zoom
            self._offset = offset
            self._fit_on_next = False
        elif not keep_view:
            self._fit_on_next = True
            if self.width() > 0:
                self.fit()
        self.zoomChanged.emit(self._zoom)
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

    def set_select_mode(self, mode):
        self.select_mode = mode

    def _apply_cursor(self):
        if self.tool in ("brush", "eraser"):
            self.setCursor(make_cursor(self.tool))
        elif self.tool == "lasso":
            self.setCursor(make_cursor("lasso"))
        elif self.tool == "zoom":
            self.setCursor(make_cursor("zoom"))
        elif self.tool == "move":
            self.setCursor(make_cursor("move"))
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_tool(self, tool):
        self.tool = tool
        self._apply_cursor()
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
            rgba[tint] = (60, 150, 255, 105)
            self._mask_qimage = rgba_to_qimage(rgba)
        return self._mask_qimage

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(52, 55, 60))
        if self._qimage is None:
            painter.setPen(QColor(190, 190, 190))
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
            painter.setPen(QPen(QColor(255, 200, 60), 1.6, Qt.PenStyle.DashLine))
            path = QPainterPath(self._sketch[0])
            for point in self._sketch[1:]:
                path.lineTo(point)
            painter.drawPath(path)

        if self.editable and self.tool in ("brush", "eraser"):
            center = self.image_to_screen(self._cursor_image)
            radius = self.brush_radius * self._zoom
            color = QColor(255, 90, 90, 230) if self.tool == "eraser" else QColor(255, 255, 255, 230)
            painter.setPen(QPen(color, 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, max(3.0, radius), max(3.0, radius))

    # ---------- вспомогательное ----------
    @staticmethod
    def _mods(event):
        """Модификаторы с самого события плюс текущее состояние клавиш."""
        mods = event.modifiers() | QApplication.keyboardModifiers()
        return mods

    def _mode_for(self, event):
        """Alt — добавить, Ctrl — вычесть, иначе берём переключатель в панели."""
        mods = self._mods(event)
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
        x, y = int(round(point.x())), int(round(point.y()))
        cv2.circle(self.mask, (x, y), self.brush_radius, value, -1)
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
        mods = self._mods(event)

        if button == Qt.MouseButton.MiddleButton:
            self._panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if button == Qt.MouseButton.RightButton and self.tool in ("brush", "eraser") and self.editable:
            self._painting = True
            self._erasing = True
            self._paint_mode = "subtract"
            self._strokes.append((self.mask.copy(), "subtract"))
            self._paint_dot(self._cursor_image, "subtract")
            return

        if button != Qt.MouseButton.LeftButton:
            return

        # Перемещение: слева двигаем слой, справа тянем выделение
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
            factor = 0.8 if mods & Qt.KeyboardModifier.AltModifier else 1.25
            self.zoom_step(factor, pos)
            return

        if not self.editable:
            return

        if self.tool == "eraser":
            if self.layer_drag:
                self._erasing_layer = True
                self._cursor_image = self.screen_to_image(pos)
                self.layerErased.emit(self._cursor_image.x(), self._cursor_image.y(), float(self.brush_radius))
            elif self.mask is not None:
                self._painting = True
                self._paint_mode = "subtract"
                self._strokes.append((self.mask.copy(), "subtract"))
                self._paint_dot(self._cursor_image, "subtract")
            return

        if self.tool == "brush" and self.mask is not None:
            self._painting = True
            self._paint_mode = self._mode_for(event)
            self._strokes.append((self.mask.copy(), self._paint_mode))
            if self._paint_mode == "replace":
                self.mask[:] = 0
                self._mask_qimage = None
            self._paint_dot(self._cursor_image, self._paint_mode)
            return

        if self.tool == "lasso":
            # Лассо всегда рисует. Перенос делает «Перемещение» или автопереход после обводки.
            self._strokes.append((self.mask.copy(), self._mode_for(event)))
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
                # Возвращаем маску к исходной: переносим то, что выделили.
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
        mode = self._strokes[-1][1] if self._strokes else "replace"
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
