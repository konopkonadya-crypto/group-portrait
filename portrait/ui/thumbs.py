from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget


class Thumbnail(QLabel):
    clicked = Signal(int)
    rightClicked = Signal(int)

    def __init__(self, index, path, size=150, parent=None):
        super().__init__(parent)
        self._index = index
        self._ref = False
        self._work = False
        self.setFixedSize(size + 8, size + 28)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pixmap)
        self.setToolTip(path)
        self.set_selected(False)

    def set_selected(self, active):
        self._ref = active
        self._apply_style()

    def set_work(self, active):
        self._work = active
        self._apply_style()

    def _apply_style(self):
        if getattr(self, "_work", False):
            color, width = "#4aa3ff", "3px"
        elif getattr(self, "_ref", False):
            color, width = "#ffb020", "3px"
        else:
            color, width = "#3a3d44", "2px"
        self.setStyleSheet(
            f"border: {width} solid {color}; border-radius: 6px; padding: 2px; background: #2b2d33;"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._index)
        elif event.button() == Qt.MouseButton.RightButton:
            self.rightClicked.emit(self._index)
        super().mousePressEvent(event)


class ThumbStrip(QScrollArea):
    picked = Signal(int)
    referenceRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFixedHeight(190)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._host = QWidget()
        self._layout = QVBoxLayout(self._host)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._row = None
        self._host.setLayout(self._layout)
        self.setWidget(self._host)
        self._thumbs = []
        self._active = -1

    def set_photos(self, entries):
        if self._row is not None:
            self._row.deleteLater()
            self._row = None
        self._thumbs = []
        self._active = -1
        self._row = QWidget()
        row_layout = QHBoxLayout(self._row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        for index, (_, thumb) in enumerate(entries):
            item = Thumbnail(index, thumb)
            item.clicked.connect(self._on_click)
            item.rightClicked.connect(self.referenceRequested)
            row_layout.addWidget(item)
            self._thumbs.append(item)
        row_layout.addStretch(1)
        self._layout.addWidget(self._row)

    def _on_click(self, index):
        self.set_active(index)
        self.picked.emit(index)

    def set_active(self, index):
        """Подсветка эталонного снимка."""
        self._active = index
        for i, item in enumerate(self._thumbs):
            item.set_selected(i == index)

    def set_work(self, index):
        """Подсветка рабочего снимка: отдельная рамка."""
        for i, item in enumerate(self._thumbs):
            item.set_work(i == index)
