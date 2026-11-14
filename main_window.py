from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
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

from .. import __version__
from ..catalog import Catalog
from ..compose import build_layer, refine_edge
from ..document import Document
from ..face import FaceFinder
from ..imaging import load_rgb, save_jpeg
from ..paths import data_dir
from .canvas import Canvas, draw_icon
from .thumbs import ThumbStrip

TOOL_HINTS = {
    "move": ("Перемещение", "Тянет выделение справа или готовый слой слева."),
    "brush": ("Кисть", "Левая рисует, правая стирает выделение. Alt — добавить, Ctrl — вычесть."),
    "eraser": ("Ластик", "Стирает выделение на эталоне или часть готового слоя слева."),
    "lasso": ("Лассо", "Обведите область. Alt — добавить, Ctrl — вычесть."),
    "zoom": ("Лупа", "Клик приближает, Alt+клик отдаляет."),
    "view": ("Просмотр", "Перемещает вид, не трогая снимок."),
}

HELP_HTML = """
<h2>Как работать</h2>
<ol>
<li><b>Новая серия.</b> Выберите сразу несколько дублей.</li>
<li><b>Рабочее слева.</b> Перетащите снимок из миниатюр в левое окно или кликните по миниатюре.</li>
<li><b>Эталон справа.</b> Клик по миниатюре ставит снимок справа. Правый клик — сразу эталоном.</li>
<li><b>Выделение на эталоне.</b> Кистью (B) или лассо (L) обведите лицо с запасом.</li>
<li><b>Прибавка и вычитание.</b> Alt+рисовать — добавить к выделению, Ctrl+рисовать — вычесть.</li>
<li><b>Перенос.</b> Инструмент «Перемещение» (V), тянете выделение справа налево на рабочее фото.</li>
<li><b>Подгонка.</b> Программа сама выравнивает лицо по точкам и подстраивает цвет и свет.</li>
<li><b>Экспорт.</b> Ctrl+E сохраняет рабочее фото в JPEG 95, исходное разрешение.</li>
</ol>
<h3>Несколько лиц сразу</h3>
<p>Можно обвести ласо двух людей из одной серии. Программа разделит выделение между лицами
и посадит каждое на своё место рабочего снимка. Если пара не нашлась, слой ставится по курсору,
и его можно поправить перетаскиванием слева.</p>
<h3>Горячие клавиши</h3>
<table cellpadding="4">
<tr><td><b>V</b></td><td>Перемещение</td></tr>
<tr><td><b>B</b></td><td>Кисть</td></tr>
<tr><td><b>L</b></td><td>Лассо</td></tr>
<tr><td><b>Z</b></td><td>Лупа</td></tr>
<tr><td><b>H</b></td><td>Просмотр</td></tr>
<tr><td><b>Ъ</b></td><td>Увеличить кисть</td></tr>
<tr><td><b>Х</b></td><td>Уменьшить кисть</td></tr>
<tr><td><b>Ctrl+O</b></td><td>Импорт</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Экспорт</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>Отменить</td></tr>
<tr><td><b>Ctrl+Y</b></td><td>Повторить</td></tr>
<tr><td><b>Ctrl+0</b></td><td>Вписать в окно</td></tr>
<tr><td><b>Ctrl+1</b></td><td>Масштаб 100%</td></tr>
<tr><td><b>Delete</b></td><td>Снять выделение</td></tr>
<tr><td><b>F1</b></td><td>Справка</td></tr>
</table>
<h3>Важно</h3>
<p>Геометрия лица не искажается: перенос, масштаб и поворот в плоскости кадра.
Если голова повёрнута иначе, поправьте «Поворот» и «Масштаб» ползунками.</p>
"""


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка и горячие клавиши")
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
        self.setWindowTitle(f"Групповые портреты {__version__}")
        self.resize(1500, 950)

        root = data_dir()
        self.catalog = Catalog(root / "catalog.sqlite", root / "thumbs")
        self.finder = FaceFinder()
        self._faces = {}

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
        self.work_canvas.layerDragStarted.connect(self._begin_layer_change)
        self.work_canvas.layerDragged.connect(self.drag_layer)
        self.work_canvas.layerErased.connect(self.erase_layer)
        self.ref_canvas.brushChanged.connect(self._sync_brush)
        self.ref_canvas.strokeFinished.connect(self._after_selection)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        for title, widget in (("Рабочий снимок", self.work_canvas), ("Эталон — с него берём выделение", self.ref_canvas)):
            panel = QWidget()
            box = QVBoxLayout(panel)
            box.setContentsMargins(0, 0, 0, 0)
            header = QLabel(f"<b>{title}</b>")
            header.setStyleSheet("padding: 4px 8px;")
            box.addWidget(header)
            box.addWidget(widget)
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
        box.addWidget(QLabel("<b>Слои (лица и части тела)</b>"))
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(150)
        self.layer_list.setToolTip("Список перенесённых фрагментов. Клик — выбрать и править именно его")
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
            (self.mode_add, "add", "Добавлять к текущему выделению (или Alt)"),
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
        self.slider_color = self._slider(form, "Цвет и свет", 0, 100, 80, "Подгонка под рабочий снимок")
        self.slider_angle = self._slider(form, "Поворот", -30, 30, 0, "Поворот слоя, градусы")
        self.slider_scale = self._slider(form, "Масштаб", 70, 140, 100, "Размер слоя, проценты")
        self.slider_brush = self._slider(form, "Кисть", 3, 400, 24, "Размер кисти, пиксели")
        box.addLayout(form)

        self.chk_poisson = QCheckBox("Бесшовное смешивание (Poisson)")
        self.chk_poisson.setChecked(True)
        self.chk_poisson.setToolTip("Выключите, если смешивание даёт ореол на лице")
        self.chk_poisson.stateChanged.connect(lambda: self._render_work())
        box.addWidget(self.chk_poisson)

        self.btn_confirm = QPushButton("✓ Подтвердить перенос")
        self.btn_confirm.setToolTip("Зафиксировать текущий слой и начать следующий")
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
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_actions(self):
        bar = QToolBar("Инструменты")
        bar.setMovable(False)
        bar.setIconSize(QSize(22, 22))
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
            ("Снять выделение", "Delete", self.clear_selection),
            ("Вписать", "Ctrl+0", self.fit_view),
            ("100%", "Ctrl+1", self.zoom_100),
        ):
            action = QAction(title, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(slot)
            bar.addAction(action)

        bar.addSeparator()
        for title, shortcut, slot in (
            ("Экспорт", "Ctrl+E", self.export_image),
            ("Импорт", "Ctrl+O", self.import_photos),
            ("Справка", "F1", lambda: HelpDialog(self).exec()),
        ):
            action = QAction(title, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(slot)
            bar.addAction(action)

        bigger = QAction("Кисть +", self)
        bigger.setShortcut(QKeySequence("Ъ"))
        bigger.triggered.connect(lambda: self.ref_canvas.bump_brush(4))
        self.addAction(bigger)
        smaller = QAction("Кисть -", self)
        smaller.setShortcut(QKeySequence("Х"))
        smaller.triggered.connect(lambda: self.ref_canvas.bump_brush(-4))
        self.addAction(smaller)

    def _set_tool(self, tool):
        self.ref_canvas.set_tool(tool)
        self.work_canvas.set_tool(tool)
        for key, action in self.tool_actions.items():
            action.setChecked(key == tool)
        title, tip = TOOL_HINTS[tool]
        self.hint.setText(f"{title}: {tip}")
        self.work_canvas.layer_drag = self.layer is not None and tool == "move"
        self.ref_canvas.layer_drag = False

    def _after_selection(self):
        """После обводки лассо сразу переходим к переносу, чтобы тянуть не переключаясь."""
        if self.ref_canvas.tool == "lasso" and self.ref_canvas.has_selection():
            self._set_tool("move")
            self.statusBar().showMessage("Выделение готово. Тяните его на рабочий снимок слева", 6000)

    def _sync_brush(self, value):
        self.slider_brush.blockSignals(True)
        self.slider_brush.setValue(value)
        self.slider_brush.blockSignals(False)

    # ---------- каталог ----------
    def refresh_projects(self):
        self.project_list.clear()
        for pid, name, created, count in self.catalog.projects():
            item = QListWidgetItem(f"{name}  ·  {count} фото  ·  {created}")
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
        self._faces.clear()
        self.document = None
        self.layer = None
        self.work_index = None
        self.ref_index = None
        self.thumbs.set_photos(self.entries)
        self.thumbs.set_work(None)
        if self.entries:
            self.set_work(0)
            self.set_reference(0 if len(self.entries) == 1 else 1)
        self.refresh_projects()

    def delete_project(self):
        item = self.project_list.currentItem()
        if item is None:
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        answer = QMessageBox.question(self, "Удалить серию", "Удалить серию из каталога? Файлы на диске останутся.")
        if answer != QMessageBox.StandardButton.Yes:
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
        if key not in self._faces:
            self._faces[key] = self.finder.detect(rgb)
        return self._faces[key]

    def set_work(self, index):
        """Рабочий снимок слева: к нему применяются правки."""
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
        keep = self.work_canvas.has_image()
        self.work_canvas.set_image(self.work_rgb, keep_view=keep)
        self.thumbs.set_work(index)
        self._refresh_layer_ui()
        self._update_status()

    def set_reference(self, index):
        """Эталон справа: с него берём выделение."""
        if not self.entries or not 0 <= index < len(self.entries):
            return
        try:
            self.ref_rgb = load_rgb(self.entries[index][0])
        except OSError as error:
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть снимок:\n{error}")
            return
        self.ref_index = index
        keep = self.ref_canvas.has_image()
        self.ref_canvas.set_image(self.ref_rgb, keep_view=keep)
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
                f"Лиц: {layer.faces}, выровнено автоматически: {layer.aligned}. "
                f"Остальные поправьте перетаскиванием слева и нажмите «Подтвердить перенос»", 8000
            )
        else:
            self.statusBar().showMessage("Слой создан. Проверьте результат и нажмите «Подтвердить перенос»", 6000)

    def _drop_point(self):
        """Точка на рабочем снимке, где отпустили выделение.

        Берём реальную позицию курсора и переводим её в координаты левого окна,
        иначе выделение с эталона переносилось бы по своим координатам.
        """
        from PySide6.QtGui import QCursor

        pos = QCursor.pos()
        local = self.work_canvas.mapFromGlobal(pos)
        if self.work_canvas.rect().contains(local):
            point = self.work_canvas.screen_to_image(local)
            return (point.x(), point.y())
        return None

    def confirm_layer(self):
        """Фиксирует текущий слой: он остаётся в списке, начинается новый."""
        if self.layer is None:
            return
        self.layer_confirmed = True
        self.statusBar().showMessage(f"«{self.layer.name}» закреплён. Выделите следующее лицо на эталоне", 6000)
        self.ref_canvas.clear_mask()
        self._refresh_layer_ui()

    def _select_layer(self, row):
        """Переключение между слоями, как в палитре слоёв."""
        if self.document is None or row < 0 or row >= len(self.document.layers):
            return
        self.layer = self.document.layers[row]
        self.layer_confirmed = True
        self._sync_sliders_from_layer()
        self._refresh_layer_ui()

    def _sync_sliders_from_layer(self):
        layer = self.layer
        if layer is None:
            return
        for slider, value in ((self.slider_feather, layer.feather),
                              (self.slider_color, int(layer.color * 100)),
                              (self.slider_angle, int(layer.angle)),
                              (self.slider_scale, int(layer.scale * 100))):
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
        """Ластик стирает часть готового слоя на рабочем снимке."""
        if self.layer is None:
            return
        self.layer.erase(x, y, radius)
        self._render_work(preview=True)

    def _render_work(self, preview=False):
        if self.document is None:
            return
        poisson = self.chk_poisson.isChecked()
        self.work_canvas.set_image(self.document.render(self.work_rgb, poisson=poisson and not preview), keep_view=True)

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

    def _set_select_mode(self, mode):
        self.select_mode = mode
        for btn, value in ((self.mode_replace, "replace"), (self.mode_add, "add"), (self.mode_sub, "subtract")):
            btn.setChecked(value == mode)
        self.ref_canvas.set_select_mode(mode)

    def _refresh_layer_ui(self):
        layers = self.document.layers if self.document else []
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for layer in reversed(layers):
            mark = "" if getattr(layer, "erased", None) else ""
            item = QListWidgetItem(f"{layer.name}{mark}")
            self.layer_list.addItem(item)
            if layer is self.layer:
                self.layer_list.setCurrentRow(self.layer_list.count() - 1)
        self.layer_list.blockSignals(False)

        if self.layer is None:
            self.layer_info.setText("Слой не создан. Выделите область справа и перетащите на рабочий снимок.")
        else:
            self.layer_info.setText(
                f"Активный слой: {self.layer.name}. Лиц: {self.layer.faces}, выровнено: {self.layer.aligned}."
            )
        self.btn_confirm.setEnabled(self.layer is not None and not getattr(self, "layer_confirmed", False))
        draggable = self.layer is not None and self.work_canvas.tool in ("move", "eraser")
        self.work_canvas.layer_drag = draggable

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
                "Выделение есть. Тяните его инструментом «Перемещение» (V) с эталона на рабочее фото."
            )
        elif self.document is not None:
            self.statusBar().showMessage(f"Слоёв на рабочем: {len(self.document.layers)}")

    def closeEvent(self, event):
        self.finder.close()
        super().closeEvent(event)
