from .compose import Layer

HISTORY_LIMIT = 40


class Document:
    """Слои одного рабочего снимка. Отмена и повтор хранят параметры слоёв."""

    def __init__(self):
        self.layers = []
        self._undo = []
        self._redo = []

    def _snapshot(self):
        return [(layer, layer.params()) for layer in self.layers]

    def _restore(self, snapshot):
        self.layers = []
        for layer, params in snapshot:
            layer.apply_params(params)
            self.layers.append(layer)

    def begin_change(self):
        self._undo.append(self._snapshot())
        del self._undo[:-HISTORY_LIMIT]
        self._redo.clear()

    def add(self, layer: Layer):
        self.begin_change()
        self.layers.append(layer)

    def remove(self, layer: Layer):
        self.begin_change()
        self.layers = [item for item in self.layers if item is not layer]

    def can_undo(self):
        return bool(self._undo)

    def can_redo(self):
        return bool(self._redo)

    def undo(self):
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self):
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    def render(self, base, poisson=True):
        out = base
        for layer in self.layers:
            if layer.visible:
                out, _ = layer.apply(out, poisson)
        return out
