import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(sys.executable).parent))

from PySide6.QtWidgets import QApplication

from portrait.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Групповые портреты")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
