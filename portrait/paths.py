import os
import sys
from pathlib import Path

APP_NAME = "GroupPortrait"
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def resource(relative):
    return str(ROOT / relative)


def data_dir():
    base = os.environ.get("APPDATA") or str(Path.home() / ".local" / "share")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path
