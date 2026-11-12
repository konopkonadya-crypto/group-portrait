import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portrait.catalog import Catalog


@pytest.fixture
def photo_set(tmp_path):
    paths = []
    for i, shade in enumerate((60, 120, 200)):
        img = np.full((180, 240, 3), shade, np.uint8)
        path = tmp_path / f"shot{i}.jpg"
        cv2.imwrite(str(path), img)
        paths.append(path)
    return paths


def test_create_list_delete(tmp_path, photo_set):
    cat = Catalog(tmp_path / "db.sqlite", tmp_path / "thumbs")
    pid = cat.create("Семья, июль", photo_set)

    listed = cat.projects()
    assert len(listed) == 1
    assert listed[0][1] == "Семья, июль"
    assert listed[0][3] == 3

    photos = cat.photos(pid)
    assert [Path(p).name for p, _ in photos] == ["shot0.jpg", "shot1.jpg", "shot2.jpg"]
    assert all(Path(t).exists() for _, t in photos)

    assert cat.projects("июль")[0][0] == pid
    assert cat.projects("август") == []

    cat.delete(pid)
    assert cat.projects() == []
    assert not (tmp_path / "thumbs" / str(pid)).exists()


def test_broken_file_does_not_leave_project(tmp_path, photo_set):
    cat = Catalog(tmp_path / "db.sqlite", tmp_path / "thumbs")
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(OSError):
        cat.create("Битая серия", [photo_set[0], bad])
    assert cat.projects() == []
