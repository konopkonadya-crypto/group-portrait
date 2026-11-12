import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from .imaging import make_thumbnail

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    thumb TEXT NOT NULL,
    position INTEGER NOT NULL
);
"""


class Catalog:
    def __init__(self, db_path, thumb_dir):
        self._db = sqlite3.connect(db_path)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._thumb_dir = Path(thumb_dir)

    def projects(self, search=""):
        like = f"%{search.strip()}%"
        return self._db.execute(
            """
            SELECT p.id, p.name, p.created, COUNT(ph.id)
            FROM projects p LEFT JOIN photos ph ON ph.project_id = p.id
            WHERE p.name LIKE ? OR p.created LIKE ?
            GROUP BY p.id ORDER BY p.created DESC
            """,
            (like, like),
        ).fetchall()

    def photos(self, project_id):
        return self._db.execute(
            "SELECT path, thumb FROM photos WHERE project_id = ? ORDER BY position",
            (project_id,),
        ).fetchall()

    def create(self, name, paths):
        created = datetime.now().strftime("%Y-%m-%d %H:%M")
        project_id = self._db.execute(
            "INSERT INTO projects(name, created) VALUES (?, ?)", (name, created)
        ).lastrowid
        thumb_dir = self._thumb_dir / str(project_id)
        thumb_dir.mkdir(parents=True, exist_ok=True)
        try:
            for position, path in enumerate(paths):
                thumb = thumb_dir / f"{position:03d}.jpg"
                make_thumbnail(path, thumb)
                self._db.execute(
                    "INSERT INTO photos(project_id, path, thumb, position) VALUES (?, ?, ?, ?)",
                    (project_id, str(path), str(thumb), position),
                )
        except OSError:
            self._db.rollback()
            shutil.rmtree(thumb_dir, ignore_errors=True)
            raise
        self._db.commit()
        return project_id

    def add_photos(self, project_id, paths):
        thumb_dir = self._thumb_dir / str(project_id)
        thumb_dir.mkdir(parents=True, exist_ok=True)
        start = self._db.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM photos WHERE project_id = ?", (project_id,)
        ).fetchone()[0]
        try:
            for step, path in enumerate(paths):
                thumb = thumb_dir / f"{start + step:03d}.jpg"
                make_thumbnail(path, thumb)
                self._db.execute(
                    "INSERT INTO photos(project_id, path, thumb, position) VALUES (?, ?, ?, ?)",
                    (project_id, str(path), str(thumb), start + step),
                )
        except OSError:
            self._db.rollback()
            raise
        self._db.commit()
        return start

    def delete(self, project_id):
        self._db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        self._db.commit()
        shutil.rmtree(self._thumb_dir / str(project_id), ignore_errors=True)
