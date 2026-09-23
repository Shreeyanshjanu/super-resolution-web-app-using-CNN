from __future__ import annotations
import json
import re
import shutil
from pathlib import Path

ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def object_directory(root: Path, identifier: str) -> Path:
    if not ID_PATTERN.fullmatch(identifier):
        raise ValueError("Invalid identifier.")
    root = root.resolve()
    target = (root / identifier).resolve()
    if target.parent != root:
        raise ValueError("Path is outside the storage directory.")
    return target


def remove_object(root: Path, identifier: str):
    # Resolve and constrain the actual target before recursive removal.
    target = object_directory(root, identifier)
    if target.is_dir():
        shutil.rmtree(target)


class SceneStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, scene_id):
        return object_directory(self.directory, scene_id)

    def save(self, scene):
        directory = self.path(scene["scene_id"])
        temporary = directory / "metadata.json.tmp"
        temporary.write_text(json.dumps(scene), encoding="utf-8")
        temporary.replace(directory / "metadata.json")

    def get(self, scene_id):
        try:
            path = self.path(scene_id) / "metadata.json"
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except ValueError:
            return None
