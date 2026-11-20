"""Small file helpers shared by the runner: atomic JSON writes and append-only JSONL."""
from __future__ import annotations

import json
import os
from pathlib import Path


def dumps_line(obj: object) -> str:
    """One compact JSON line (no trailing newline). Rejects NaN/Infinity so rows stay valid JSON."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to a temp file next to ``path`` and rename it over ``path``."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: object) -> None:
    atomic_write_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    """Return the parseable JSON-object lines of ``path`` (empty if missing).

    A line that cannot be parsed (for example a write cut short by a crash) is skipped.
    A final line without a newline counts only if it parses, because a cut-off JSON
    object can never parse.
    """
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    rows: list[dict] = []
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:  # includes UnicodeDecodeError
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


class JsonlAppender:
    """Append-only JSONL writer. Each line is flushed and fsynced; existing bytes are never modified.

    If the file ends without a newline (a write cut short by a crash), the next line is
    started on a fresh line, so the fragment can never merge with a new record.
    Not thread-safe: the runner calls it from one thread only.
    """

    def __init__(self, path: Path) -> None:
        self._f = open(path, "ab")
        if self._f.tell() > 0:
            with open(path, "rb") as r:
                r.seek(-1, os.SEEK_END)
                needs_newline = r.read(1) != b"\n"
            if needs_newline:
                self._write(b"\n")

    def _write(self, data: bytes) -> None:
        self._f.write(data)
        self._f.flush()
        os.fsync(self._f.fileno())

    def append(self, line: str) -> None:
        self._write(line.encode("utf-8") + b"\n")

    def close(self) -> None:
        self._f.close()
