"""status.json: atomic progress file read by ``guardbench-run status`` (no LLM involved)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from guardbench_classifier.runner._files import atomic_write_json
from guardbench_classifier.runner.manifest import now_utc


class StatusFile:
    """Writer for one invocation. Every update replaces status.json atomically."""

    def __init__(self, run_dir: Path, total: int) -> None:
        self._path = run_dir / "status.json"
        self._started_at = now_utc()
        self.total = total
        self.done = 0
        self.error_rows = 0

    def write(self, state: str) -> None:
        atomic_write_json(
            self._path,
            {
                "state": state,
                "total": self.total,
                "done": self.done,
                "error_rows": self.error_rows,
                "pid": os.getpid(),
                "started_at": self._started_at,
                "updated_at": now_utc(),
            },
        )


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def read_status(run_dir: Path) -> dict:
    """status.json plus whether its pid is alive. Raises FileNotFoundError if there is no status."""
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    status["pid_alive"] = pid_alive(int(status["pid"]))
    return status
